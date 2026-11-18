# Vectorised (batched) environment: auto-reset step over many envs, on CPU
# (parallelize over env chunks, env-major layout) or GPU (one thread per env,
# field-major layout).

from std.memory import Pointer
from max.algorithm import parallelize
from max.gpu import global_idx

from .constants import *
from .state import *
from .rng import Rng, SplitMix, seed_key
from .worldgen import generate_world, generate_world_local
from .game import step, is_game_over
from .obs import write_obs


struct Buffers(Copyable):
    """Raw pointers to a batch's state; the caller owns the memory."""

    var ints: Pointer[Int32, MutUntrackedOrigin]  # n * NUM_I
    var floats: Pointer[Float32, MutUntrackedOrigin]  # n * NUM_F
    var map: Pointer[UInt8, MutUntrackedOrigin]  # n * MAP_CELLS
    var mob_map: Pointer[UInt8, MutUntrackedOrigin]  # n * MAP_CELLS
    var rng: Pointer[UInt64, MutUntrackedOrigin]  # n * 2 (key, counter)
    var obs: Pointer[Float32, MutUntrackedOrigin]  # n * OBS_SIZE (env-major)
    var reward: Pointer[Float32, MutUntrackedOrigin]  # n
    var done: Pointer[UInt8, MutUntrackedOrigin]  # n
    var n: Int

    def __init__(
        out self,
        ints: Pointer[Int32, MutUntrackedOrigin],
        floats: Pointer[Float32, MutUntrackedOrigin],
        map: Pointer[UInt8, MutUntrackedOrigin],
        mob_map: Pointer[UInt8, MutUntrackedOrigin],
        rng: Pointer[UInt64, MutUntrackedOrigin],
        obs: Pointer[Float32, MutUntrackedOrigin],
        reward: Pointer[Float32, MutUntrackedOrigin],
        done: Pointer[UInt8, MutUntrackedOrigin],
        n: Int,
    ):
        self.ints = ints
        self.floats = floats
        self.map = map
        self.mob_map = mob_map
        self.rng = rng
        self.obs = obs
        self.reward = reward
        self.done = done
        self.n = n

    @inline(.always)
    def view[FM: Bool](self, e: Int) -> EnvView[FM]:
        return EnvView[FM](self.ints, self.floats, self.map, self.mob_map, self.n, e)

    @inline(.always)
    def load_rng[FM: Bool](self, e: Int) -> SplitMix:
        comptime if FM:
            return SplitMix(
                self.rng[unsafe_offset=e], self.rng[unsafe_offset = self.n + e]
            )
        else:
            return SplitMix(
                self.rng[unsafe_offset = 2 * e], self.rng[unsafe_offset = 2 * e + 1]
            )

    @inline(.always)
    def store_rng[FM: Bool](self, e: Int, rng: SplitMix):
        comptime if FM:
            self.rng[unsafe_offset = self.n + e] = rng.ctr
        else:
            self.rng[unsafe_offset = 2 * e + 1] = rng.ctr

    @inline(.always)
    def obs_ptr(self, e: Int) -> Pointer[Float32, MutUntrackedOrigin]:
        return self.obs.unsafe_offset(e * OBS_SIZE)


@inline(.always)
def _generate[FM: Bool](env: EnvView[FM], mut rng: SplitMix):
    # FM (field-major) is the GPU layout; GPU threads cannot heap-allocate.
    comptime if FM:
        generate_world_local(env, rng)
    else:
        generate_world(env, rng)


@inline(.always)
def reset_one[FM: Bool](b: Buffers, e: Int, seed: UInt64):
    var key = seed_key(seed, UInt64(e))
    comptime if FM:
        b.rng[unsafe_offset=e] = key
    else:
        b.rng[unsafe_offset = 2 * e] = key
    var rng = SplitMix(key, 0)
    var env = b.view[FM](e)
    _generate[FM](env, rng)
    write_obs(env, b.obs_ptr(e))
    b.store_rng[FM](e, rng)


@inline(.always)
def step_one[FM: Bool](b: Buffers, e: Int, action: Int):
    var rng = b.load_rng[FM](e)
    var env = b.view[FM](e)
    var r = step(env, action, rng)
    var d = is_game_over(env)
    if d:
        _generate[FM](env, rng)
    write_obs(env, b.obs_ptr(e))
    b.reward[unsafe_offset=e] = r
    b.done[unsafe_offset=e] = UInt8(1) if d else UInt8(0)
    b.store_rng[FM](e, rng)


@inline(.always)
def random_action[FM: Bool](b: Buffers, e: Int) -> Int:
    """Uniform random action from the env's own stream (benchmark policy)."""
    var rng = b.load_rng[FM](e)
    var a = rng.choice(NUM_ACTIONS)
    b.store_rng[FM](e, rng)
    return a


# ---- CPU -----------------------------------------------------------------
def _chunks(n: Int, workers: Int) -> Int:
    # ~32 envs (~30us of work) per chunk amortises the ~70us pool dispatch.
    return max(1, min(n // 32, workers * 4))



def reset_cpu(b: Buffers, seed: UInt64, workers: Int):
    var n = b.n
    var nc = _chunks(n, workers)

    def work(c: Int) {imm}:
        for e in range(c * n // nc, (c + 1) * n // nc):
            reset_one[False](b, e, seed)

    if nc == 1:  # skip the thread pool entirely
        work(0)
    else:
        parallelize(work, nc)


def step_cpu(
    b: Buffers, actions: Pointer[Int32, MutUntrackedOrigin], workers: Int
):
    var n = b.n
    var nc = _chunks(n, workers)

    def work(c: Int) {imm}:
        for e in range(c * n // nc, (c + 1) * n // nc):
            step_one[False](b, e, Int(actions[unsafe_offset=e]))

    if nc == 1:  # skip the thread pool entirely
        work(0)
    else:
        parallelize(work, nc)


def step_random_cpu(b: Buffers, workers: Int):
    var n = b.n
    var nc = _chunks(n, workers)

    def work(c: Int) {imm}:
        for e in range(c * n // nc, (c + 1) * n // nc):
            step_one[False](b, e, random_action[False](b, e))

    if nc == 1:  # skip the thread pool entirely
        work(0)
    else:
        parallelize(work, nc)


# ---- GPU kernels (one thread per env, field-major state) -----------------
# Kernel arguments must be device-passable, so the buffers are passed
# individually and n as Int32.
comptime I32P = Pointer[Int32, MutUntrackedOrigin]
comptime F32P = Pointer[Float32, MutUntrackedOrigin]
comptime U8P = Pointer[UInt8, MutUntrackedOrigin]
comptime U64P = Pointer[UInt64, MutUntrackedOrigin]


def reset_kernel(
    ints: I32P, floats: F32P, map: U8P, mob_map: U8P, rng: U64P, obs: F32P,
    reward: F32P, done: U8P, n: Int32, seed: UInt64,
):
    var e = global_idx.x
    if e < Int(n):
        var b = Buffers(ints, floats, map, mob_map, rng, obs, reward, done, Int(n))
        reset_one[True](b, e, seed)


def step_random_kernel(
    ints: I32P, floats: F32P, map: U8P, mob_map: U8P, rng: U64P, obs: F32P,
    reward: F32P, done: U8P, n: Int32,
):
    var e = global_idx.x
    if e < Int(n):
        var b = Buffers(ints, floats, map, mob_map, rng, obs, reward, done, Int(n))
        step_one[True](b, e, random_action[True](b, e))


def step_kernel(
    ints: I32P, floats: F32P, map: U8P, mob_map: U8P, rng: U64P, obs: F32P,
    reward: F32P, done: U8P, n: Int32, actions: I32P,
):
    var e = global_idx.x
    if e < Int(n):
        var b = Buffers(ints, floats, map, mob_map, rng, obs, reward, done, Int(n))
        step_one[True](b, e, Int(actions[unsafe_offset=e]))

# The native RNG is per-env and layout-independent, so the CPU (env-major)
# and GPU (field-major) batched paths must produce identical trajectories from
# the same seed. Steps both for K steps and compares every state field, the
# observation, reward and done flag.
#
#   mojo run -I . tests/test_gpu_cpu.mojo [n] [steps] [naive]

from std.memory import Pointer
from std.sys import argv, num_performance_cores
from max.gpu.host import DeviceContext

from craftax_mojo.constants import *
from craftax_mojo.state import NUM_I, NUM_F
from craftax_mojo import vec
from craftax_mojo.gpu_host import GpuBatch


@inline(.always)
def _p[T: AnyType](mut l: List[T]) -> Pointer[T, MutUntrackedOrigin]:
    return Pointer[T, MutUntrackedOrigin](unsafe_from_address=Int(l.unsafe_ptr()))


def main() raises:
    var args = argv()
    var n = Int(String(args[1])) if len(args) > 1 else 1024
    var steps = Int(String(args[2])) if len(args) > 2 else 500
    var naive = len(args) > 3 and String(args[3]) == "naive"
    var workers = num_performance_cores()

    # CPU, env-major
    var ints = List[Int32](length=n * NUM_I, fill=0)
    var floats = List[Float32](length=n * NUM_F, fill=0)
    var map = List[UInt8](length=n * MAP_CELLS, fill=0)
    var mob = List[UInt8](length=n * MAP_CELLS, fill=0)
    var rng = List[UInt64](length=n * 2, fill=0)
    var obs = List[Float32](length=n * OBS_SIZE, fill=0)
    var reward = List[Float32](length=n, fill=0)
    var done = List[UInt8](length=n, fill=0)
    var b = vec.Buffers(
        _p(ints), _p(floats), _p(map), _p(mob), _p(rng), _p(obs), _p(reward),
        _p(done), n,
    )

    # GPU, field-major
    var ctx = DeviceContext()
    var g = GpuBatch(ctx, n)
    vec.reset_cpu(b, 7, workers)
    g.reset(7)

    var reward_mismatch = 0
    var done_mismatch = 0
    var episodes = 0
    for _ in range(steps):
        vec.step_random_cpu(b, workers)
        if naive:
            g.step_random_naive()
        else:
            g.step_random()
        with g.reward.map_to_host() as gr, g.done.map_to_host() as gd:
            for e in range(n):
                if gr[e] != reward[e]:
                    reward_mismatch += 1
                if gd[e] != done[e]:
                    done_mismatch += 1
                episodes += Int(done[e])

    var int_mismatch = 0
    var float_mismatch = 0
    var map_mismatch = 0
    var obs_mismatch = 0
    with g.ints.map_to_host() as gi, g.floats.map_to_host() as gf:
        for e in range(n):
            for f in range(NUM_I):
                if gi[f * n + e] != ints[e * NUM_I + f]:
                    int_mismatch += 1
            for f in range(NUM_F):
                if abs(gf[f * n + e] - floats[e * NUM_F + f]) > 1e-5:
                    float_mismatch += 1
    with g.map.map_to_host() as gm, g.mob.map_to_host() as gb:
        for e in range(n):
            for c in range(MAP_CELLS):
                if gm[e * MAP_CELLS + c] != map[e * MAP_CELLS + c]:
                    map_mismatch += 1
                if gb[e * MAP_CELLS + c] != mob[e * MAP_CELLS + c]:
                    map_mismatch += 1
    with g.obs.map_to_host() as go:
        for i in range(n * OBS_SIZE):
            if abs(go[i] - obs[i]) > 1e-5:
                obs_mismatch += 1

    print(
        "naive" if naive else "deferred-reset", "envs", n, "steps", steps, "episodes finished", episodes,
        "| mismatches: reward", reward_mismatch, "done", done_mismatch,
        "ints", int_mismatch, "floats", float_mismatch, "map", map_mismatch,
        "obs", obs_mismatch,
    )
    var ok = (
        reward_mismatch + done_mismatch + int_mismatch + float_mismatch
        + map_mismatch + obs_mismatch
    ) == 0
    print("PASS" if ok else "FAIL")
    _ = ints^
    _ = floats^
    _ = map^
    _ = mob^
    _ = rng^
    _ = obs^
    _ = reward^
    _ = done^

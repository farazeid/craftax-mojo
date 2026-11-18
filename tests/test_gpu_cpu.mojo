# The native RNG is per-env and layout-independent, so the CPU (env-major)
# and GPU (field-major) batched paths must produce identical trajectories from
# the same seed. Steps both for K steps and compares every state field, the
# observation, reward and done flag.
#
#   mojo run -I . tests/test_gpu_cpu.mojo [n] [steps]

from std.memory import Pointer
from std.sys import argv, num_performance_cores
from max.gpu.host import DeviceContext

from craftax_mojo.constants import *
from craftax_mojo.state import NUM_I, NUM_F
from craftax_mojo import vec


@inline(.always)
def _p[T: AnyType](mut l: List[T]) -> Pointer[T, MutUntrackedOrigin]:
    return Pointer[T, MutUntrackedOrigin](unsafe_from_address=Int(l.unsafe_ptr()))


def main() raises:
    var args = argv()
    var n = Int(String(args[1])) if len(args) > 1 else 1024
    var steps = Int(String(args[2])) if len(args) > 2 else 500
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
    var g_ints = ctx.enqueue_create_buffer[DType.int32](n * NUM_I)
    var g_floats = ctx.enqueue_create_buffer[DType.float32](n * NUM_F)
    var g_map = ctx.enqueue_create_buffer[DType.uint8](n * MAP_CELLS)
    var g_mob = ctx.enqueue_create_buffer[DType.uint8](n * MAP_CELLS)
    var g_rng = ctx.enqueue_create_buffer[DType.uint64](n * 2)
    var g_obs = ctx.enqueue_create_buffer[DType.float32](n * OBS_SIZE)
    var g_reward = ctx.enqueue_create_buffer[DType.float32](n)
    var g_done = ctx.enqueue_create_buffer[DType.uint8](n)
    comptime block = 128
    var grid = (n + block - 1) // block

    vec.reset_cpu(b, 7, workers)
    ctx.enqueue_function[vec.reset_kernel](
        g_ints, g_floats, g_map, g_mob, g_rng, g_obs, g_reward, g_done,
        Int32(n), UInt64(7), grid_dim=grid, block_dim=block,
    )

    var reward_mismatch = 0
    var done_mismatch = 0
    var episodes = 0
    for _ in range(steps):
        vec.step_random_cpu(b, workers)
        ctx.enqueue_function[vec.step_random_kernel](
            g_ints, g_floats, g_map, g_mob, g_rng, g_obs, g_reward, g_done,
            Int32(n), grid_dim=grid, block_dim=block,
        )
        with g_reward.map_to_host() as gr, g_done.map_to_host() as gd:
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
    with g_ints.map_to_host() as gi, g_floats.map_to_host() as gf:
        for e in range(n):
            for f in range(NUM_I):
                if gi[f * n + e] != ints[e * NUM_I + f]:
                    int_mismatch += 1
            for f in range(NUM_F):
                if abs(gf[f * n + e] - floats[e * NUM_F + f]) > 1e-5:
                    float_mismatch += 1
    with g_map.map_to_host() as gm, g_mob.map_to_host() as gb:
        for e in range(n):
            for c in range(MAP_CELLS):
                if gm[c * n + e] != map[e * MAP_CELLS + c]:
                    map_mismatch += 1
                if gb[c * n + e] != mob[e * MAP_CELLS + c]:
                    map_mismatch += 1
    with g_obs.map_to_host() as go:
        for i in range(n * OBS_SIZE):
            if abs(go[i] - obs[i]) > 1e-5:
                obs_mismatch += 1

    print(
        "envs", n, "steps", steps, "episodes finished", episodes,
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

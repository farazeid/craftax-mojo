# Throughput benchmark for the Mojo port: random-policy, auto-reset steps per
# second, including observation writes, for a sweep of batch sizes.
#
#   bench_mojo cpu <steps> <n1,n2,...> [workers]
#   bench_mojo gpu|gpu-naive <steps> <n1,n2,...> [block_size]
#
# Prints one JSON object per batch size.

from std.memory import Pointer
from std.sys import argv, num_performance_cores
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from craftax_mojo.constants import *
from craftax_mojo.state import NUM_I, NUM_F
from craftax_mojo import vec
from craftax_mojo.gpu_host import GpuBatch

comptime SEED: UInt64 = 0
comptime WARMUP = 5


def _list_of_ints(s: String) raises -> List[Int]:
    var out = List[Int]()
    for part in s.split(","):
        out.append(Int(String(part)))
    return out^


def _emit(device: String, name: String, n: Int, steps: Int, secs: Float64, extra: String):
    var sps = Float64(n * steps) / secs
    print(
        '{"impl": "mojo", "device": "'
        + device
        + '", "device_name": "'
        + name
        + '", "num_envs": '
        + String(n)
        + ', "steps": '
        + String(steps)
        + ', "seconds": '
        + String(secs)
        + ', "steps_per_second": '
        + String(sps)
        + extra
        + "}"
    )


@inline(.always)
def _p[T: AnyType](mut l: List[T]) -> Pointer[T, MutUntrackedOrigin]:
    return Pointer[T, MutUntrackedOrigin](unsafe_from_address=Int(l.unsafe_ptr()))


def bench_cpu(steps: Int, sizes: List[Int], workers: Int) raises:
    for n in sizes:
        var ints = List[Int32](length=n * NUM_I, fill=0)
        var floats = List[Float32](length=n * NUM_F, fill=0)
        var map = List[UInt8](length=n * MAP_CELLS, fill=0)
        var mob = List[UInt8](length=n * MAP_CELLS, fill=0)
        var rng = List[UInt64](length=n * 2, fill=0)
        var obs = List[Float32](length=n * OBS_SIZE, fill=0)
        var reward = List[Float32](length=n, fill=0)
        var done = List[UInt8](length=n, fill=0)
        var b = vec.Buffers(
            _p(ints), _p(floats), _p(map), _p(mob), _p(rng), _p(obs),
            _p(reward), _p(done), n,
        )
        vec.reset_cpu(b, SEED, workers)
        for _ in range(WARMUP):
            vec.step_random_cpu(b, workers)
        var dones = 0
        var t0 = perf_counter_ns()
        for _ in range(steps):
            vec.step_random_cpu(b, workers)
        var t1 = perf_counter_ns()
        for e in range(n):
            dones += Int(done[e])
        _emit(
            "cpu", "Apple M-series CPU", n, steps, Float64(t1 - t0) * 1e-9,
            ', "workers": ' + String(workers),
        )
        _ = ints^
        _ = floats^
        _ = map^
        _ = mob^
        _ = rng^
        _ = obs^
        _ = reward^
        _ = done^


def bench_gpu(steps: Int, sizes: List[Int], block: Int, naive: Bool) raises:
    var ctx = DeviceContext()
    var name = String(ctx.name())
    for n in sizes:
        var g = GpuBatch(ctx, n, block)
        g.reset(SEED)
        for _ in range(WARMUP):
            if naive:
                g.step_random_naive()
            else:
                g.step_random()
        g.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(steps):
            if naive:
                g.step_random_naive()
            else:
                g.step_random()
        g.synchronize()
        var t1 = perf_counter_ns()
        _emit(
            "gpu-naive" if naive else "gpu", name, n, steps,
            Float64(t1 - t0) * 1e-9, ', "block": ' + String(block),
        )


def main() raises:
    var args = argv()
    if len(args) < 4:
        print("usage: bench_mojo <cpu|gpu> <steps> <n1,n2,...> [workers|block]")
        return
    var device = String(args[1])
    var steps = Int(String(args[2]))
    var sizes = _list_of_ints(String(args[3]))
    if device == "cpu":
        var workers = Int(String(args[4])) if len(args) > 4 else num_performance_cores()
        bench_cpu(steps, sizes, workers)
    else:
        var block = Int(String(args[4])) if len(args) > 4 else 128
        bench_gpu(steps, sizes, block, device == "gpu-naive")

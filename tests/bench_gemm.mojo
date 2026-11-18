# Times the GEMMs of one PPO minibatch update (default 4096 rows, 1345 -> 512 -> 512
# -> 512 -> 17|1, actor + critic, forward + backward) through nn.Gemm, to
# separate matmul time from everything else in the update.
#
#   mojo run -I . tests/bench_gemm.mojo [tf32 0|1] [rows]

from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from craftax_mojo.nn import Gemm, F32P

comptime D = 1345
comptime W = 512


def main() raises:
    var args = argv()
    var tf32 = len(args) < 2 or String(args[1]) != "0"
    var ROWS = Int(String(args[2])) if len(args) > 2 else 4096
    var ctx = DeviceContext()
    var gemm = Gemm(ctx, tf32)
    var x = ctx.enqueue_create_buffer[DType.float32](ROWS * D)
    var h = ctx.enqueue_create_buffer[DType.float32](ROWS * W)
    var h2 = ctx.enqueue_create_buffer[DType.float32](ROWS * W)
    var w1 = ctx.enqueue_create_buffer[DType.float32](D * W)
    var w2 = ctx.enqueue_create_buffer[DType.float32](W * W)
    var o = ctx.enqueue_create_buffer[DType.float32](ROWS * 17)
    var w4 = ctx.enqueue_create_buffer[DType.float32](W * 17)
    for b in [x, h, h2, w1, w2, o, w4]:
        b.enqueue_fill(0.01)

    @parameter
    def p(b: type_of(x)) -> F32P:
        return F32P(unsafe_from_address=Int(b.unsafe_ptr()))

    @parameter
    def one_update() raises:
        for _ in range(2):  # actor, critic
            # forward
            gemm(ctx, p(h), p(x), p(w1), ROWS, W, D)
            gemm(ctx, p(h2), p(h), p(w2), ROWS, W, W)
            gemm(ctx, p(h), p(h2), p(w2), ROWS, W, W)
            gemm(ctx, p(o), p(h), p(w4), ROWS, 17, W)
            # backward: dW and dH per layer
            gemm(ctx, p(w4), p(h), p(o), W, 17, ROWS, trans_a=True)
            gemm(ctx, p(h2), p(o), p(w4), ROWS, W, 17, trans_b=True)
            gemm(ctx, p(w2), p(h), p(h2), W, W, ROWS, trans_a=True)
            gemm(ctx, p(h), p(h2), p(w2), ROWS, W, W, trans_b=True)
            gemm(ctx, p(w2), p(h), p(h2), W, W, ROWS, trans_a=True)
            gemm(ctx, p(h2), p(h), p(w2), ROWS, W, W, trans_b=True)
            gemm(ctx, p(w1), p(x), p(h2), D, W, ROWS, trans_a=True)

    for _ in range(5):
        one_update()
    ctx.synchronize()
    comptime REPS = 64
    var t0 = perf_counter_ns()
    for _ in range(REPS):
        one_update()
    var t1 = perf_counter_ns()
    ctx.synchronize()
    var t2 = perf_counter_ns()
    var flops = 2.0 * 3.0 * Float64(ROWS) * Float64(D * W + 2 * W * W + W * 17) * 2.0
    var secs = Float64(t2 - t0) * 1e-9 / REPS
    print(
        ctx.name(), "tf32" if tf32 else "fp32", "| GEMMs per minibatch update:",
        secs * 1e3, "ms (host enqueue", Float64(t1 - t0) * 1e-6 / REPS, "ms),",
        flops / secs * 1e-12, "TFLOP/s | rows", ROWS,
    )

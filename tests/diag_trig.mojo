# Diagnostic: do GPU and CPU sin/cos agree bitwise? Evaluates cos/sin(2*pi*u)
# for the same uniforms on both devices (library or det_sincos) and counts
# differences; also reports det_sincos's distance from the CPU libm.
from std.math import cos, sin
from std.memory import Pointer
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from craftax_mojo.rng import SplitMix
from craftax_mojo.worldgen import TWO_PI, det_sincos

comptime N = 1 << 22
comptime DET = True  # False: library cos/sin

def trig_kernel(dst: Pointer[Float32, MutUntrackedOrigin]):
    var i = global_idx.x
    if i < N:
        var r = SplitMix(42, 0)
        var a = TWO_PI * r.at(0, i)
        comptime if DET:
            var cs = det_sincos(a)
            dst[unsafe_offset = 2 * i] = cs[0]
            dst[unsafe_offset = 2 * i + 1] = cs[1]
        else:
            dst[unsafe_offset = 2 * i] = cos(a)
            dst[unsafe_offset = 2 * i + 1] = sin(a)

def main() raises:
    var ctx = DeviceContext()
    var buf = ctx.enqueue_create_buffer[DType.float32](2 * N)
    ctx.enqueue_function[trig_kernel](buf, grid_dim=N // 256, block_dim=256)
    var r = SplitMix(42, 0)
    var diff = 0
    var maxd: Float32 = 0
    var libm_err: Float32 = 0
    with buf.map_to_host() as h:
        for i in range(N):
            var a = TWO_PI * r.at(0, i)
            var c: Float32
            var sn: Float32
            comptime if DET:
                var cs = det_sincos(a)
                c = cs[0]
                sn = cs[1]
            else:
                c = cos(a)
                sn = sin(a)
            var dc = abs(h[2 * i] - c)
            var ds = abs(h[2 * i + 1] - sn)
            libm_err = max(libm_err, max(abs(c - cos(a)), abs(sn - sin(a))))
            if dc != 0 or ds != 0:
                diff += 1
                maxd = max(maxd, max(dc, ds))
    print(
        ctx.name(), "det" if DET else "libm", "angles:", N, "GPU!=CPU:", diff,
        "max abs diff:", maxd, "| max |det - CPU libm|:", libm_err,
    )

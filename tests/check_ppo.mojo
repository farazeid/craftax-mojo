# One PPO update and one GAE pass on inputs written by
# parity/test_ppo_update.py; writes the results back for comparison with
# jax.grad + optax.
#
#   check_ppo <dir> <rows> <lr> <T> <N>
#
# Reads  params.bin, obs.bin, act.bin (int32), logp.bin, adv.bin, ret.bin,
#        values.bin, rewards.bin, dones.bin (uint8), last_value.bin
# Writes grads.bin, params_after.bin, stats.bin, gae_adv.bin, gae_ret.bin

from std.memory import Pointer
from std.sys import argv, size_of
from max.gpu.host import DeviceContext, DeviceBuffer

from craftax_mojo.constants import OBS_SIZE, NUM_ACTIONS
from craftax_mojo.ppo import (
    Config, Nets, Optimizer, Minibatch, Profiler, update_minibatch, fp, ip, up,
)
from craftax_mojo import nn
from craftax_mojo.nn import Gemm, grid_for, BLOCK


def read_bytes(path: String) raises -> List[UInt8]:
    with open(path, "r") as f:
        return f.read_bytes()


def upload[dt: DType](ctx: DeviceContext, path: String) raises -> DeviceBuffer[dt]:
    var bytes = read_bytes(path)
    var n = len(bytes) // size_of[Scalar[dt]]()
    var buf = ctx.enqueue_create_buffer[dt](n)
    with buf.map_to_host() as h:
        var src = bytes.unsafe_ptr().unsafe_bitcast[Scalar[dt]]()
        for i in range(n):
            h[i] = src[unsafe_offset=i]
    return buf^


def download(buf: DeviceBuffer[DType.float32], n: Int, path: String) raises:
    var bytes = List[UInt8](length=4 * n, fill=0)
    with buf.map_to_host() as h:
        var dst = bytes.unsafe_ptr().unsafe_bitcast[Float32]()
        for i in range(n):
            dst[unsafe_offset=i] = h[i]
    with open(path, "w") as f:
        f.write_bytes(bytes)


def main() raises:
    var args = argv()
    var dir = String(args[1])
    var rows = Int(String(args[2]))
    var lr = Float32(Float64(String(args[3])))
    var T = Int(String(args[4]))
    var N = Int(String(args[5]))
    var ctx = DeviceContext()
    var cfg = Config()
    cfg.tf32 = False
    var gemm = Gemm(ctx, False)

    var nets = Nets(OBS_SIZE)
    var params_bytes = read_bytes(dir + "/params.bin")
    var host = List[Float32](length=len(params_bytes) // 4, fill=0)
    var src = params_bytes.unsafe_ptr().unsafe_bitcast[Float32]()
    for i in range(len(host)):
        host[i] = src[unsafe_offset=i]
    var opt = Optimizer(ctx, host)

    var mb = Minibatch(ctx, rows)
    mb.obs = upload[DType.float32](ctx, dir + "/obs.bin")
    mb.act = upload[DType.int32](ctx, dir + "/act.bin")
    mb.logp = upload[DType.float32](ctx, dir + "/logp.bin")
    mb.adv = upload[DType.float32](ctx, dir + "/adv.bin")
    mb.ret = upload[DType.float32](ctx, dir + "/ret.bin")
    var stats = ctx.enqueue_create_buffer[DType.float32](3)
    stats.enqueue_fill(0)
    var prof = Profiler(False)
    update_minibatch(ctx, gemm, cfg, opt, mb, nets, lr, fp(stats), prof)
    ctx.synchronize()
    download(opt.g, opt.n, dir + "/grads.bin")
    download(opt.p, opt.n, dir + "/params_after.bin")
    download(stats, 3, dir + "/stats.bin")

    var values = upload[DType.float32](ctx, dir + "/values.bin")
    var rewards = upload[DType.float32](ctx, dir + "/rewards.bin")
    var dones = upload[DType.uint8](ctx, dir + "/dones.bin")
    var last_value = upload[DType.float32](ctx, dir + "/last_value.bin")
    var adv = ctx.enqueue_create_buffer[DType.float32](T * N)
    var ret = ctx.enqueue_create_buffer[DType.float32](T * N)
    ctx.enqueue_function[nn.gae_kernel](
        fp(values), fp(rewards), up(dones), fp(last_value), Int32(N), Int32(T),
        cfg.gamma, cfg.gae_lambda, fp(adv), fp(ret),
        grid_dim=grid_for(N), block_dim=BLOCK,
    )
    ctx.synchronize()
    download(adv, T * N, dir + "/gae_adv.bin")
    download(ret, T * N, dir + "/gae_ret.bin")
    print("ok")

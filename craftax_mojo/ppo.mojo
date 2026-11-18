# End-to-end PPO on Craftax-Classic-Symbolic, entirely on the GPU, in Mojo.
#
# The algorithm and hyperparameters match train/ppo_jax.py (baxtub's PPO with
# standard GAE): actor and critic MLPs (3 x 512, tanh, orthogonal init), 512
# envs x 64 steps per batch, 4 epochs x 8 minibatches, Adam(2e-4, eps 1e-5)
# with linear annealing and global-norm clipping at 1.0.
#
# The host only enqueues work: environment kernels (step, episode stats,
# block-cooperative resets), cuBLAS GEMMs (MAX's vendor BLAS binding) and
# the kernels in nn.mojo. Rollout observations are written by the
# environment straight into the rollout buffer. Nothing is copied back until
# training ends. One difference from the JAX version: finished envs get a
# freshly generated world (exact auto-reset) instead of one of the
# reset_ratio-shared optimistic resets.

from std.math import sqrt, log, cos
from std.memory import Pointer
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from .constants import OBS_SIZE, NUM_ACTIONS, MAP_CELLS, NUM_ACHIEVEMENTS
from .state import NUM_I, NUM_F, I_ACH
from .rng import SplitMix, seed_key
from . import gpu
from . import nn
from .nn import F32P, I32P, U8P, Gemm, grid_for, BLOCK

comptime U64P = Pointer[UInt64, MutUntrackedOrigin]
comptime WIDTH = 512
comptime DEPTH = 4  # linear layers per network
comptime STATS = 3 + NUM_ACHIEVEMENTS + 4  # episodes, return, length, achievements, pg, vf, ent, samples


struct Config(Copyable):
    var num_envs: Int
    var num_steps: Int
    var total_steps: Int
    var gamma: Float32
    var gae_lambda: Float32
    var clip: Float32
    var vf_coef: Float32
    var ent_coef: Float32
    var max_grad_norm: Float32
    var num_epochs: Int
    var num_minibatches: Int
    var lr: Float32
    var anneal: Bool
    var seed: UInt64
    var tf32: Bool

    def __init__(out self):
        self.num_envs = 512
        self.num_steps = 64
        self.total_steps = 1_000_000
        self.gamma = 0.99
        self.gae_lambda = 0.8
        self.clip = 0.2
        self.vf_coef = 0.5
        self.ent_coef = 0.01
        self.max_grad_norm = 1.0
        self.num_epochs = 4
        self.num_minibatches = 8
        self.lr = 2e-4
        self.anneal = True
        self.seed = 42
        self.tf32 = True

    def batch_size(self) -> Int:
        return self.num_envs * self.num_steps

    def num_batches(self) -> Int:
        return self.total_steps // self.batch_size()

    def minibatch_size(self) -> Int:
        return self.batch_size() // self.num_minibatches


# ---- Parameter layout --------------------------------------------------------

struct Net(Copyable):
    """Offsets of one MLP's weights ([in, out], row-major) and biases."""

    var sizes: List[Int]
    var w: List[Int]
    var b: List[Int]
    var end: Int

    def __init__(out self, start: Int, din: Int, dout: Int):
        self.sizes = [din, WIDTH, WIDTH, WIDTH, dout]
        self.w = List[Int]()
        self.b = List[Int]()
        var off = start
        for l in range(DEPTH):
            self.w.append(off)
            off += self.sizes[l] * self.sizes[l + 1]
            self.b.append(off)
            off += self.sizes[l + 1]
        self.end = off


def _gaussian(mut rng: SplitMix) -> Float32:
    # Box-Muller
    var u1 = max(rng.uniform(), Float32(1e-7))
    var u2 = rng.uniform()
    return sqrt(-2.0 * log(u1)) * cos(Float32(6.283185307179586) * u2)


def orthogonal(mut dst: List[Float32], off: Int, rows: Int, cols: Int, scale: Float32, mut rng: SplitMix):
    """Haar-orthogonal [rows, cols] (rows >= cols) via modified Gram-Schmidt
    on Gaussian columns (equivalent to QR with sign correction), times scale
    (jax.nn.initializers.orthogonal)."""
    var q = List[Float64](length=rows * cols, fill=0)
    for i in range(rows * cols):
        q[i] = Float64(_gaussian(rng))
    for j in range(cols):
        for k in range(j):
            var dot: Float64 = 0
            for i in range(rows):
                dot += q[i * cols + j] * q[i * cols + k]
            for i in range(rows):
                q[i * cols + j] -= dot * q[i * cols + k]
        var norm: Float64 = 0
        for i in range(rows):
            norm += q[i * cols + j] * q[i * cols + j]
        norm = sqrt(norm)
        for i in range(rows):
            q[i * cols + j] /= norm
    for i in range(rows * cols):
        dst[off + i] = Float32(q[i]) * scale


def init_params(actor: Net, critic: Net, seed: UInt64) -> List[Float32]:
    var p = List[Float32](length=critic.end, fill=0)
    var rng = SplitMix(seed_key(seed, 0xBEEF), 0)
    for net_i in range(2):
        var net = actor.copy() if net_i == 0 else critic.copy()
        for l in range(DEPTH):
            var scale = Float32(sqrt(2.0))
            if l == DEPTH - 1:
                scale = 0.01 if net_i == 0 else 1.0
            orthogonal(p, net.w[l], net.sizes[l], net.sizes[l + 1], scale, rng)
    return p^


# ---- Device helpers ------------------------------------------------------------

@inline(.always)
def fp(b: DeviceBuffer[DType.float32]) -> F32P:
    return F32P(unsafe_from_address=Int(b.unsafe_ptr()))


@inline(.always)
def ip(b: DeviceBuffer[DType.int32]) -> I32P:
    return I32P(unsafe_from_address=Int(b.unsafe_ptr()))


@inline(.always)
def up(b: DeviceBuffer[DType.uint8]) -> U8P:
    return U8P(unsafe_from_address=Int(b.unsafe_ptr()))


@inline(.always)
def u64p(b: DeviceBuffer[DType.uint64]) -> U64P:
    return U64P(unsafe_from_address=Int(b.unsafe_ptr()))


struct Acts(Copyable):
    """Per-layer activations of one network for `rows` inputs."""

    var h: List[DeviceBuffer[DType.float32]]

    def __init__(out self, ctx: DeviceContext, net: Net, rows: Int) raises:
        self.h = List[DeviceBuffer[DType.float32]]()
        for l in range(DEPTH):
            self.h.append(ctx.enqueue_create_buffer[DType.float32](rows * net.sizes[l + 1]))

    def ptr(self, l: Int) -> F32P:
        return fp(self.h[l])


def forward(
    ctx: DeviceContext, gemm: Gemm, params: F32P, net: Net, x: F32P, rows: Int,
    acts: Acts,
) raises:
    var inp = x
    for l in range(DEPTH):
        var dst = acts.ptr(l)
        var din = net.sizes[l]
        var dout = net.sizes[l + 1]
        gemm(ctx, dst, inp, params.unsafe_offset(net.w[l]), rows, dout, din)
        ctx.enqueue_function[nn.bias_act_kernel](
            dst, params.unsafe_offset(net.b[l]), Int32(rows), Int32(dout),
            Int32(1) if l < DEPTH - 1 else Int32(0),
            grid_dim=grid_for(rows * dout), block_dim=BLOCK,
        )
        inp = dst


def backward(
    ctx: DeviceContext, gemm: Gemm, params: F32P, grads: F32P, net: Net,
    x: F32P, rows: Int, acts: Acts, dout_buf: F32P, dh_a: F32P, dh_b: F32P,
) raises:
    """Gradients of a scalar loss given d loss / d output in dout_buf."""
    var dz = dout_buf
    var spare = dh_a
    for l in range(DEPTH - 1, -1, -1):
        var din = net.sizes[l]
        var dout = net.sizes[l + 1]
        var inp = x if l == 0 else acts.ptr(l - 1)
        # dW = inp^T dz ; db = colsum(dz)
        gemm(ctx, grads.unsafe_offset(net.w[l]), inp, dz, din, dout, rows, trans_a=True)
        ctx.enqueue_function[nn.colsum_kernel](
            grads.unsafe_offset(net.b[l]), dz, Int32(rows), Int32(dout),
            grid_dim=dout, block_dim=BLOCK,
        )
        if l == 0:
            break
        # d inp = dz W^T, then through tanh
        var dh = spare
        gemm(ctx, dh, dz, params.unsafe_offset(net.w[l]), rows, din, dout, trans_b=True)
        ctx.enqueue_function[nn.tanh_bwd_kernel](
            dh, acts.ptr(l - 1), Int32(rows * din),
            grid_dim=grid_for(rows * din), block_dim=BLOCK,
        )
        spare = dh_b if dh == dh_a else dh_a
        dz = dh


struct Optimizer:
    var p: DeviceBuffer[DType.float32]
    var g: DeviceBuffer[DType.float32]
    var m: DeviceBuffer[DType.float32]
    var v: DeviceBuffer[DType.float32]
    var sumsq: DeviceBuffer[DType.float32]
    var count: Int
    var n: Int

    def __init__(out self, ctx: DeviceContext, host_params: List[Float32]) raises:
        self.n = len(host_params)
        self.p = ctx.enqueue_create_buffer[DType.float32](self.n)
        self.g = ctx.enqueue_create_buffer[DType.float32](self.n)
        self.m = ctx.enqueue_create_buffer[DType.float32](self.n)
        self.v = ctx.enqueue_create_buffer[DType.float32](self.n)
        self.sumsq = ctx.enqueue_create_buffer[DType.float32](1)
        self.m.enqueue_fill(0)
        self.v.enqueue_fill(0)
        with self.p.map_to_host() as h:
            for i in range(self.n):
                h[i] = host_params[i]
        self.count = 0

    def step(mut self, ctx: DeviceContext, lr: Float32, max_norm: Float32) raises:
        self.sumsq.enqueue_fill(0)
        ctx.enqueue_function[nn.sumsq_kernel](
            fp(self.g), Int32(self.n), fp(self.sumsq), grid_dim=1024, block_dim=BLOCK,
        )
        self.count += 1
        var b1: Float32 = 0.9
        var b2: Float32 = 0.999
        var bc1 = Float32(1.0 - Float64(b1) ** Float64(self.count))
        var bc2 = Float32(1.0 - Float64(b2) ** Float64(self.count))
        ctx.enqueue_function[nn.adam_kernel](
            fp(self.p), fp(self.g), fp(self.m), fp(self.v), Int32(self.n),
            fp(self.sumsq), max_norm, lr, b1, b2, Float32(1e-5), bc1, bc2,
            grid_dim=grid_for(self.n), block_dim=BLOCK,
        )


struct Minibatch:
    var obs: DeviceBuffer[DType.float32]
    var act: DeviceBuffer[DType.int32]
    var logp: DeviceBuffer[DType.float32]
    var adv: DeviceBuffer[DType.float32]
    var ret: DeviceBuffer[DType.float32]
    var actor: Acts
    var critic: Acts
    var dlogits: DeviceBuffer[DType.float32]
    var dvalue: DeviceBuffer[DType.float32]
    var dh_a: DeviceBuffer[DType.float32]
    var dh_b: DeviceBuffer[DType.float32]
    var rows: Int

    def __init__(out self, ctx: DeviceContext, rows: Int, actor: Net, critic: Net) raises:
        self.rows = rows
        self.obs = ctx.enqueue_create_buffer[DType.float32](rows * OBS_SIZE)
        self.act = ctx.enqueue_create_buffer[DType.int32](rows)
        self.logp = ctx.enqueue_create_buffer[DType.float32](rows)
        self.adv = ctx.enqueue_create_buffer[DType.float32](rows)
        self.ret = ctx.enqueue_create_buffer[DType.float32](rows)
        self.actor = Acts(ctx, actor, rows)
        self.critic = Acts(ctx, critic, rows)
        self.dlogits = ctx.enqueue_create_buffer[DType.float32](rows * NUM_ACTIONS)
        self.dvalue = ctx.enqueue_create_buffer[DType.float32](rows)
        self.dh_a = ctx.enqueue_create_buffer[DType.float32](rows * WIDTH)
        self.dh_b = ctx.enqueue_create_buffer[DType.float32](rows * WIDTH)


def update_minibatch(
    ctx: DeviceContext, gemm: Gemm, cfg: Config, mut opt: Optimizer, mb: Minibatch,
    actor: Net, critic: Net, lr: Float32, stats: F32P,
) raises:
    """One PPO gradient step on the gathered minibatch."""
    var params = fp(opt.p)
    var grads = fp(opt.g)
    var rows = mb.rows
    forward(ctx, gemm, params, actor, fp(mb.obs), rows, mb.actor)
    forward(ctx, gemm, params, critic, fp(mb.obs), rows, mb.critic)
    ctx.enqueue_function[nn.ppo_grad_kernel](
        mb.actor.ptr(DEPTH - 1), mb.critic.ptr(DEPTH - 1), ip(mb.act), fp(mb.logp),
        fp(mb.adv), fp(mb.ret), Int32(rows), Int32(NUM_ACTIONS), cfg.clip,
        cfg.vf_coef, cfg.ent_coef, fp(mb.dlogits), fp(mb.dvalue), stats,
        grid_dim=grid_for(rows), block_dim=BLOCK,
    )
    backward(ctx, gemm, params, grads, actor, fp(mb.obs), rows, mb.actor,
             fp(mb.dlogits), fp(mb.dh_a), fp(mb.dh_b))
    backward(ctx, gemm, params, grads, critic, fp(mb.obs), rows, mb.critic,
             fp(mb.dvalue), fp(mb.dh_a), fp(mb.dh_b))
    opt.step(ctx, lr, cfg.max_grad_norm)


# ---- Training ------------------------------------------------------------------

def train(cfg: Config, out_path: String) raises:
    var ctx = DeviceContext()
    var gemm = Gemm(ctx, cfg.tf32)
    var N = cfg.num_envs
    var T = cfg.num_steps
    var B = cfg.batch_size()
    var MB = cfg.minibatch_size()
    var NB = cfg.num_batches()
    var bits = 0
    while (1 << bits) < B:
        bits += 1

    var actor = Net(0, OBS_SIZE, NUM_ACTIONS)
    var critic = Net(actor.end, OBS_SIZE, 1)
    var opt = Optimizer(ctx, init_params(actor, critic, cfg.seed))

    # Environment state (field-major, as in gpu.mojo).
    var ints = ctx.enqueue_create_buffer[DType.int32](N * NUM_I)
    var floats = ctx.enqueue_create_buffer[DType.float32](N * NUM_F)
    var map = ctx.enqueue_create_buffer[DType.uint8](N * MAP_CELLS)
    var mob = ctx.enqueue_create_buffer[DType.uint8](N * MAP_CELLS)
    var rng = ctx.enqueue_create_buffer[DType.uint64](N * 2)
    var done_idx = ctx.enqueue_create_buffer[DType.int32](N)
    var done_count = ctx.enqueue_create_buffer[DType.int32](1)
    comptime RESET_GRID = 64
    var scratch = ctx.enqueue_create_buffer[DType.float32](RESET_GRID * gpu.SCRATCH_PER_BLOCK)
    var ep_ret = ctx.enqueue_create_buffer[DType.float32](N)
    var ep_len = ctx.enqueue_create_buffer[DType.float32](N)
    ep_ret.enqueue_fill(0)
    ep_len.enqueue_fill(0)

    # Rollout storage, [T, N, ...]; obs has T + 1 slots (slot T bootstraps).
    var obs = ctx.enqueue_create_buffer[DType.float32]((T + 1) * N * OBS_SIZE)
    var act = ctx.enqueue_create_buffer[DType.int32](T * N)
    var logp = ctx.enqueue_create_buffer[DType.float32](T * N)
    var val = ctx.enqueue_create_buffer[DType.float32](T * N)
    var rew = ctx.enqueue_create_buffer[DType.float32](T * N)
    var done = ctx.enqueue_create_buffer[DType.uint8](T * N)
    var adv = ctx.enqueue_create_buffer[DType.float32](T * N)
    var ret = ctx.enqueue_create_buffer[DType.float32](T * N)
    var last_val = ctx.enqueue_create_buffer[DType.float32](N)
    var moments = ctx.enqueue_create_buffer[DType.float32](2)
    var stats = ctx.enqueue_create_buffer[DType.float32](NB * STATS)
    stats.enqueue_fill(0)
    var r_actor = Acts(ctx, actor, N)
    var r_critic = Acts(ctx, critic, N)
    var mb = Minibatch(ctx, MB, actor, critic)

    @parameter
    def obs_slot(t: Int) -> F32P:
        return fp(obs).unsafe_offset(t * N * OBS_SIZE)

    @parameter
    def env_reset(t: Int, all_envs: Bool) raises:
        ctx.enqueue_function[gpu.reset_kernel](
            ip(ints), fp(floats), up(map), up(mob), u64p(rng), obs_slot(t),
            fp(rew), up(done), Int32(N), ip(done_idx), ip(done_count),
            fp(scratch), cfg.seed, Int32(1) if all_envs else Int32(0),
            grid_dim=RESET_GRID, block_dim=gpu.RESET_BLOCK,
        )

    env_reset(0, True)
    var t_start = perf_counter_ns()
    var global_step = 0
    for batch in range(NB):
        var batch_stats = fp(stats).unsafe_offset(batch * STATS)
        # ---- Rollout ----
        for t in range(T):
            forward(ctx, gemm, fp(opt.p), actor, obs_slot(t), N, r_actor)
            forward(ctx, gemm, fp(opt.p), critic, obs_slot(t), N, r_critic)
            ctx.enqueue_copy(
                fp(val).unsafe_offset(t * N), r_critic.ptr(DEPTH - 1), N
            )
            ctx.enqueue_function[nn.sample_kernel](
                r_actor.ptr(DEPTH - 1), Int32(N), Int32(NUM_ACTIONS), cfg.seed,
                UInt64(global_step), ip(act).unsafe_offset(t * N),
                fp(logp).unsafe_offset(t * N), grid_dim=grid_for(N), block_dim=BLOCK,
            )
            done_count.enqueue_fill(0)
            ctx.enqueue_function[gpu.step_actions_kernel](
                ip(ints), fp(floats), up(map), up(mob), u64p(rng), obs_slot(t + 1),
                fp(rew).unsafe_offset(t * N), up(done).unsafe_offset(t * N), Int32(N),
                ip(done_idx), ip(done_count), ip(act).unsafe_offset(t * N),
                grid_dim=grid_for(N), block_dim=BLOCK,
            )
            ctx.enqueue_function[nn.episode_stats_kernel](
                fp(rew).unsafe_offset(t * N), up(done).unsafe_offset(t * N), ip(ints),
                Int32(N), Int32(I_ACH), Int32(NUM_ACHIEVEMENTS), fp(ep_ret),
                fp(ep_len), batch_stats, grid_dim=grid_for(N), block_dim=BLOCK,
            )
            env_reset(t + 1, False)
            global_step += 1

        # ---- Advantages ----
        forward(ctx, gemm, fp(opt.p), critic, obs_slot(T), N, r_critic)
        ctx.enqueue_copy(fp(last_val), r_critic.ptr(DEPTH - 1), N)
        ctx.enqueue_function[nn.gae_kernel](
            fp(val), fp(rew), up(done), fp(last_val), Int32(N), Int32(T),
            cfg.gamma, cfg.gae_lambda, fp(adv), fp(ret),
            grid_dim=grid_for(N), block_dim=BLOCK,
        )
        ctx.enqueue_function[nn.moments_kernel](
            fp(adv), Int32(B), fp(moments), grid_dim=1, block_dim=BLOCK
        )
        ctx.enqueue_function[nn.normalize_kernel](
            fp(adv), Int32(B), fp(moments), grid_dim=grid_for(B), block_dim=BLOCK
        )

        # ---- Updates ----
        var frac = Float32(1.0) - Float32(batch) / Float32(NB) if cfg.anneal else Float32(1.0)
        var lr = cfg.lr * frac
        for epoch in range(cfg.num_epochs):
            var key = seed_key(cfg.seed ^ 0xA11CE, UInt64(batch * cfg.num_epochs + epoch))
            for k in range(cfg.num_minibatches):
                ctx.enqueue_function[nn.gather_kernel](
                    obs_slot(0), ip(act), fp(logp), fp(adv), fp(ret),
                    fp(mb.obs), ip(mb.act), fp(mb.logp), fp(mb.adv), fp(mb.ret),
                    Int32(OBS_SIZE), Int32(MB), Int32(k * MB), Int32(B), Int32(bits), key,
                    grid_dim=grid_for(MB * OBS_SIZE), block_dim=BLOCK,
                )
                update_minibatch(ctx, gemm, cfg, opt, mb, actor, critic, lr,
                                 batch_stats.unsafe_offset(3 + NUM_ACHIEVEMENTS))
        # Next batch starts from the bootstrap observation.
        ctx.enqueue_copy(obs_slot(0), obs_slot(T), N * OBS_SIZE)
    ctx.synchronize()
    var secs = Float64(perf_counter_ns() - t_start) * 1e-9

    var steps = NB * B
    var samples_per_batch = Float32(cfg.num_epochs * B)
    var js = String('{"impl": "mojo", "device_name": "') + String(ctx.name())
    js += '", "env_steps": ' + String(steps) + ', "train_seconds": ' + String(secs)
    js += ', "steps_per_second": ' + String(Float64(steps) / secs)
    js += ', "tf32": ' + ("true" if cfg.tf32 else "false")
    js += ', "config": {"num_envs": ' + String(N) + ', "num_steps": ' + String(T)
    js += ', "total_steps": ' + String(cfg.total_steps) + ', "seed": ' + String(cfg.seed) + "}"
    var summary = js + "}"
    js += ', "metrics": {'
    with stats.map_to_host() as h:
        var names: List[String] = ["episodes", "episode_return", "episode_length"]
        for col in range(3):
            js += ('"' + names[col] + '": [') if col == 0 else (', "' + names[col] + '": [')
            for b in range(NB):
                var n_ep = h[b * STATS]
                var v = h[b * STATS + col]
                if col > 0:
                    v = v / max(n_ep, Float32(1.0))
                js += (", " if b > 0 else "") + String(v)
            js += "]"
        js += ', "achievements": ['
        for b in range(NB):
            var n_ep = max(h[b * STATS], Float32(1.0))
            js += (", " if b > 0 else "") + "["
            for k in range(NUM_ACHIEVEMENTS):
                js += (", " if k > 0 else "") + String(h[b * STATS + 3 + k] / n_ep)
            js += "]"
        js += "]"
        var lnames: List[String] = ["policy_loss", "value_loss", "entropy"]
        for c in range(3):
            js += ', "' + lnames[c] + '": ['
            for b in range(NB):
                js += (", " if b > 0 else "") + String(
                    h[b * STATS + 3 + NUM_ACHIEVEMENTS + c] / samples_per_batch
                )
            js += "]"
        var tail = max(1, NB // 10)
        var r_sum: Float32 = 0
        var l_sum: Float32 = 0
        for b in range(NB - tail, NB):
            var n_ep = max(h[b * STATS], Float32(1.0))
            r_sum += h[b * STATS + 1] / n_ep
            l_sum += h[b * STATS + 2] / n_ep
        print(summary)
        print("final episode_return", r_sum / Float32(tail), "episode_length", l_sum / Float32(tail))
    js += "}}"
    if out_path:
        with open(out_path, "w") as f:
            f.write(js)



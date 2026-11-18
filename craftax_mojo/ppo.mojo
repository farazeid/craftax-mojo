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
comptime ENV_BLOCK = 32
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
    var profile: Bool

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
        self.profile = False

    def batch_size(self) -> Int:
        return self.num_envs * self.num_steps

    def num_batches(self) -> Int:
        return self.total_steps // self.batch_size()

    def minibatch_size(self) -> Int:
        return self.batch_size() // self.num_minibatches


# ---- Parameter layout --------------------------------------------------------
# Actor and critic have the same hidden shapes, so each hidden layer stores
# the actor's then the critic's weights contiguously ([2, in, 512]) and the
# two networks run as one strided-batched GEMM per layer. Weights are
# [in, out] row-major. Order: for hidden layers 0..2: W_actor, W_critic,
# b_actor, b_critic; then the heads: W_actor [512, 17], b_actor, W_critic
# [512, 1], b_critic.

struct Nets(Copyable):
    var din: List[Int]  # input width of hidden layer l
    var w: List[Int]  # hidden layer l: offset of [2, din, WIDTH]
    var b: List[Int]  # hidden layer l: offset of [2, WIDTH]
    var wa: Int  # actor head
    var ba: Int
    var wc: Int  # critic head
    var bc: Int
    var end: Int

    def __init__(out self, obs_dim: Int):
        self.din = [obs_dim, WIDTH, WIDTH]
        self.w = List[Int]()
        self.b = List[Int]()
        var off = 0
        for l in range(DEPTH - 1):
            self.w.append(off)
            off += 2 * self.din[l] * WIDTH
            self.b.append(off)
            off += 2 * WIDTH
        self.wa = off
        off += WIDTH * NUM_ACTIONS
        self.ba = off
        off += NUM_ACTIONS
        self.wc = off
        off += WIDTH
        self.bc = off
        off += 1
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


def init_params(nets: Nets, seed: UInt64) -> List[Float32]:
    var p = List[Float32](length=nets.end, fill=0)
    var rng = SplitMix(seed_key(seed, 0xBEEF), 0)
    var gain = Float32(sqrt(2.0))
    for l in range(DEPTH - 1):
        for k in range(2):
            orthogonal(p, nets.w[l] + k * nets.din[l] * WIDTH, nets.din[l], WIDTH, gain, rng)
    orthogonal(p, nets.wa, WIDTH, NUM_ACTIONS, 0.01, rng)
    orthogonal(p, nets.wc, WIDTH, 1, 1.0, rng)
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
    """Activations for `rows` inputs: hidden layers as [2, rows, WIDTH]
    (actor, critic), then logits [rows, NUM_ACTIONS] and values [rows]."""

    var h: List[DeviceBuffer[DType.float32]]
    var logits: DeviceBuffer[DType.float32]
    var value: DeviceBuffer[DType.float32]
    var rows: Int

    def __init__(out self, ctx: DeviceContext, rows: Int) raises:
        self.rows = rows
        self.h = List[DeviceBuffer[DType.float32]]()
        for _ in range(DEPTH - 1):
            self.h.append(ctx.enqueue_create_buffer[DType.float32](2 * rows * WIDTH))
        self.logits = ctx.enqueue_create_buffer[DType.float32](rows * NUM_ACTIONS)
        self.value = ctx.enqueue_create_buffer[DType.float32](rows)

    def hidden(self, l: Int) -> F32P:
        return fp(self.h[l])

    def critic_hidden(self, l: Int) -> F32P:
        return fp(self.h[l]).unsafe_offset(self.rows * WIDTH)


def forward(
    ctx: DeviceContext, gemm: Gemm, params: F32P, nets: Nets, x: F32P, acts: Acts
) raises:
    """Actor logits and critic values for x [rows, obs_dim]."""
    var rows = acts.rows
    var rw = rows * WIDTH
    for l in range(DEPTH - 1):
        var din = nets.din[l]
        var inp = x if l == 0 else acts.hidden(l - 1)
        gemm(
            ctx, acts.hidden(l), inp, params.unsafe_offset(nets.w[l]), rows, WIDTH, din,
            batch=2, stride_a=0 if l == 0 else rw, stride_b=din * WIDTH, stride_c=rw,
        )
        ctx.enqueue_function[nn.bias_act_batched_kernel](
            acts.hidden(l), params.unsafe_offset(nets.b[l]), Int32(rows), Int32(WIDTH),
            Int32(2), Int32(1), grid_dim=grid_for(2 * rw), block_dim=BLOCK,
        )
    var last = DEPTH - 2
    gemm(ctx, fp(acts.logits), acts.hidden(last), params.unsafe_offset(nets.wa), rows, NUM_ACTIONS, WIDTH)
    ctx.enqueue_function[nn.bias_act_kernel](
        fp(acts.logits), params.unsafe_offset(nets.ba), Int32(rows), Int32(NUM_ACTIONS),
        Int32(0), grid_dim=grid_for(rows * NUM_ACTIONS), block_dim=BLOCK,
    )
    gemm(ctx, fp(acts.value), acts.critic_hidden(last), params.unsafe_offset(nets.wc), rows, 1, WIDTH)
    ctx.enqueue_function[nn.bias_act_kernel](
        fp(acts.value), params.unsafe_offset(nets.bc), Int32(rows), Int32(1),
        Int32(0), grid_dim=grid_for(rows), block_dim=BLOCK,
    )


def backward(
    ctx: DeviceContext, gemm: Gemm, params: F32P, grads: F32P, nets: Nets,
    x: F32P, acts: Acts, dlogits: F32P, dvalue: F32P, dh_a: F32P, dh_b: F32P,
    ones: F32P,
) raises:
    """Parameter gradients given d loss / d logits and d loss / d values.
    dh_a, dh_b are [2, rows, WIDTH] scratch; `ones` holds `rows` 1.0s (bias
    gradients are ones^T dz)."""
    var rows = acts.rows
    var rw = rows * WIDTH
    var last = DEPTH - 2
    # Heads.
    gemm(ctx, grads.unsafe_offset(nets.wa), acts.hidden(last), dlogits, WIDTH, NUM_ACTIONS, rows, trans_a=True)
    gemm(ctx, grads.unsafe_offset(nets.ba), ones, dlogits, 1, NUM_ACTIONS, rows)
    gemm(ctx, grads.unsafe_offset(nets.wc), acts.critic_hidden(last), dvalue, WIDTH, 1, rows, trans_a=True)
    gemm(ctx, grads.unsafe_offset(nets.bc), ones, dvalue, 1, 1, rows)
    gemm(ctx, dh_a, dlogits, params.unsafe_offset(nets.wa), rows, WIDTH, NUM_ACTIONS, trans_b=True)
    gemm(ctx, dh_a.unsafe_offset(rw), dvalue, params.unsafe_offset(nets.wc), rows, WIDTH, 1, trans_b=True)
    ctx.enqueue_function[nn.tanh_bwd_kernel](
        dh_a, acts.hidden(last), Int32(2 * rw), grid_dim=grid_for(2 * rw), block_dim=BLOCK,
    )
    # Hidden layers, both networks per GEMM.
    var dz = dh_a
    var spare = dh_b
    for l in range(DEPTH - 2, -1, -1):
        var din = nets.din[l]
        var inp = x if l == 0 else acts.hidden(l - 1)
        var s_in = 0 if l == 0 else rw
        gemm(
            ctx, grads.unsafe_offset(nets.w[l]), inp, dz, din, WIDTH, rows, trans_a=True,
            batch=2, stride_a=s_in, stride_b=rw, stride_c=din * WIDTH,
        )
        gemm(
            ctx, grads.unsafe_offset(nets.b[l]), ones, dz, 1, WIDTH, rows,
            batch=2, stride_a=0, stride_b=rw, stride_c=WIDTH,
        )
        if l == 0:
            break
        gemm(
            ctx, spare, dz, params.unsafe_offset(nets.w[l]), rows, WIDTH, WIDTH, trans_b=True,
            batch=2, stride_a=rw, stride_b=WIDTH * WIDTH, stride_c=rw,
        )
        ctx.enqueue_function[nn.tanh_bwd_kernel](
            spare, acts.hidden(l - 1), Int32(2 * rw), grid_dim=grid_for(2 * rw), block_dim=BLOCK,
        )
        var t = dz
        dz = spare
        spare = t


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
        # float32, like optax's bias_correction (1 - decay**count)
        var bc1 = Float32(1.0) - b1 ** Float32(self.count)
        var bc2 = Float32(1.0) - b2 ** Float32(self.count)
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
    var acts: Acts
    var dlogits: DeviceBuffer[DType.float32]
    var dvalue: DeviceBuffer[DType.float32]
    var dh_a: DeviceBuffer[DType.float32]
    var dh_b: DeviceBuffer[DType.float32]
    var ones: DeviceBuffer[DType.float32]
    var rows: Int

    def __init__(out self, ctx: DeviceContext, rows: Int) raises:
        self.rows = rows
        self.obs = ctx.enqueue_create_buffer[DType.float32](rows * OBS_SIZE)
        self.act = ctx.enqueue_create_buffer[DType.int32](rows)
        self.logp = ctx.enqueue_create_buffer[DType.float32](rows)
        self.adv = ctx.enqueue_create_buffer[DType.float32](rows)
        self.ret = ctx.enqueue_create_buffer[DType.float32](rows)
        self.acts = Acts(ctx, rows)
        self.dlogits = ctx.enqueue_create_buffer[DType.float32](rows * NUM_ACTIONS)
        self.dvalue = ctx.enqueue_create_buffer[DType.float32](rows)
        self.dh_a = ctx.enqueue_create_buffer[DType.float32](2 * rows * WIDTH)
        self.dh_b = ctx.enqueue_create_buffer[DType.float32](2 * rows * WIDTH)
        self.ones = ctx.enqueue_create_buffer[DType.float32](rows)
        self.ones.enqueue_fill(1.0)


struct Profiler:
    """Optional per-stage timing: when enabled, mark() synchronises the
    device and charges the elapsed wall time to a named stage. Diagnostic
    only (synchronising removes host/device overlap)."""

    var enabled: Bool
    var names: List[String]
    var secs: List[Float64]
    var t: Int

    def __init__(out self, enabled: Bool):
        self.enabled = enabled
        self.names = List[String]()
        self.secs = List[Float64]()
        self.t = Int(perf_counter_ns())

    def mark(mut self, ctx: DeviceContext, name: String) raises:
        if not self.enabled:
            return
        ctx.synchronize()
        var now = Int(perf_counter_ns())
        var dt = Float64(now - self.t) * 1e-9
        self.t = now
        for i in range(len(self.names)):
            if self.names[i] == name:
                self.secs[i] += dt
                return
        self.names.append(name)
        self.secs.append(dt)

    def report(self, batches: Int):
        if not self.enabled:
            return
        var total: Float64 = 0
        for i in range(len(self.secs)):
            total += self.secs[i]
        for i in range(len(self.names)):
            print(
                "  ", self.names[i], ":", self.secs[i] / Float64(batches) * 1e3,
                "ms/batch (", self.secs[i] / total * 100, "%)",
            )


def update_minibatch(
    ctx: DeviceContext, gemm: Gemm, cfg: Config, mut opt: Optimizer, mb: Minibatch,
    nets: Nets, lr: Float32, stats: F32P, mut prof: Profiler,
) raises:
    """One PPO gradient step on the gathered minibatch."""
    var params = fp(opt.p)
    var grads = fp(opt.g)
    var rows = mb.rows
    forward(ctx, gemm, params, nets, fp(mb.obs), mb.acts)
    prof.mark(ctx, "update: forward")
    ctx.enqueue_function[nn.ppo_grad_kernel](
        fp(mb.acts.logits), fp(mb.acts.value), ip(mb.act), fp(mb.logp),
        fp(mb.adv), fp(mb.ret), Int32(rows), Int32(NUM_ACTIONS), cfg.clip,
        cfg.vf_coef, cfg.ent_coef, fp(mb.dlogits), fp(mb.dvalue), stats,
        grid_dim=grid_for(rows), block_dim=BLOCK,
    )
    prof.mark(ctx, "update: loss grad")
    backward(ctx, gemm, params, grads, nets, fp(mb.obs), mb.acts, fp(mb.dlogits),
             fp(mb.dvalue), fp(mb.dh_a), fp(mb.dh_b), fp(mb.ones))
    prof.mark(ctx, "update: backward")
    opt.step(ctx, lr, cfg.max_grad_norm)
    prof.mark(ctx, "update: clip + adam")


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

    var nets = Nets(OBS_SIZE)
    var opt = Optimizer(ctx, init_params(nets, cfg.seed))

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
    var r_acts = Acts(ctx, N)
    var mb = Minibatch(ctx, MB)

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
    var prof = Profiler(cfg.profile)
    for batch in range(NB):
        var batch_stats = fp(stats).unsafe_offset(batch * STATS)
        # ---- Rollout ----
        for t in range(T):
            forward(ctx, gemm, fp(opt.p), nets, obs_slot(t), r_acts)
            ctx.enqueue_copy(fp(val).unsafe_offset(t * N), fp(r_acts.value), N)
            prof.mark(ctx, "rollout: policy forward")
            ctx.enqueue_function[nn.sample_kernel](
                fp(r_acts.logits), Int32(N), Int32(NUM_ACTIONS), cfg.seed,
                UInt64(global_step), ip(act).unsafe_offset(t * N),
                fp(logp).unsafe_offset(t * N), grid_dim=grid_for(N), block_dim=BLOCK,
            )
            prof.mark(ctx, "rollout: sample")
            done_count.enqueue_fill(0)
            # One warp per block spreads the (latency-bound) env threads
            # over more SMs: 1.2M -> 2.1M random steps/s at 512 envs.
            ctx.enqueue_function[gpu.step_actions_kernel](
                ip(ints), fp(floats), up(map), up(mob), u64p(rng), obs_slot(t + 1),
                fp(rew).unsafe_offset(t * N), up(done).unsafe_offset(t * N), Int32(N),
                ip(done_idx), ip(done_count), ip(act).unsafe_offset(t * N),
                grid_dim=(N + ENV_BLOCK - 1) // ENV_BLOCK, block_dim=ENV_BLOCK,
            )
            ctx.enqueue_function[nn.episode_stats_kernel](
                fp(rew).unsafe_offset(t * N), up(done).unsafe_offset(t * N), ip(ints),
                Int32(N), Int32(I_ACH), Int32(NUM_ACHIEVEMENTS), fp(ep_ret),
                fp(ep_len), batch_stats, grid_dim=grid_for(N), block_dim=BLOCK,
            )
            prof.mark(ctx, "rollout: env step + stats")
            env_reset(t + 1, False)
            prof.mark(ctx, "rollout: env reset")
            global_step += 1

        # ---- Advantages ----
        forward(ctx, gemm, fp(opt.p), nets, obs_slot(T), r_acts)
        ctx.enqueue_copy(fp(last_val), fp(r_acts.value), N)
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

        prof.mark(ctx, "advantages")
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
                prof.mark(ctx, "update: gather")
                update_minibatch(ctx, gemm, cfg, opt, mb, nets, lr,
                                 batch_stats.unsafe_offset(3 + NUM_ACHIEVEMENTS), prof)
        # Next batch starts from the bootstrap observation.
        ctx.enqueue_copy(obs_slot(0), obs_slot(T), N * OBS_SIZE)
        prof.mark(ctx, "next-batch copy")
    ctx.synchronize()
    prof.report(NB)
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



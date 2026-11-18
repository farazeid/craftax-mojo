# GPU kernels for end-to-end PPO: GEMM dispatch, MLP forward/backward glue,
# policy sampling, PPO loss gradients, GAE, minibatch gathering, gradient
# clipping and Adam. All matrices are row-major float32.

from std.math import exp, log, sqrt, tanh
from std.memory import Pointer
from std.atomic import Atomic
from max.gpu import global_idx, thread_idx, block_idx, grid_dim
from max.gpu.host import DeviceContext
from max.gpu.primitives import block
import linalg.matmul.vendor.blas as vendor_blas
from _cublas.cublas import (
    _get_dylib_function,
    check_cublas_error,
    cublasSetMathMode,
    cublasMath_t,
    cublasOperation_t,
    ComputeType,
    Algorithm,
)
from _cublas.dtype import DataType
from _cublas.result import Result

from .rng import SplitMix, seed_key

comptime F32P = Pointer[Float32, MutUntrackedOrigin]
comptime I32P = Pointer[Int32, MutUntrackedOrigin]
comptime U8P = Pointer[UInt8, MutUntrackedOrigin]

comptime BLOCK = 256


@inline(.always)
def grid_for(n: Int) -> Int:
    return (n + BLOCK - 1) // BLOCK


# ---- GEMM ------------------------------------------------------------------
# Row-major, strided-batched: for each batch i,
#   C_i[M,N] = alpha * op(A_i) @ op(B_i) + beta * C_i
# where op(A_i) is [M,K] (stored [K,M] if trans_a) at A + i * stride_a, etc.
# A stride of 0 shares one operand across the batch.

def gemm_naive_kernel(
    c: F32P, a: F32P, b: F32P, m: Int32, n: Int32, k: Int32,
    trans_a: Int32, trans_b: Int32, alpha: Float32, beta: Float32,
    batch: Int32, sa: Int32, sb: Int32, sc: Int32,
):
    var idx = global_idx.x
    var M = Int(m)
    var N = Int(n)
    var K = Int(k)
    if idx >= Int(batch) * M * N:
        return
    var bi = idx // (M * N)
    var r = idx % (M * N)
    var i = r // N
    var j = r % N
    var ap = a.unsafe_offset(bi * Int(sa))
    var bp = b.unsafe_offset(bi * Int(sb))
    var acc: Float32 = 0
    for p in range(K):
        var av = ap[unsafe_offset = p * M + i] if trans_a != 0 else ap[unsafe_offset = i * K + p]
        var bv = bp[unsafe_offset = j * K + p] if trans_b != 0 else bp[unsafe_offset = p * N + j]
        acc += av * bv
    var res = alpha * acc
    var ci = bi * Int(sc) + i * N + j
    if beta != 0:
        res += beta * c[unsafe_offset=ci]
    c[unsafe_offset=ci] = res


comptime _VoidP = OpaquePointer[MutAnyOrigin]


@inline(.always)
def _void(p: F32P) -> _VoidP:
    return _VoidP(unsafe_from_address=Int(p))


struct Gemm:
    """cuBLAS (cublasGemmStridedBatchedEx) on NVIDIA GPUs, a naive Mojo
    kernel elsewhere.

    MAX's vendor_blas.matmul resolves a global handle by name, re-binds it to
    the stream, pushes a context, opens a trace scope and sets the math mode
    on every call, and has no strided-batched entry point. PPO issues ~1000
    GEMMs per batch, many of them actor/critic pairs of identical shape, so
    this binds cublasGemmStridedBatchedEx through MAX's cuBLAS loader, with
    one handle whose stream and math mode are set once.
    """

    var use_blas: Bool
    var tf32: Bool
    var handle: Optional[vendor_blas.Handle[vendor_blas.Backend.CUBLAS]]

    def __init__(out self, ctx: DeviceContext, tf32: Bool) raises:
        self.use_blas = String(ctx.api()) == "cuda"
        self.tf32 = tf32
        self.handle = None
        if self.use_blas:
            var h = vendor_blas.Handle[vendor_blas.Backend.CUBLAS]()
            vendor_blas._attach_handle_to_stream(ctx, h)
            check_cublas_error(cublasSetMathMode(
                h._get_cublas(),
                cublasMath_t.CUBLAS_TF32_TENSOR_OP_MATH if tf32 else cublasMath_t.CUBLAS_DEFAULT_MATH,
            ))
            self.handle = h

    def __call__(
        self, ctx: DeviceContext, c: F32P, a: F32P, b: F32P, m: Int, n: Int, k: Int,
        trans_a: Bool = False, trans_b: Bool = False, alpha: Float32 = 1.0,
        beta: Float32 = 0.0, batch: Int = 1, stride_a: Int = 0, stride_b: Int = 0,
        stride_c: Int = 0,
    ) raises:
        if not self.use_blas:
            ctx.enqueue_function[gemm_naive_kernel](
                c, a, b, Int32(m), Int32(n), Int32(k),
                Int32(1) if trans_a else Int32(0), Int32(1) if trans_b else Int32(0),
                alpha, beta, Int32(batch), Int32(stride_a), Int32(stride_b),
                Int32(stride_c), grid_dim=grid_for(batch * m * n), block_dim=BLOCK,
            )
            return
        # Row-major C = op(A) op(B) is column-major C^T = op(B)^T op(A)^T.
        var h = self.handle.value()._get_cublas()
        var al = alpha
        var be = beta
        var op_a = cublasOperation_t.CUBLAS_OP_T if trans_a else cublasOperation_t.CUBLAS_OP_N
        var op_b = cublasOperation_t.CUBLAS_OP_T if trans_b else cublasOperation_t.CUBLAS_OP_N
        var lda = Int32(m if trans_a else k)
        var ldb = Int32(k if trans_b else n)
        var compute = ComputeType.COMPUTE_32F_FAST_TF32 if self.tf32 else ComputeType.COMPUTE_32F
        var gemm_fn = _get_dylib_function[
            "cublasGemmStridedBatchedEx",
            def(
                type_of(h), cublasOperation_t, cublasOperation_t, Int32, Int32, Int32,
                _VoidP, _VoidP, DataType, Int32, Int64, _VoidP, DataType, Int32, Int64,
                _VoidP, _VoidP, DataType, Int32, Int64, Int32, ComputeType, Algorithm,
            ) thin -> Result,
        ]()
        check_cublas_error(gemm_fn(
            h, op_b, op_a, Int32(n), Int32(m), Int32(k),
            _VoidP(unsafe_from_address=Int(Pointer(to=al))),
            _void(b), DataType.R_32F, ldb, Int64(stride_b),
            _void(a), DataType.R_32F, lda, Int64(stride_a),
            _VoidP(unsafe_from_address=Int(Pointer(to=be))),
            _void(c), DataType.R_32F, Int32(n), Int64(stride_c), Int32(batch),
            compute, Algorithm.DEFAULT,
        ))
        _ = al
        _ = be


# ---- Elementwise -----------------------------------------------------------

def bias_act_kernel(z: F32P, b: F32P, m: Int32, n: Int32, act: Int32):
    """z[i, j] = act(z[i, j] + b[j]); act 1 = tanh, 0 = identity."""
    var idx = global_idx.x
    if idx >= Int(m) * Int(n):
        return
    var v = z[unsafe_offset=idx] + b[unsafe_offset = idx % Int(n)]
    z[unsafe_offset=idx] = tanh(v) if act != 0 else v


def bias_act_batched_kernel(
    z: F32P, b: F32P, rows: Int32, n: Int32, batch: Int32, act: Int32
):
    """z is [batch, rows, n]; z[k, i, j] = act(z[k, i, j] + b[k * n + j])."""
    var idx = global_idx.x
    var per = Int(rows) * Int(n)
    if idx >= Int(batch) * per:
        return
    var k = idx // per
    var v = z[unsafe_offset=idx] + b[unsafe_offset = k * Int(n) + idx % Int(n)]
    z[unsafe_offset=idx] = tanh(v) if act != 0 else v


def tanh_bwd_kernel(dh: F32P, h: F32P, count: Int32):
    """dh *= 1 - h^2 (h = tanh output)."""
    var idx = global_idx.x
    if idx >= Int(count):
        return
    var hv = h[unsafe_offset=idx]
    dh[unsafe_offset=idx] = dh[unsafe_offset=idx] * (1.0 - hv * hv)


def colsum_kernel(db: F32P, dz: F32P, m: Int32, n: Int32):
    """db[j] = sum_i dz[i, j]; one block per column."""
    var j = block_idx.x
    var acc: Float32 = 0
    for i in range(thread_idx.x, Int(m), BLOCK):
        acc += dz[unsafe_offset = i * Int(n) + j]
    var total = block.sum[block_size=BLOCK, broadcast=False](val=acc)
    if thread_idx.x == 0:
        db[unsafe_offset=j] = total[0]


# ---- Policy sampling ------------------------------------------------------

def sample_kernel(
    logits: F32P, n: Int32, num_actions: Int32, seed: UInt64, step: UInt64,
    actions: I32P, logp: F32P,
):
    """Categorical sample per env by inverse CDF; also records log pi(a)."""
    var e = global_idx.x
    if e >= Int(n):
        return
    var A = Int(num_actions)
    var row = logits.unsafe_offset(e * A)
    var mx = row[unsafe_offset=0]
    for a in range(1, A):
        mx = max(mx, row[unsafe_offset=a])
    var z: Float32 = 0
    for a in range(A):
        z += exp(row[unsafe_offset=a] - mx)
    var rng = SplitMix(seed_key(seed ^ 0x5EED5EED, UInt64(e)), step)
    var u = rng.uniform() * z
    var chosen = A - 1
    var cum: Float32 = 0
    for a in range(A):
        cum += exp(row[unsafe_offset=a] - mx)
        if u < cum:
            chosen = a
            break
    actions[unsafe_offset=e] = Int32(chosen)
    logp[unsafe_offset=e] = row[unsafe_offset=chosen] - mx - log(z)


# ---- PPO loss --------------------------------------------------------------

def ppo_grad_kernel(
    logits: F32P, value: F32P, action: I32P, old_logp: F32P, adv: F32P,
    ret: F32P, m: Int32, num_actions: Int32, clip: Float32, vf_coef: Float32,
    ent_coef: Float32, dlogits: F32P, dvalue: F32P, stats: F32P,
):
    """Per-sample gradient of the (mean) PPO loss w.r.t. logits and value.

    loss = mean(max(-A r, -A clip(r))) + vf_coef * mean(0.5 (v - R)^2)
           - ent_coef * mean(H). stats += (pg, vf, ent) sums.
    """
    var i = global_idx.x
    if i >= Int(m):
        return
    var A = Int(num_actions)
    var row = logits.unsafe_offset(i * A)
    var drow = dlogits.unsafe_offset(i * A)
    var mx = row[unsafe_offset=0]
    for a in range(1, A):
        mx = max(mx, row[unsafe_offset=a])
    var z: Float32 = 0
    for a in range(A):
        z += exp(row[unsafe_offset=a] - mx)
    var logz = log(z) + mx
    var ent: Float32 = 0
    for a in range(A):
        var lp = row[unsafe_offset=a] - logz
        ent -= exp(lp) * lp
    var act = Int(action[unsafe_offset=i])
    var logp = row[unsafe_offset=act] - logz
    var ratio = exp(logp - old_logp[unsafe_offset=i])
    var advv = adv[unsafe_offset=i]
    var l1 = -advv * ratio
    var l2 = -advv * min(max(ratio, 1.0 - clip), 1.0 + clip)
    var pg = max(l1, l2)
    # d pg / d logp: the unclipped branch is active unless clipping made the
    # loss strictly larger (ties: both branches equal, same gradient).
    var dlogp: Float32 = -advv * ratio if l1 >= l2 else 0.0
    var inv_m = 1.0 / Float32(Int(m))
    for a in range(A):
        var lp = row[unsafe_offset=a] - logz
        var p = exp(lp)
        var onehot: Float32 = 1.0 if a == act else 0.0
        var g = dlogp * (onehot - p)
        # d(-ent_coef * H)/dlogit_a = ent_coef * p_a * (log p_a + H)
        g += ent_coef * p * (lp + ent)
        drow[unsafe_offset=a] = g * inv_m
    var v = value[unsafe_offset=i]
    var r = ret[unsafe_offset=i]
    dvalue[unsafe_offset=i] = vf_coef * (v - r) * inv_m
    _ = Atomic[Float32].fetch_add(stats.unsafe_offset(0), pg)
    _ = Atomic[Float32].fetch_add(stats.unsafe_offset(1), 0.5 * (v - r) * (v - r))
    _ = Atomic[Float32].fetch_add(stats.unsafe_offset(2), ent)


# ---- GAE -------------------------------------------------------------------

def gae_kernel(
    values: F32P, rewards: F32P, dones: U8P, last_value: F32P, n: Int32,
    t: Int32, gamma: Float32, lam: Float32, adv: F32P, ret: F32P,
):
    """Per env, reverse over time; arrays are [T, N]."""
    var e = global_idx.x
    if e >= Int(n):
        return
    var N = Int(n)
    var g: Float32 = 0
    var next_value = last_value[unsafe_offset=e]
    for s in range(Int(t) - 1, -1, -1):
        var k = s * N + e
        var not_done: Float32 = 0.0 if dones[unsafe_offset=k] != 0 else 1.0
        var v = values[unsafe_offset=k]
        var delta = rewards[unsafe_offset=k] + gamma * next_value * not_done - v
        g = delta + gamma * lam * not_done * g
        adv[unsafe_offset=k] = g
        ret[unsafe_offset=k] = g + v
        next_value = v


def moments_kernel(x: F32P, count: Int32, dst: F32P):
    """Single block: dst = (mean, std) of x (population std, like jnp.std)."""
    var s: Float32 = 0
    for i in range(thread_idx.x, Int(count), BLOCK):
        s += x[unsafe_offset=i]
    var mean = block.sum[block_size=BLOCK, broadcast=True](val=s)[0] / Float32(Int(count))
    var q: Float32 = 0
    for i in range(thread_idx.x, Int(count), BLOCK):
        var d = x[unsafe_offset=i] - mean
        q += d * d
    var var_ = block.sum[block_size=BLOCK, broadcast=True](val=q)[0] / Float32(Int(count))
    if thread_idx.x == 0:
        dst[unsafe_offset=0] = mean
        dst[unsafe_offset=1] = sqrt(var_)


def normalize_kernel(x: F32P, count: Int32, moments: F32P):
    var i = global_idx.x
    if i >= Int(count):
        return
    x[unsafe_offset=i] = (x[unsafe_offset=i] - moments[unsafe_offset=0]) / (
        moments[unsafe_offset=1] + 1e-8
    )


# ---- Minibatch gather ------------------------------------------------------

@inline(.always)
def permute(i: Int, bits: Int, key: UInt64) -> Int:
    """A keyed bijection on [0, 2^bits): rounds of odd-multiply-add and
    xor-shift, each invertible mod 2^bits."""
    var mask = (UInt64(1) << UInt64(bits)) - 1
    var x = UInt64(i)
    var k = key
    for _ in range(4):
        var mul = (k & mask) | 1
        x = (x * mul + (k >> 32)) & mask
        x ^= x >> UInt64(bits // 2 + 1)
        k = k * 0x9E3779B97F4A7C15 + 0xD1B54A32D192ED03
    return Int(x)


@inline(.always)
def batch_index(i: Int, size: Int, bits: Int, key: UInt64) -> Int:
    """Shuffled index of i for a batch of `size` (cycle-walking if size is
    not a power of two)."""
    var j = permute(i, bits, key)
    while j >= size:
        j = permute(j, bits, key)
    return j


def gather_kernel(
    src_obs: F32P, src_act: I32P, src_logp: F32P, src_adv: F32P, src_ret: F32P,
    dst_obs: F32P, dst_act: I32P, dst_logp: F32P, dst_adv: F32P, dst_ret: F32P,
    obs_dim: Int32, mb: Int32, offset: Int32, size: Int32, bits: Int32, key: UInt64,
):
    """Row r of the minibatch is batch element perm(offset + r)."""
    var idx = global_idx.x
    var D = Int(obs_dim)
    if idx >= Int(mb) * D:
        return
    var r = idx // D
    var col = idx % D
    var src = batch_index(Int(offset) + r, Int(size), Int(bits), key)
    dst_obs[unsafe_offset=idx] = src_obs[unsafe_offset = src * D + col]
    if col == 0:
        dst_act[unsafe_offset=r] = src_act[unsafe_offset=src]
        dst_logp[unsafe_offset=r] = src_logp[unsafe_offset=src]
        dst_adv[unsafe_offset=r] = src_adv[unsafe_offset=src]
        dst_ret[unsafe_offset=r] = src_ret[unsafe_offset=src]


# ---- Optimiser -------------------------------------------------------------

def sumsq_kernel(g: F32P, count: Int32, dst: F32P):
    var acc: Float32 = 0
    var stride = grid_dim.x * BLOCK
    for i in range(global_idx.x, Int(count), stride):
        var v = g[unsafe_offset=i]
        acc += v * v
    var total = block.sum[block_size=BLOCK, broadcast=False](val=acc)
    if thread_idx.x == 0:
        _ = Atomic[Float32].fetch_add(dst, total[0])


def adam_kernel(
    p: F32P, g: F32P, m: F32P, v: F32P, count: Int32, sumsq: F32P,
    max_norm: Float32, lr: Float32, b1: Float32, b2: Float32, eps: Float32,
    bc1: Float32, bc2: Float32,
):
    """optax.chain(clip_by_global_norm(max_norm), adam(lr, b1, b2, eps))."""
    var i = global_idx.x
    if i >= Int(count):
        return
    var norm = sqrt(sumsq[unsafe_offset=0])
    var gi = g[unsafe_offset=i]
    if norm >= max_norm:
        gi = gi / norm * max_norm
    var mi = b1 * m[unsafe_offset=i] + (1.0 - b1) * gi
    var vi = b2 * v[unsafe_offset=i] + (1.0 - b2) * gi * gi
    m[unsafe_offset=i] = mi
    v[unsafe_offset=i] = vi
    var mhat = mi / bc1
    var vhat = vi / bc2
    p[unsafe_offset=i] = p[unsafe_offset=i] - lr * mhat / (sqrt(vhat) + eps)


# ---- Episode statistics ----------------------------------------------------

def episode_stats_kernel(
    reward: F32P, done: U8P, ints: I32P, n: Int32, ach_offset: Int32,
    num_ach: Int32, ep_ret: F32P, ep_len: F32P, stats: F32P,
):
    """Accumulates per-env returns/lengths; on termination adds the episode
    (count, return, length, achievements) to stats. Must run before resets.

    `ints` is the env int32 state in field-major layout.
    """
    var e = global_idx.x
    if e >= Int(n):
        return
    var r = ep_ret[unsafe_offset=e] + reward[unsafe_offset=e]
    var l = ep_len[unsafe_offset=e] + 1.0
    if done[unsafe_offset=e] != 0:
        _ = Atomic[Float32].fetch_add(stats.unsafe_offset(0), 1.0)
        _ = Atomic[Float32].fetch_add(stats.unsafe_offset(1), r)
        _ = Atomic[Float32].fetch_add(stats.unsafe_offset(2), l)
        for k in range(Int(num_ach)):
            if ints[unsafe_offset = (Int(ach_offset) + k) * Int(n) + e] != 0:
                _ = Atomic[Float32].fetch_add(stats.unsafe_offset(3 + k), 1.0)
        r = 0
        l = 0
    ep_ret[unsafe_offset=e] = r
    ep_len[unsafe_offset=e] = l

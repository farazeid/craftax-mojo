# GPU pipeline with deferred, cooperative resets.
#
# The naive kernel (vec.step_random_kernel) regenerates a finished env's
# world inside the step kernel. With ~0.6% of envs finishing per step, most
# warps contain a resetting thread and stall behind a ~4096-cell serial loop.
# Here a step is two kernels:
#
#   step_kernel   one thread per env: random action, game step, reward/done,
#                 observation; finished envs append themselves to a done list
#                 (atomic) instead of resetting.
#   reset_kernel  one thread *block* per finished env: the block generates the
#                 world cooperatively (16 cells per thread, block-wide min/max,
#                 stone count and prefix sum for the diamond), then writes the
#                 observation (one element per thread).
#
# All three consume each env's counter-based RNG exactly as the CPU path does,
# so trajectories stay bit-identical to vec.step_random_cpu.

from std.memory import Pointer
from std.atomic import Atomic
from max.gpu import global_idx, thread_idx, block_idx, grid_dim
from max.gpu.sync import barrier
from max.gpu.primitives import block

from .constants import *
from .state import *
from .rng import SplitMix, seed_key
from .worldgen import (
    D_TOTAL,
    T_SIZE,
    fill_table_entry,
    water_raw,
    mountain_raw,
    path_raw,
    tree_raw,
    MinMax,
    cell_block,
    init_scalars,
)
from .game import step, is_game_over
from .obs import write_obs
from .vec import Buffers, random_action, I32P, F32P, U8P, U64P

comptime RESET_BLOCK = 256
comptime CELLS_PER_THREAD = MAP_CELLS // RESET_BLOCK  # 16
comptime SCRATCH_PER_BLOCK = 2 * T_SIZE + MAP_CELLS


def step_kernel(
    ints: I32P, floats: F32P, map: U8P, mob_map: U8P, rng: U64P, obs: F32P,
    reward: F32P, done: U8P, n: Int32, done_idx: I32P, done_count: I32P,
):
    var e = global_idx.x
    if e >= Int(n):
        return
    var b = Buffers(ints, floats, map, mob_map, rng, obs, reward, done, Int(n))
    var action = random_action[True](b, e)
    var r = b.load_rng[True](e)
    var env = b.view[True](e)
    var rew = step(env, action, r)
    var d = is_game_over(env)
    b.store_rng[True](e, r)
    reward[unsafe_offset=e] = rew
    done[unsafe_offset=e] = UInt8(1) if d else UInt8(0)
    if not d:
        write_obs(env, b.obs_ptr(e))
    else:
        var slot = Atomic[Int32].fetch_add(done_count, Int32(1))
        done_idx[unsafe_offset = Int(slot)] = Int32(e)


def reset_kernel(
    ints: I32P, floats: F32P, map: U8P, mob_map: U8P, rng: U64P, obs: F32P,
    reward: F32P, done: U8P, n: Int32, done_idx: I32P, done_count: I32P,
    scratch: F32P, seed: UInt64, all_envs: Int32,
):
    """Regenerates the worlds of envs in the done list (or all envs).

    Launch with block_dim=RESET_BLOCK and any grid; blocks stride over the
    list. `scratch` holds SCRATCH_PER_BLOCK floats per block.
    """
    var t = thread_idx.x
    var b = Buffers(ints, floats, map, mob_map, rng, obs, reward, done, Int(n))
    var total = Int(n) if all_envs != 0 else Int(done_count[unsafe_offset=0])
    var tab = scratch.unsafe_offset(block_idx.x * SCRATCH_PER_BLOCK)
    var praw = tab.unsafe_offset(2 * T_SIZE)

    for j in range(block_idx.x, total, grid_dim.x):
        var e = j if all_envs != 0 else Int(done_idx[unsafe_offset=j])
        var key: UInt64
        var ctr: UInt64
        if all_envs != 0:
            key = seed_key(seed, UInt64(e))
            ctr = 0
        else:
            key = rng[unsafe_offset=e]
            ctr = rng[unsafe_offset = Int(n) + e]
        var r = SplitMix(key, ctr)
        var start = Int(ctr)
        var env = b.view[True](e)

        for k in range(t, T_SIZE, RESET_BLOCK):
            fill_table_entry(r, start, k, tab)
        barrier()

        # Pass 1: raw noise for this thread's 16 contiguous cells.
        var wv = Array[Float32, CELLS_PER_THREAD](fill=0)
        var mv = Array[Float32, CELLS_PER_THREAD](fill=0)
        var tv = Array[Float32, CELLS_PER_THREAD](fill=0)
        var mm_w = MinMax()
        var mm_m = MinMax()
        var mm_p = MinMax()
        var mm_t = MinMax()
        var c0 = t * CELLS_PER_THREAD
        for q in range(CELLS_PER_THREAD):
            var c = c0 + q
            var x = c // MAP_H
            var y = c % MAP_H
            wv[q] = water_raw(tab, x, y)
            mv[q] = mountain_raw(tab, x, y)
            tv[q] = tree_raw(tab, x, y)
            var p = path_raw(tab, x, y)
            praw[unsafe_offset=c] = p
            mm_w.add(wv[q])
            mm_m.add(mv[q])
            mm_p.add(p)
            mm_t.add(tv[q])
        var w_lo = block.min[block_size=RESET_BLOCK, broadcast=True](val=mm_w.lo)[0]
        var w_hi = block.max[block_size=RESET_BLOCK, broadcast=True](val=mm_w.hi)[0]
        var m_lo = block.min[block_size=RESET_BLOCK, broadcast=True](val=mm_m.lo)[0]
        var m_hi = block.max[block_size=RESET_BLOCK, broadcast=True](val=mm_m.hi)[0]
        var p_lo = block.min[block_size=RESET_BLOCK, broadcast=True](val=mm_p.lo)[0]
        var p_hi = block.max[block_size=RESET_BLOCK, broadcast=True](val=mm_p.hi)[0]
        var t_lo = block.min[block_size=RESET_BLOCK, broadcast=True](val=mm_t.lo)[0]
        var t_hi = block.max[block_size=RESET_BLOCK, broadcast=True](val=mm_t.hi)[0]
        mm_w = MinMax(w_lo, w_hi)
        mm_m = MinMax(m_lo, m_hi)
        mm_p = MinMax(p_lo, p_hi)
        mm_t = MinMax(t_lo, t_hi)
        barrier()  # praw complete

        # Pass 2: block rules.
        var stones = 0
        for q in range(CELLS_PER_THREAD):
            var c = c0 + q
            var x = c // MAP_H
            var y = c % MAP_H
            var blk = cell_block(
                r,
                start,
                x,
                y,
                mm_w.norm(wv[q]),
                mm_m.norm(mv[q]),
                mm_p.norm(praw[unsafe_offset=c]),
                mm_p.norm(praw[unsafe_offset = y * MAP_H + x]),
                mm_t.norm(tv[q]),
            )
            env.set_cell(c, blk)
            env.set_mob_cell(c, False)
            if blk == STONE:
                stones += 1

        # always_diamond: the k-th stone cell in row-major order.
        var total_stone = Int(
            block.sum[block_size=RESET_BLOCK, broadcast=True](val=Int32(stones))[0]
        )
        var before = Int(
            block.prefix_sum[dtype=DType.int32, block_size=RESET_BLOCK, exclusive=True](
                val=Int32(stones)
            )[0]
        )
        var ctr_after = UInt64(start + D_TOTAL)
        if total_stone > 0:
            var r2 = SplitMix(key, ctr_after)
            var k = r2.choice(total_stone)
            ctr_after += 1
            if k >= before and k < before + stones:
                var left = k - before
                for q in range(CELLS_PER_THREAD):
                    var c = c0 + q
                    if env.cell(c) == STONE:
                        if left == 0:
                            env.set_cell(c, DIAMOND)
                            break
                        left -= 1

        if t == 0:
            init_scalars(env)
            rng[unsafe_offset=e] = key
            rng[unsafe_offset = Int(n) + e] = ctr_after
        barrier()  # map and scalars complete

        var o = b.obs_ptr(e)
        for i in range(t, OBS_SIZE, RESET_BLOCK):
            o[unsafe_offset=i] = obs_value(env, i)
        barrier()  # scratch is reused by the next env


@inline(.always)
def obs_value[FM: Bool](env: EnvView[FM], i: Int) -> Float32:
    """Element i of render_craftax_symbolic (see obs.write_obs)."""
    var px = env.gi(I_PX)
    var py = env.gi(I_PY)
    if i < OBS_MAP_SIZE:
        var cell = i // OBS_CHANNELS
        var k = i % OBS_CHANNELS
        var x = px - OBS_W // 2 + cell // OBS_H
        var y = py - OBS_H // 2 + cell % OBS_H
        if k < NUM_BLOCKS:
            var blk = env.cell(x * MAP_H + y) if in_bounds(x, y) else OUT_OF_BOUNDS
            return Float32(1.0) if blk == k else Float32(0.0)
        var kind = k - NUM_BLOCKS
        for m in range(mob_count(kind)):
            if (
                env.mob_mask(kind, m)
                and env.mob_x(kind, m) == x
                and env.mob_y(kind, m) == y
            ):
                return 1.0
        return 0.0
    var o = i - OBS_MAP_SIZE
    if o < NUM_INV:
        return Float32(Float16(Float32(env.inv(o))) * Float16(0.1))
    o -= NUM_INV
    if o < 4:
        var v = env.gi(I_HEALTH + o)  # health, food, drink, energy
        return Float32(Float16(Float32(v)) * Float16(0.1))
    o -= 4
    if o < 4:
        return Float32(1.0) if o == env.gi(I_DIR) - 1 else Float32(0.0)
    o -= 4
    if o == 0:
        return env.gf(F_LIGHT)
    return Float32(1.0) if env.gb(I_SLEEPING) else Float32(0.0)

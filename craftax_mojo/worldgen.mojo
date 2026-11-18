# Port of craftax/craftax_classic/world_gen.py::generate_world and
# util/noise.py (fractal Perlin noise, octaves=1).
#
# JAX materialises five 64x64 noise maps; here each noise field keeps only its
# gradient table and is evaluated per cell, in two passes (min/max for the
# normalisation, then the block rules). Random draws are consumed in JAX's
# order: water angles, mountain angles, path angles, coal/iron/diamond
# uniforms, tree angles, tree uniforms, diamond position.

from std.memory import Pointer
from std.math import cos, sin, floor

from .constants import *
from .state import *
from .rng import Rng

comptime TWO_PI: Float32 = 6.283185307179586
comptime SQRT2: Float32 = 1.4142135623730951


@inline(.always)
def calculate_light_level(timestep: Int) -> Float32:
    # XLA rewrites t / 300 as t * (1/300) under jit; match it.
    var progress = Float32(timestep) * (Float32(1.0) / Float32(DAY_LENGTH))
    progress = progress - floor(progress) + Float32(0.3)
    var c = abs(cos(Float32(3.141592653589793) * progress))
    return Float32(1.0) - c * c * c


@inline(.always)
def _fade(t: Float32) -> Float32:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


# ---- Random draw layout ----------------------------------------------------
# JAX draws, in order: water angles (5x5), mountain angles (5x5), path angles
# (9x33), coal/iron/diamond uniforms (64x64 each), tree angles (17x17), tree
# uniforms (64x64), then the diamond position. The array draws are reserved
# as one contiguous block; these are offsets into it.
comptime D_WATER = 0
comptime D_MOUNTAIN = D_WATER + 25
comptime D_PATH = D_MOUNTAIN + 25
comptime D_COAL = D_PATH + 9 * 33
comptime D_IRON = D_COAL + MAP_CELLS
comptime D_DIAMOND = D_IRON + MAP_CELLS
comptime D_TREE = D_DIAMOND + MAP_CELLS
comptime D_TREE_U = D_TREE + 17 * 17
comptime D_TOTAL = D_TREE_U + MAP_CELLS

# Gradient table (cos, sin interleaved) offsets for the four noise fields.
comptime T_WATER = 0
comptime T_MOUNTAIN = T_WATER + 25
comptime T_PATH = T_MOUNTAIN + 25
comptime T_TREE = T_PATH + 9 * 33
comptime T_SIZE = T_TREE + 17 * 17  # 636 entries


@inline(.always)
def table_draw(k: Int) -> Int:
    """Offset of the angle draw for gradient-table entry k."""
    if k < T_TREE:
        return k  # water, mountain, path are contiguous at the start
    return D_TREE + (k - T_TREE)


@inline(.always)
def fill_table_entry[R: Rng](
    rng: R, start: Int, k: Int, tab: Pointer[Float32, MutUntrackedOrigin]
):
    var a = TWO_PI * rng.at(start, table_draw(k))
    tab[unsafe_offset = 2 * k] = cos(a)
    tab[unsafe_offset = 2 * k + 1] = sin(a)


@inline(.always)
def perlin_raw[R0: Int, R1: Int](
    tab: Pointer[Float32, MutUntrackedOrigin], i: Int, j: Int
) -> Float32:
    """Unnormalised octave-1 Perlin noise at cell (i, j) (noise.py)."""
    comptime d0 = MAP_W // R0
    comptime d1 = MAP_H // R1
    comptime W = R1 + 1
    var gx = Float32(i) * (Float32(R0) / Float32(MAP_W))
    var gy = Float32(j) * (Float32(R1) / Float32(MAP_H))
    gx = gx - floor(gx)
    gy = gy - floor(gy)
    var a = i // d0
    var b = j // d1
    var k00 = 2 * (a * W + b)
    var k10 = 2 * ((a + 1) * W + b)
    var k01 = 2 * (a * W + b + 1)
    var k11 = 2 * ((a + 1) * W + b + 1)
    var n00 = gx * tab[unsafe_offset=k00] + gy * tab[unsafe_offset = k00 + 1]
    var n10 = (gx - 1.0) * tab[unsafe_offset=k10] + gy * tab[unsafe_offset = k10 + 1]
    var n01 = gx * tab[unsafe_offset=k01] + (gy - 1.0) * tab[unsafe_offset = k01 + 1]
    var n11 = (gx - 1.0) * tab[unsafe_offset=k11] + (gy - 1.0) * tab[
        unsafe_offset = k11 + 1
    ]
    var tx = _fade(gx)
    var ty = _fade(gy)
    var n0 = n00 * (1.0 - tx) + tx * n10
    var n1 = n01 * (1.0 - tx) + tx * n11
    return SQRT2 * ((1.0 - ty) * n0 + ty * n1)


@inline(.always)
def water_raw(tab: Pointer[Float32, MutUntrackedOrigin], i: Int, j: Int) -> Float32:
    return perlin_raw[4, 4](tab.unsafe_offset(2 * T_WATER), i, j)


@inline(.always)
def mountain_raw(tab: Pointer[Float32, MutUntrackedOrigin], i: Int, j: Int) -> Float32:
    return perlin_raw[4, 4](tab.unsafe_offset(2 * T_MOUNTAIN), i, j)


@inline(.always)
def path_raw(tab: Pointer[Float32, MutUntrackedOrigin], i: Int, j: Int) -> Float32:
    return perlin_raw[8, 32](tab.unsafe_offset(2 * T_PATH), i, j)


@inline(.always)
def tree_raw(tab: Pointer[Float32, MutUntrackedOrigin], i: Int, j: Int) -> Float32:
    return perlin_raw[16, 16](tab.unsafe_offset(2 * T_TREE), i, j)


struct MinMax(Copyable):
    var lo: Float32
    var hi: Float32

    def __init__(out self):
        self.lo = Float32.MAX
        self.hi = Float32.MIN

    def __init__(out self, lo: Float32, hi: Float32):
        self.lo = lo
        self.hi = hi

    @inline(.always)
    def add(mut self, v: Float32):
        if v < self.lo:
            self.lo = v
        if v > self.hi:
            self.hi = v

    @inline(.always)
    def norm(self, v: Float32) -> Float32:
        return (v - self.lo) / (self.hi - self.lo)


@inline(.always)
def cell_block[R: Rng](
    rng: R,
    start: Int,
    x: Int,
    y: Int,
    water: Float32,
    mountain: Float32,
    path_x: Float32,
    path_y: Float32,
    tree: Float32,
) -> Int:
    """Block rules of generate_world for one cell, from normalised noise."""
    comptime cx = MAP_W // 2
    comptime cy = MAP_H // 2
    var c = x * MAP_H + y
    var prox = Float32(abs(x - cx) + abs(y - cy)) / Float32(5.0)
    prox = min(max(prox, Float32(0.0)), Float32(1.0))
    var w = water + prox - 1.0
    var m = mountain + 0.05
    m = m + prox - 1.0

    var b = WATER if w > 0.7 else GRASS
    if w < 0.75 and w > 0.6 and b != WATER:
        b = SAND
    if m > 0.7:
        b = STONE
    if m > 0.7 and path_x > 0.8:
        b = PATH
    if m > 0.7 and path_y > 0.8:
        b = PATH
    if m > 0.85 and w > 0.4:
        b = PATH
    if b == STONE and rng.at(start, D_COAL + c) < 0.04:
        b = COAL
    if b == STONE and rng.at(start, D_IRON + c) < 0.03:
        b = IRON
    if m > 0.8 and rng.at(start, D_DIAMOND + c) < 0.005 and b == STONE:
        b = DIAMOND
    var tree_gate = Float32(1.0) if tree > 0.5 else Float32(0.0)
    if tree_gate * rng.at(start, D_TREE_U + c) > 0.8 and b == GRASS:
        b = TREE
    if m > 0.85 and tree > 0.7:
        b = LAVA
    if x == cx and y == cy:
        b = GRASS  # the player spawns on grass
    return b


@inline(.always)
def init_scalars[FM: Bool](env: EnvView[FM]):
    for f in range(NUM_I):
        env.si(f, 0)
    for f in range(NUM_F):
        env.sf(f, 0.0)
    env.si(I_PX, MAP_W // 2)
    env.si(I_PY, MAP_H // 2)
    env.si(I_DIR, UP)
    env.si(I_HEALTH, 9)
    env.si(I_FOOD, 9)
    env.si(I_DRINK, 9)
    env.si(I_ENERGY, 9)
    for k in range(2 * MAX_ARROWS):
        env.si(I_ARROW_DIR + k, 1)
    env.sf(F_LIGHT, calculate_light_level(0))


def generate_world[FM: Bool, R: Rng](env: EnvView[FM], mut rng: R):
    """Single-threaded world generation (CPU and tape replay).

    Pass 1 evaluates the four noise fields once per cell into a scratch
    buffer and accumulates their min/max; pass 2 normalises and applies the
    block rules. path_y is path_x transposed, read back from the scratch.
    """
    var start = rng.reserve(D_TOTAL)
    var scratch = List[Float32](length=2 * T_SIZE + 4 * MAP_CELLS, fill=0)
    var tab = Pointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=Int(scratch.unsafe_ptr())
    )
    var raw = tab.unsafe_offset(2 * T_SIZE)
    for k in range(T_SIZE):
        fill_table_entry(rng, start, k, tab)

    var mm_w = MinMax()
    var mm_m = MinMax()
    var mm_p = MinMax()
    var mm_t = MinMax()
    for x in range(MAP_W):
        for y in range(MAP_H):
            var c = x * MAP_H + y
            var w = water_raw(tab, x, y)
            var m = mountain_raw(tab, x, y)
            var p = path_raw(tab, x, y)
            var t = tree_raw(tab, x, y)
            raw[unsafe_offset=c] = w
            raw[unsafe_offset = MAP_CELLS + c] = m
            raw[unsafe_offset = 2 * MAP_CELLS + c] = p
            raw[unsafe_offset = 3 * MAP_CELLS + c] = t
            mm_w.add(w)
            mm_m.add(m)
            mm_p.add(p)
            mm_t.add(t)

    for x in range(MAP_W):
        for y in range(MAP_H):
            var c = x * MAP_H + y
            var b = cell_block(
                rng,
                start,
                x,
                y,
                mm_w.norm(raw[unsafe_offset=c]),
                mm_m.norm(raw[unsafe_offset = MAP_CELLS + c]),
                mm_p.norm(raw[unsafe_offset = 2 * MAP_CELLS + c]),
                mm_p.norm(raw[unsafe_offset = 2 * MAP_CELLS + y * MAP_H + x]),
                mm_t.norm(raw[unsafe_offset = 3 * MAP_CELLS + c]),
            )
            env.set_cell(c, b)
            env.set_mob_cell(c, False)
    _ = scratch^

    # always_diamond: one extra diamond on a uniformly chosen stone cell.
    var diamond_index = -1
    comptime if R.IS_TAPE:
        diamond_index = rng.choice(MAP_CELLS)
    else:
        var count = 0
        for c in range(MAP_CELLS):
            if env.cell(c) == STONE:
                count += 1
        if count > 0:
            var k = rng.choice(count)
            for c in range(MAP_CELLS):
                if env.cell(c) == STONE:
                    if k == 0:
                        diamond_index = c
                        break
                    k -= 1
    if diamond_index >= 0:
        env.set_cell(diamond_index, DIAMOND)

    init_scalars(env)

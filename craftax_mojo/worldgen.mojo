# Port of craftax/craftax_classic/world_gen.py::generate_world and
# util/noise.py (fractal Perlin noise, octaves=1).
#
# JAX materialises five 64x64 noise maps; here each noise field keeps only its
# gradient table and is evaluated per cell, in two passes (min/max for the
# normalisation, then the block rules). Random draws are consumed in JAX's
# order: water angles, mountain angles, path angles, coal/iron/diamond
# uniforms, tree angles, tree uniforms, diamond position.

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


struct Perlin[R0: Int, R1: Int](Copyable):
    """Gradients of a (R0+1) x (R1+1) Perlin lattice over a 64x64 map."""

    comptime G = (Self.R0 + 1) * (Self.R1 + 1)
    var c: Array[Float32, Self.G]
    var s: Array[Float32, Self.G]

    def __init__[R: Rng](out self, mut rng: R):
        self.c = Array[Float32, Self.G](fill=0)
        self.s = Array[Float32, Self.G](fill=0)
        var base = rng.reserve(Self.G)
        for k in range(Self.G):
            var a = TWO_PI * rng.at(base, k)
            self.c[k] = cos(a)
            self.s[k] = sin(a)

    @inline(.always)
    def raw(self, i: Int, j: Int) -> Float32:
        comptime d0 = MAP_W // Self.R0
        comptime d1 = MAP_H // Self.R1
        var gx = Float32(i) * (Float32(Self.R0) / Float32(MAP_W))
        var gy = Float32(j) * (Float32(Self.R1) / Float32(MAP_H))
        gx = gx - floor(gx)
        gy = gy - floor(gy)
        var a = i // d0
        var b = j // d1
        comptime W = Self.R1 + 1
        var k00 = a * W + b
        var k10 = (a + 1) * W + b
        var k01 = a * W + b + 1
        var k11 = (a + 1) * W + b + 1
        var n00 = gx * self.c[k00] + gy * self.s[k00]
        var n10 = (gx - 1.0) * self.c[k10] + gy * self.s[k10]
        var n01 = gx * self.c[k01] + (gy - 1.0) * self.s[k01]
        var n11 = (gx - 1.0) * self.c[k11] + (gy - 1.0) * self.s[k11]
        var tx = _fade(gx)
        var ty = _fade(gy)
        var n0 = n00 * (1.0 - tx) + tx * n10
        var n1 = n01 * (1.0 - tx) + tx * n11
        return SQRT2 * ((1.0 - ty) * n0 + ty * n1)


struct MinMax(Copyable):
    var lo: Float32
    var hi: Float32

    def __init__(out self):
        self.lo = Float32.MAX
        self.hi = Float32.MIN

    @inline(.always)
    def add(mut self, v: Float32):
        if v < self.lo:
            self.lo = v
        if v > self.hi:
            self.hi = v

    @inline(.always)
    def norm(self, v: Float32) -> Float32:
        return (v - self.lo) / (self.hi - self.lo)


def generate_world[FM: Bool, R: Rng](env: EnvView[FM], mut rng: R):
    var water = Perlin[4, 4](rng)
    var mountain = Perlin[4, 4](rng)
    var path = Perlin[8, 32](rng)
    var coal_u = rng.reserve(MAP_CELLS)
    var iron_u = rng.reserve(MAP_CELLS)
    var diamond_u = rng.reserve(MAP_CELLS)
    var tree = Perlin[16, 16](rng)
    var tree_u = rng.reserve(MAP_CELLS)

    var water_mm = MinMax()
    var mountain_mm = MinMax()
    var path_mm = MinMax()
    var tree_mm = MinMax()
    for x in range(MAP_W):
        for y in range(MAP_H):
            water_mm.add(water.raw(x, y))
            mountain_mm.add(mountain.raw(x, y))
            path_mm.add(path.raw(x, y))
            tree_mm.add(tree.raw(x, y))

    comptime cx = MAP_W // 2
    comptime cy = MAP_H // 2
    for x in range(MAP_W):
        for y in range(MAP_H):
            var c = x * MAP_H + y
            var prox = Float32(abs(x - cx) + abs(y - cy)) / Float32(5.0)
            prox = min(max(prox, Float32(0.0)), Float32(1.0))
            var w = water_mm.norm(water.raw(x, y)) + prox - 1.0
            var m = mountain_mm.norm(mountain.raw(x, y)) + 0.05
            m = m + prox - 1.0
            var path_x = path_mm.norm(path.raw(x, y))
            var path_y = path_mm.norm(path.raw(y, x))  # path_x.T
            var t = tree_mm.norm(tree.raw(x, y))

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
            if b == STONE and rng.at(coal_u, c) < 0.04:
                b = COAL
            if b == STONE and rng.at(iron_u, c) < 0.03:
                b = IRON
            if m > 0.8 and rng.at(diamond_u, c) < 0.005 and b == STONE:
                b = DIAMOND
            var tree_gate = Float32(1.0) if t > 0.5 else Float32(0.0)
            if tree_gate * rng.at(tree_u, c) > 0.8 and b == GRASS:
                b = TREE
            if m > 0.85 and t > 0.7:
                b = LAVA
            env.set_cell(c, b)
            env.set_mob_cell(c, False)

    env.set_cell(cx * MAP_H + cy, GRASS)

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

    # Scalar state.
    for f in range(NUM_I):
        env.si(f, 0)
    for f in range(NUM_F):
        env.sf(f, 0.0)
    env.si(I_PX, cx)
    env.si(I_PY, cy)
    env.si(I_DIR, UP)
    env.si(I_HEALTH, 9)
    env.si(I_FOOD, 9)
    env.si(I_DRINK, 9)
    env.si(I_ENERGY, 9)
    for k in range(2 * MAX_ARROWS):
        env.si(I_ARROW_DIR + k, 1)
    env.sf(F_LIGHT, calculate_light_level(0))

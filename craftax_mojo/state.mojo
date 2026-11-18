# Flat state layout for a batch of Craftax-Classic environments.
#
# Every EnvState field is stored in one of four buffers (int32 scalars,
# float32 scalars, uint8 map, uint8 mob_map). `FM` (field-major) selects the
# memory order: field-major (`field * n + env`) gives coalesced access when one
# GPU thread owns one env; env-major (`env * stride + field`) keeps each env
# contiguous, which is what CPU caches want. The two maps are env-major in
# both layouts (see EnvView._c).

from std.memory import Pointer

from .constants import *

# ---- int32 scalar fields -------------------------------------------------
comptime I_PX = 0
comptime I_PY = 1
comptime I_DIR = 2
comptime I_HEALTH = 3
comptime I_FOOD = 4
comptime I_DRINK = 5
comptime I_ENERGY = 6
comptime I_SLEEPING = 7
comptime I_TIMESTEP = 8
comptime I_INV = 9  # 12 slots
comptime I_MOBS = I_INV + NUM_INV  # 21

# Mob kinds; each kind stores pos (2*c), health (c), mask (c), cooldown (c).
comptime ZOMBIE = 0
comptime COW = 1
comptime SKELETON = 2
comptime ARROW = 3

comptime I_ZOMBIE = I_MOBS
comptime I_COW = I_ZOMBIE + 5 * MAX_ZOMBIES
comptime I_SKELETON = I_COW + 5 * MAX_COWS
comptime I_ARROW = I_SKELETON + 5 * MAX_SKELETONS
comptime I_ARROW_DIR = I_ARROW + 5 * MAX_ARROWS  # 2 per arrow
comptime I_PLANT_POS = I_ARROW_DIR + 2 * MAX_ARROWS
comptime I_PLANT_AGE = I_PLANT_POS + 2 * MAX_PLANTS
comptime I_PLANT_MASK = I_PLANT_AGE + MAX_PLANTS
comptime I_ACH = I_PLANT_MASK + MAX_PLANTS
comptime NUM_I = I_ACH + NUM_ACHIEVEMENTS  # 144

# ---- float32 scalar fields -----------------------------------------------
comptime F_RECOVER = 0
comptime F_HUNGER = 1
comptime F_THIRST = 2
comptime F_FATIGUE = 3
comptime F_LIGHT = 4
comptime NUM_F = 5


@inline(.always)
def mob_base(kind: Int) -> Int:
    if kind == ZOMBIE:
        return I_ZOMBIE
    if kind == COW:
        return I_COW
    if kind == SKELETON:
        return I_SKELETON
    return I_ARROW


@inline(.always)
def mob_count(kind: Int) -> Int:
    if kind == ZOMBIE:
        return MAX_ZOMBIES
    if kind == COW:
        return MAX_COWS
    if kind == SKELETON:
        return MAX_SKELETONS
    return MAX_ARROWS


# JAX indexing semantics for a traced index into an axis of size `n`:
# reads normalise negatives (i + n) and then clamp; writes normalise negatives
# and drop anything still out of range.
@inline(.always)
def jax_read_index(i: Int, n: Int) -> Int:
    var j = i + n if i < 0 else i
    if j < 0:
        return 0
    if j >= n:
        return n - 1
    return j


@inline(.always)
def jax_write_index(i: Int, n: Int) -> Int:
    """Returns -1 when JAX would drop the write."""
    var j = i + n if i < 0 else i
    if j < 0 or j >= n:
        return -1
    return j


@inline(.always)
def in_bounds(x: Int, y: Int) -> Bool:
    return x >= 0 and x < MAP_W and y >= 0 and y < MAP_H


struct EnvView[FM: Bool](Copyable):
    """One environment's slice of a batch; all game logic goes through this."""

    var ints: Pointer[Int32, MutUntrackedOrigin]
    var floats: Pointer[Float32, MutUntrackedOrigin]
    var map: Pointer[UInt8, MutUntrackedOrigin]
    var mob_map: Pointer[UInt8, MutUntrackedOrigin]
    var n: Int
    var e: Int

    def __init__(
        out self,
        ints: Pointer[Int32, MutUntrackedOrigin],
        floats: Pointer[Float32, MutUntrackedOrigin],
        map: Pointer[UInt8, MutUntrackedOrigin],
        mob_map: Pointer[UInt8, MutUntrackedOrigin],
        n: Int,
        e: Int,
    ):
        self.ints = ints
        self.floats = floats
        self.map = map
        self.mob_map = mob_map
        self.n = n
        self.e = e

    # ---- addressing -------------------------------------------------------
    @inline(.always)
    def _i(self, f: Int) -> Int:
        comptime if Self.FM:
            return f * self.n + self.e
        else:
            return self.e * NUM_I + f

    @inline(.always)
    def _f(self, f: Int) -> Int:
        comptime if Self.FM:
            return f * self.n + self.e
        else:
            return self.e * NUM_F + f

    @inline(.always)
    def _c(self, cell: Int) -> Int:
        # Maps are env-major in both layouts: a thread scans its own map
        # around its own player, so contiguous per-env maps cost far fewer
        # cache lines than cell-major interleaving (which only coalesces
        # when every thread reads the same cell).
        return self.e * MAP_CELLS + cell

    # ---- scalars ----------------------------------------------------------
    @inline(.always)
    def gi(self, f: Int) -> Int:
        return Int(self.ints[unsafe_offset = self._i(f)])

    @inline(.always)
    def si(self, f: Int, v: Int):
        self.ints[unsafe_offset = self._i(f)] = Int32(v)

    @inline(.always)
    def gb(self, f: Int) -> Bool:
        return self.ints[unsafe_offset = self._i(f)] != 0

    @inline(.always)
    def sb(self, f: Int, v: Bool):
        self.ints[unsafe_offset = self._i(f)] = Int32(1) if v else Int32(0)

    @inline(.always)
    def gf(self, f: Int) -> Float32:
        return self.floats[unsafe_offset = self._f(f)]

    @inline(.always)
    def sf(self, f: Int, v: Float32):
        self.floats[unsafe_offset = self._f(f)] = v

    # ---- map (raw cell index, no bounds handling) --------------------------
    @inline(.always)
    def cell(self, c: Int) -> Int:
        return Int(self.map[unsafe_offset = self._c(c)])

    @inline(.always)
    def set_cell(self, c: Int, v: Int):
        self.map[unsafe_offset = self._c(c)] = UInt8(v)

    @inline(.always)
    def mob_cell(self, c: Int) -> Bool:
        return self.mob_map[unsafe_offset = self._c(c)] != 0

    @inline(.always)
    def set_mob_cell(self, c: Int, v: Bool):
        self.mob_map[unsafe_offset = self._c(c)] = UInt8(1) if v else UInt8(0)

    # ---- map with JAX gather / scatter semantics ---------------------------
    @inline(.always)
    def block(self, x: Int, y: Int) -> Int:
        return self.cell(
            jax_read_index(x, MAP_W) * MAP_H + jax_read_index(y, MAP_H)
        )

    @inline(.always)
    def set_block(self, x: Int, y: Int, v: Int):
        var i = jax_write_index(x, MAP_W)
        var j = jax_write_index(y, MAP_H)
        if i >= 0 and j >= 0:
            self.set_cell(i * MAP_H + j, v)

    @inline(.always)
    def mob_at(self, x: Int, y: Int) -> Bool:
        return self.mob_cell(
            jax_read_index(x, MAP_W) * MAP_H + jax_read_index(y, MAP_H)
        )

    @inline(.always)
    def set_mob_at(self, x: Int, y: Int, v: Bool):
        var i = jax_write_index(x, MAP_W)
        var j = jax_write_index(y, MAP_H)
        if i >= 0 and j >= 0:
            self.set_mob_cell(i * MAP_H + j, v)

    # ---- grouped fields ----------------------------------------------------
    @inline(.always)
    def inv(self, k: Int) -> Int:
        return self.gi(I_INV + k)

    @inline(.always)
    def set_inv(self, k: Int, v: Int):
        self.si(I_INV + k, v)

    @inline(.always)
    def ach(self, k: Int) -> Bool:
        return self.gb(I_ACH + k)

    @inline(.always)
    def grant(self, k: Int, cond: Bool):
        if cond:
            self.sb(I_ACH + k, True)

    @inline(.always)
    def num_ach(self) -> Int:
        var s = 0
        for k in range(NUM_ACHIEVEMENTS):
            s += self.gi(I_ACH + k)
        return s

    @inline(.always)
    def mob_x(self, kind: Int, i: Int) -> Int:
        return self.gi(mob_base(kind) + 2 * i)

    @inline(.always)
    def mob_y(self, kind: Int, i: Int) -> Int:
        return self.gi(mob_base(kind) + 2 * i + 1)

    @inline(.always)
    def set_mob_pos(self, kind: Int, i: Int, x: Int, y: Int):
        self.si(mob_base(kind) + 2 * i, x)
        self.si(mob_base(kind) + 2 * i + 1, y)

    @inline(.always)
    def mob_health(self, kind: Int, i: Int) -> Int:
        return self.gi(mob_base(kind) + 2 * mob_count(kind) + i)

    @inline(.always)
    def set_mob_health(self, kind: Int, i: Int, v: Int):
        self.si(mob_base(kind) + 2 * mob_count(kind) + i, v)

    @inline(.always)
    def mob_mask(self, kind: Int, i: Int) -> Bool:
        return self.gb(mob_base(kind) + 3 * mob_count(kind) + i)

    @inline(.always)
    def set_mob_mask(self, kind: Int, i: Int, v: Bool):
        self.sb(mob_base(kind) + 3 * mob_count(kind) + i, v)

    @inline(.always)
    def mob_cd(self, kind: Int, i: Int) -> Int:
        return self.gi(mob_base(kind) + 4 * mob_count(kind) + i)

    @inline(.always)
    def set_mob_cd(self, kind: Int, i: Int, v: Int):
        self.si(mob_base(kind) + 4 * mob_count(kind) + i, v)

    @inline(.always)
    def is_in_mob(self, x: Int, y: Int) -> Bool:
        return self.mob_at(x, y) or (
            x == self.gi(I_PX) and y == self.gi(I_PY)
        )

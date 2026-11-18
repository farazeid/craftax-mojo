# Port of craftax/craftax_classic/game_logic.py::craftax_step.
#
# The JAX code computes every branch and selects results with lax.select; this
# port applies the same updates imperatively, in the same order, reading the
# same (pre- or post-update) values. Comments name the JAX function each block
# mirrors. All random draws go through `Rng` in JAX's order (25 per step).

from std.math import fma

from .constants import *
from .state import *
from .rng import Rng
from .worldgen import calculate_light_level


@inline(.always)
def _sign(v: Int) -> Int:
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


@inline(.always)
def is_game_over[FM: Bool](env: EnvView[FM]) -> Bool:
    return (
        env.gi(I_TIMESTEP) >= MAX_TIMESTEPS
        or env.block(env.gi(I_PX), env.gi(I_PY)) == LAVA
        or env.gi(I_HEALTH) <= 0
    )


@inline(.always)
def _valid_mob_move[FM: Bool](env: EnvView[FM], x: Int, y: Int) -> Bool:
    # is_position_in_bounds_not_in_wall_not_in_mob_not_in_lava
    var b = env.block(x, y)
    return (
        in_bounds(x, y)
        and not is_solid(b)
        and not env.is_in_mob(x, y)
        and b != LAVA
    )


@inline(.always)
def _is_near_block[FM: Bool](env: EnvView[FM], block: Int) -> Bool:
    var px = env.gi(I_PX)
    var py = env.gi(I_PY)
    for k in range(8):
        var x = px + close_dx(k)
        var y = py + close_dy(k)
        if in_bounds(x, y) and env.block(x, y) == block:
            return True
    return False


def do_crafting[FM: Bool](env: EnvView[FM], action: Int):
    if action < MAKE_WOOD_PICKAXE:
        return
    var table = _is_near_block(env, CRAFTING_TABLE)
    var furnace = _is_near_block(env, FURNACE)
    var wood = env.inv(INV_WOOD)
    var stone = env.inv(INV_STONE)
    var coal = env.inv(INV_COAL)
    var iron = env.inv(INV_IRON)
    if action == MAKE_WOOD_PICKAXE and wood >= 1 and table:
        env.set_inv(INV_WOOD, wood - 1)
        env.set_inv(INV_WOOD_PICKAXE, env.inv(INV_WOOD_PICKAXE) + 1)
        env.grant(ACH_MAKE_WOOD_PICKAXE, True)
    elif action == MAKE_STONE_PICKAXE and wood >= 1 and stone >= 1 and table:
        env.set_inv(INV_STONE, stone - 1)
        env.set_inv(INV_WOOD, wood - 1)
        env.set_inv(INV_STONE_PICKAXE, env.inv(INV_STONE_PICKAXE) + 1)
        env.grant(ACH_MAKE_STONE_PICKAXE, True)
    elif (
        action == MAKE_IRON_PICKAXE
        and wood >= 1
        and stone >= 1
        and iron >= 1
        and coal >= 1
        and furnace
        and table
    ):
        env.set_inv(INV_IRON, iron - 1)
        env.set_inv(INV_WOOD, wood - 1)
        env.set_inv(INV_STONE, stone - 1)
        env.set_inv(INV_COAL, coal - 1)
        env.set_inv(INV_IRON_PICKAXE, env.inv(INV_IRON_PICKAXE) + 1)
        env.grant(ACH_MAKE_IRON_PICKAXE, True)
    elif action == MAKE_WOOD_SWORD and wood >= 1 and table:
        env.set_inv(INV_WOOD, wood - 1)
        env.set_inv(INV_WOOD_SWORD, env.inv(INV_WOOD_SWORD) + 1)
        env.grant(ACH_MAKE_WOOD_SWORD, True)
    elif action == MAKE_STONE_SWORD and stone >= 1 and wood >= 1 and table:
        env.set_inv(INV_WOOD, wood - 1)
        env.set_inv(INV_STONE, stone - 1)
        env.set_inv(INV_STONE_SWORD, env.inv(INV_STONE_SWORD) + 1)
        env.grant(ACH_MAKE_STONE_SWORD, True)
    elif (
        action == MAKE_IRON_SWORD
        and iron >= 1
        and wood >= 1
        and stone >= 1
        and coal >= 1
        and furnace
        and table
    ):
        env.set_inv(INV_WOOD, wood - 1)
        env.set_inv(INV_IRON, iron - 1)
        env.set_inv(INV_STONE, stone - 1)
        env.set_inv(INV_COAL, coal - 1)
        env.set_inv(INV_IRON_SWORD, env.inv(INV_IRON_SWORD) + 1)
        env.grant(ACH_MAKE_IRON_SWORD, True)


@inline(.always)
def _attack_mobs[
    FM: Bool
](env: EnvView[FM], kind: Int, bx: Int, by: Int, damage: Int) -> Tuple[
    Bool, Bool
]:
    """Returns (is_attacking, did_kill) for one mob kind (do_action)."""
    var n = mob_count(kind)
    var target = 0
    var attacking = False
    for i in range(n):
        if (
            env.mob_x(kind, i) == bx
            and env.mob_y(kind, i) == by
            and env.mob_mask(kind, i)
        ):
            target = i
            attacking = True
            break
    if attacking:
        env.set_mob_health(kind, target, env.mob_health(kind, target) - damage)
    var old_mask = env.mob_mask(kind, target)
    # mask = (health > 0) & mask, applied to every slot
    for i in range(n):
        if env.mob_health(kind, i) <= 0:
            env.set_mob_mask(kind, i, False)
    var did_kill = old_mask and not env.mob_mask(kind, target)
    return (attacking, did_kill)


def do_action[FM: Bool, R: Rng](env: EnvView[FM], action: Int, mut rng: R):
    if action != DO:
        rng.skip()  # the sapling draw JAX makes regardless of the action
        return

    var dir = env.gi(I_DIR)
    var bx = env.gi(I_PX) + dir_dx(dir)
    var by = env.gi(I_PY) + dir_dy(dir)

    var damage = max(
        max(1, 2 * env.inv(INV_WOOD_SWORD)),
        max(3 * env.inv(INV_STONE_SWORD), 5 * env.inv(INV_IRON_SWORD)),
    )

    var z = _attack_mobs(env, ZOMBIE, bx, by, damage)
    env.grant(ACH_DEFEAT_ZOMBIE, z[1])

    var c = _attack_mobs(env, COW, bx, by, damage)
    env.grant(ACH_EAT_COW, c[1])
    if c[1]:
        env.si(I_FOOD, min(9, env.gi(I_FOOD) + 6))
        env.sf(F_HUNGER, 0.0)

    var s = _attack_mobs(env, SKELETON, bx, by, damage)
    env.grant(ACH_DEFEAT_SKELETON, s[1])

    var did_attack_mob = z[0] or c[0] or s[0]
    var did_kill_mob = z[1] or c[1] or s[1]
    env.set_mob_at(bx, by, env.mob_at(bx, by) and not did_kill_mob)

    var b = env.block(bx, by)
    var sapling_roll = rng.uniform()

    # Block effects are discarded (lax.select) when the target is out of
    # bounds or a mob was attacked.
    if not in_bounds(bx, by) or did_attack_mob:
        return

    if b == TREE:
        env.set_block(bx, by, GRASS)
        env.set_inv(INV_WOOD, env.inv(INV_WOOD) + 1)
        env.grant(ACH_COLLECT_WOOD, True)
    elif b == STONE and env.inv(INV_WOOD_PICKAXE) != 0:
        env.set_block(bx, by, PATH)
        env.set_inv(INV_STONE, env.inv(INV_STONE) + 1)
        env.grant(ACH_COLLECT_STONE, True)
    elif b == COAL and env.inv(INV_WOOD_PICKAXE) != 0:
        env.set_block(bx, by, PATH)
        env.set_inv(INV_COAL, env.inv(INV_COAL) + 1)
        env.grant(ACH_COLLECT_COAL, True)
    elif b == IRON and env.inv(INV_STONE_PICKAXE) != 0:
        env.set_block(bx, by, PATH)
        env.set_inv(INV_IRON, env.inv(INV_IRON) + 1)
        env.grant(ACH_COLLECT_IRON, True)
    elif b == DIAMOND and env.inv(INV_IRON_PICKAXE) != 0:
        env.set_block(bx, by, PATH)
        env.set_inv(INV_DIAMOND, env.inv(INV_DIAMOND) + 1)
        env.grant(ACH_COLLECT_DIAMOND, True)
    elif b == GRASS and sapling_roll < 0.1:
        env.set_inv(INV_SAPLING, env.inv(INV_SAPLING) + 1)
        env.grant(ACH_COLLECT_SAPLING, True)
    elif b == WATER:
        env.si(I_DRINK, min(9, env.gi(I_DRINK) + 1))
        env.sf(F_THIRST, 0.0)
        env.grant(ACH_COLLECT_DRINK, True)
    elif b == RIPE_PLANT:
        env.set_block(bx, by, PLANT)
        env.si(I_FOOD, min(9, env.gi(I_FOOD) + 4))
        env.sf(F_HUNGER, 0.0)
        env.grant(ACH_EAT_PLANT, True)
        # update_plants_with_eat: argmax over matching positions (0 if none)
        var idx = 0
        for i in range(MAX_PLANTS):
            if (
                env.gi(I_PLANT_POS + 2 * i) == bx
                and env.gi(I_PLANT_POS + 2 * i + 1) == by
            ):
                idx = i
                break
        env.si(I_PLANT_AGE + idx, 0)


def place_block[FM: Bool](env: EnvView[FM], action: Int):
    if action < PLACE_STONE or action > PLACE_PLANT:
        return
    var dir = env.gi(I_DIR)
    var x = env.gi(I_PX) + dir_dx(dir)
    var y = env.gi(I_PY) + dir_dy(dir)
    # Everything is reverted unless the target is in bounds and mob-free.
    if not in_bounds(x, y) or env.is_in_mob(x, y):
        return
    var b = env.block(x, y)
    var in_wall = is_solid(b)
    if action == PLACE_TABLE:
        if not in_wall and env.inv(INV_WOOD) >= 2:
            env.set_block(x, y, CRAFTING_TABLE)
            env.set_inv(INV_WOOD, env.inv(INV_WOOD) - 2)
            env.grant(ACH_PLACE_TABLE, True)
    elif action == PLACE_FURNACE:
        if not in_wall and env.inv(INV_STONE) > 0:
            env.set_block(x, y, FURNACE)
            env.set_inv(INV_STONE, env.inv(INV_STONE) - 1)
            env.grant(ACH_PLACE_FURNACE, True)
    elif action == PLACE_STONE:
        if (b == WATER or not in_wall) and env.inv(INV_STONE) > 0:
            env.set_block(x, y, STONE)
            env.set_inv(INV_STONE, env.inv(INV_STONE) - 1)
            env.grant(ACH_PLACE_STONE, True)
    else:  # PLACE_PLANT
        if b == GRASS and env.inv(INV_SAPLING) > 0:
            env.set_block(x, y, PLANT)
            env.set_inv(INV_SAPLING, env.inv(INV_SAPLING) - 1)
            env.grant(ACH_PLACE_PLANT, True)
            # add_new_growing_plant: first empty slot, if any
            for i in range(MAX_PLANTS):
                if not env.gb(I_PLANT_MASK + i):
                    env.si(I_PLANT_POS + 2 * i, x)
                    env.si(I_PLANT_POS + 2 * i + 1, y)
                    env.si(I_PLANT_AGE + i, 0)
                    env.sb(I_PLANT_MASK + i, True)
                    break


def move_player[FM: Bool](env: EnvView[FM], action: Int):
    var dx = dir_dx(action)
    var dy = dir_dy(action)
    var px = env.gi(I_PX)
    var py = env.gi(I_PY)
    var x = px + dx
    var y = py + dy
    var valid = _valid_mob_move(env, x, y) or env.block(x, y) == LAVA
    if valid:
        env.si(I_PX, x)
        env.si(I_PY, y)
    if dx != 0 or dy != 0:
        env.si(I_DIR, action)


@inline(.always)
def _toward_player[
    FM: Bool, R: Rng
](env: EnvView[FM], mx: Int, my: Int, mut rng: R) -> Tuple[Int, Int]:
    """The 'move towards player' direction of zombies and skeletons.

    JAX samples which axis to move along with p proportional to
    (|d| == max|d|); ties (including d == 0) are uniform.
    """
    var ddx = env.gi(I_PX) - mx
    var ddy = env.gi(I_PY) - my
    var ax = abs(ddx)
    var ay = abs(ddy)
    var axis: Int
    comptime if R.IS_TAPE:
        axis = rng.choice(2)
    else:
        if ax == ay:
            axis = rng.choice(2)
        else:
            axis = 0 if ax > ay else 1
    if axis == 0:
        return (_sign(ddx), 0)
    return (0, _sign(ddy))


@inline(.always)
def _random_dir[R: Rng](mut rng: R, n: Int) -> Int:
    """jax.random.choice over DIRECTIONS[1:1+n]; returns the action index."""
    return rng.choice(n) + 1


@inline(.always)
def _relocate[
    FM: Bool
](env: EnvView[FM], kind: Int, i: Int, ox: Int, oy: Int, nx: Int, ny: Int):
    """Mob bookkeeping shared by zombies, cows and skeletons."""
    var mask = env.mob_mask(kind, i)
    var keep = (
        abs(ox - env.gi(I_PX)) + abs(oy - env.gi(I_PY)) < MOB_DESPAWN_DISTANCE
    )
    env.set_mob_at(ox, oy, env.mob_at(ox, oy) and not mask)
    var new_mask = mask and keep
    env.set_mob_at(nx, ny, env.mob_at(nx, ny) or new_mask)
    env.set_mob_pos(kind, i, nx, ny)
    env.set_mob_mask(kind, i, new_mask)


def update_mobs[FM: Bool, R: Rng](env: EnvView[FM], mut rng: R):
    # Zombies
    for i in range(MAX_ZOMBIES):
        var zx = env.mob_x(ZOMBIE, i)
        var zy = env.mob_y(ZOMBIE, i)
        var rd = _random_dir(rng, 4)
        var tp = _toward_player(env, zx, zy, rng)
        var dist = abs(zx - env.gi(I_PX)) + abs(zy - env.gi(I_PY))
        var close = dist < 10
        var roll = rng.uniform()
        close = close and roll < 0.75
        var nx = zx + (tp[0] if close else dir_dx(rd))
        var ny = zy + (tp[1] if close else dir_dy(rd))

        var cd = env.mob_cd(ZOMBIE, i)
        var attacking = dist == 1 and cd <= 0 and env.mob_mask(ZOMBIE, i)
        if attacking:
            nx = zx
            ny = zy
            var sleeping = env.gb(I_SLEEPING)
            env.si(I_HEALTH, env.gi(I_HEALTH) - (7 if sleeping else 2))
            env.sb(I_SLEEPING, False)
            env.grant(ACH_WAKE_UP, sleeping)
            env.set_mob_cd(ZOMBIE, i, 5)
        else:
            env.set_mob_cd(ZOMBIE, i, cd - 1)

        if not _valid_mob_move(env, nx, ny):
            nx = zx
            ny = zy
        _relocate(env, ZOMBIE, i, zx, zy, nx, ny)

    # Cows: DIRECTIONS[1:9], half of which are no-ops
    for i in range(MAX_COWS):
        var cx = env.mob_x(COW, i)
        var cy = env.mob_y(COW, i)
        var rd = _random_dir(rng, 8)
        var nx = cx + dir_dx(rd)
        var ny = cy + dir_dy(rd)
        if not _valid_mob_move(env, nx, ny):
            nx = cx
            ny = cy
        _relocate(env, COW, i, cx, cy, nx, ny)

    # Skeletons
    for i in range(MAX_SKELETONS):
        var sx = env.mob_x(SKELETON, i)
        var sy = env.mob_y(SKELETON, i)
        var rd = _random_dir(rng, 4)
        var rx = sx + dir_dx(rd)
        var ry = sy + dir_dy(rd)
        var tp = _toward_player(env, sx, sy, rng)
        var dist = abs(sx - env.gi(I_PX)) + abs(sy - env.gi(I_PY))
        var far = dist >= 10
        var too_close = dist <= 3
        var nx = rx
        var ny = ry
        if far:
            nx = sx + tp[0]
            ny = sy + tp[1]
        if too_close:
            nx = sx - tp[0]
            ny = sy - tp[1]
        var roll = rng.uniform()
        if not (roll > 0.85):
            nx = rx
            ny = ry

        var cd = env.mob_cd(SKELETON, i)
        var attacking = (dist >= 4 and dist <= 5) or (
            too_close and not _valid_mob_move(env, nx, ny)
        )
        attacking = attacking and cd <= 0 and env.mob_mask(SKELETON, i)

        # Spawn an arrow in the first free slot.
        var live_arrows = 0
        var slot = -1
        for a in range(MAX_ARROWS):
            if env.mob_mask(ARROW, a):
                live_arrows += 1
            elif slot < 0:
                slot = a
        if attacking and live_arrows < MAX_ARROWS:
            env.set_mob_pos(ARROW, slot, sx, sy)
            env.set_mob_mask(ARROW, slot, True)
            env.si(I_ARROW_DIR + 2 * slot, tp[0])
            env.si(I_ARROW_DIR + 2 * slot + 1, tp[1])

        if attacking:
            nx = sx
            ny = sy
            env.set_mob_cd(SKELETON, i, 4)
        else:
            env.set_mob_cd(SKELETON, i, cd - 1)

        if not _valid_mob_move(env, nx, ny):
            nx = sx
            ny = sy
        _relocate(env, SKELETON, i, sx, sy, nx, ny)

    # Arrows (dead arrows keep flying; their writes are no-ops or dropped)
    for i in range(MAX_ARROWS):
        var x = env.mob_x(ARROW, i) + env.gi(I_ARROW_DIR + 2 * i)
        var y = env.mob_y(ARROW, i) + env.gi(I_ARROW_DIR + 2 * i + 1)
        var mask = env.mob_mask(ARROW, i)
        var b = env.block(x, y)
        var in_wall = is_solid(b) and b != WATER  # arrows fly over water
        var keep = in_bounds(x, y) and not in_wall and not env.is_in_mob(x, y)
        var hit_player = (
            x == env.gi(I_PX) and y == env.gi(I_PY) and mask
        )
        var removing = (b == FURNACE or b == CRAFTING_TABLE) and mask
        env.set_mob_pos(ARROW, i, x, y)
        env.set_mob_mask(ARROW, i, keep and mask)
        if hit_player:
            env.si(I_HEALTH, env.gi(I_HEALTH) - 2)
            env.sb(I_SLEEPING, False)
        env.set_block(x, y, PATH if removing else b)


@inline(.always)
def _spawn_ok[
    FM: Bool
](env: EnvView[FM], kind: Int, c: Int, dist: Int) -> Bool:
    var b = env.cell(c)
    if env.mob_cell(c) or dist >= MOB_DESPAWN_DISTANCE:
        return False
    if kind == COW:
        return b == GRASS and dist > 3
    if kind == ZOMBIE:
        return (b == GRASS or b == PATH) and dist > 9
    return b == PATH and dist > 9


def _spawn_cell[
    FM: Bool, R: Rng
](env: EnvView[FM], kind: Int, mut rng: R) -> Int:
    """Uniform choice over valid spawn cells; -1 if there are none.

    Valid cells are all within the despawn distance, so only the diamond
    |dx| + |dy| < 14 around the player is scanned (JAX scans the full map).
    """
    var px = env.gi(I_PX)
    var py = env.gi(I_PY)
    comptime R_ = MOB_DESPAWN_DISTANCE - 1
    var x0 = max(px - R_, 0)
    var x1 = min(px + R_, MAP_W - 1)
    var count = 0
    for x in range(x0, x1 + 1):
        var r = R_ - abs(x - px)
        for y in range(max(py - r, 0), min(py + r, MAP_H - 1) + 1):
            if _spawn_ok(env, kind, x * MAP_H + y, abs(x - px) + abs(y - py)):
                count += 1
    if count == 0:
        return -1
    var k = rng.choice(count)
    for x in range(x0, x1 + 1):
        var r = R_ - abs(x - px)
        for y in range(max(py - r, 0), min(py + r, MAP_H - 1) + 1):
            var c = x * MAP_H + y
            if _spawn_ok(env, kind, c, abs(x - px) + abs(y - py)):
                if k == 0:
                    return c
                k -= 1
    return -1


def _spawn_one[
    FM: Bool, R: Rng
](env: EnvView[FM], kind: Int, chance: Float32, health: Int, mut rng: R):
    var n = mob_count(kind)
    var live = 0
    var slot = -1
    for i in range(n):
        if env.mob_mask(kind, i):
            live += 1
        elif slot < 0:
            slot = i
    if slot < 0:
        slot = 0  # argmax of all-False
    var roll = rng.uniform()
    var can = live < n and roll < chance

    var cell = -1
    comptime if R.IS_TAPE:
        var picked = rng.choice(MAP_CELLS)
        if can and picked >= 0 and picked < MAP_CELLS:
            var px = env.gi(I_PX)
            var py = env.gi(I_PY)
            var x = picked // MAP_W
            var y = picked % MAP_H
            # JAX also requires at least one valid cell; the picked one is
            # valid whenever any is.
            if _spawn_ok(env, kind, picked, abs(x - px) + abs(y - py)):
                cell = picked
    else:
        if can:
            cell = _spawn_cell(env, kind, rng)

    if cell >= 0:
        env.set_mob_pos(kind, slot, cell // MAP_W, cell % MAP_H)
        env.set_mob_health(kind, slot, health)
        env.set_mob_mask(kind, slot, True)
    # mob_map[pos[slot]] |= mask[slot], even when nothing spawned
    var sx = env.mob_x(kind, slot)
    var sy = env.mob_y(kind, slot)
    env.set_mob_at(sx, sy, env.mob_at(sx, sy) or env.mob_mask(kind, slot))


def spawn_mobs[FM: Bool, R: Rng](env: EnvView[FM], mut rng: R):
    _spawn_one(env, COW, SPAWN_COW_CHANCE, COW_HEALTH, rng)
    var dark = Float32(1.0) - env.gf(F_LIGHT)
    var zombie_chance = SPAWN_ZOMBIE_BASE_CHANCE + SPAWN_ZOMBIE_NIGHT_CHANCE * (
        dark * dark
    )
    _spawn_one(env, ZOMBIE, zombie_chance, ZOMBIE_HEALTH, rng)
    _spawn_one(env, SKELETON, SPAWN_SKELETON_CHANCE, SKELETON_HEALTH, rng)


def update_plants[FM: Bool](env: EnvView[FM]):
    for i in range(MAX_PLANTS):
        var age = env.gi(I_PLANT_AGE + i) + 1
        if not env.gb(I_PLANT_MASK + i):
            age = 0
        env.si(I_PLANT_AGE + i, age)
        if age >= 600:
            env.set_block(
                env.gi(I_PLANT_POS + 2 * i),
                env.gi(I_PLANT_POS + 2 * i + 1),
                RIPE_PLANT,
            )


def update_player_intrinsics[FM: Bool](env: EnvView[FM], action: Int):
    var sleeping = env.gb(I_SLEEPING)
    var energy = env.gi(I_ENERGY)
    if action == SLEEP and energy < 9:
        sleeping = True
    if energy >= 9 and sleeping:
        sleeping = False
        env.grant(ACH_WAKE_UP, True)
    env.sb(I_SLEEPING, sleeping)

    var rate: Float32 = 0.5 if sleeping else 1.0

    var hunger = env.gf(F_HUNGER) + rate
    if hunger > 25:
        env.si(I_FOOD, max(env.gi(I_FOOD) - 1, 0))
        hunger = 0.0
    env.sf(F_HUNGER, hunger)

    var thirst = env.gf(F_THIRST) + rate
    if thirst > 20:
        env.si(I_DRINK, max(env.gi(I_DRINK) - 1, 0))
        thirst = 0.0
    env.sf(F_THIRST, thirst)

    var fatigue = env.gf(F_FATIGUE)
    fatigue = min(fatigue - 1.0, Float32(0.0)) if sleeping else fatigue + 1.0
    if fatigue > 30:
        energy = max(energy - 1, 0)
        fatigue = 0.0
    if fatigue < -10:
        energy = min(env.gi(I_ENERGY) + 1, 9)
        fatigue = 0.0
    env.sf(F_FATIGUE, fatigue)
    env.si(I_ENERGY, energy)

    var all_ok = (
        env.gi(I_FOOD) > 0 and env.gi(I_DRINK) > 0 and (energy > 0 or sleeping)
    )
    var recover_add: Float32
    if all_ok:
        recover_add = 2.0 if sleeping else 1.0
    else:
        recover_add = -0.5 if sleeping else -1.0
    var recover = env.gf(F_RECOVER) + recover_add
    var health = env.gi(I_HEALTH)
    if recover > 25:
        health = min(health + 1, 9)
        recover = 0.0
    if recover < -15:
        health = env.gi(I_HEALTH) - 1
        recover = 0.0
    env.sf(F_RECOVER, recover)
    env.si(I_HEALTH, health)


def step[FM: Bool, R: Rng](env: EnvView[FM], action_in: Int, mut rng: R) -> Float32:
    """craftax_step: advances one env in place and returns the reward."""
    var init_ach = env.num_ach()
    var init_health = env.gi(I_HEALTH)

    var action = NOOP if env.gb(I_SLEEPING) else action_in

    do_crafting(env, action)
    do_action(env, action, rng)
    place_block(env, action)
    move_player(env, action)
    update_mobs(env, rng)
    spawn_mobs(env, rng)
    update_plants(env)
    update_player_intrinsics(env, action)

    # cap_inventory
    for k in range(NUM_INV):
        env.set_inv(k, min(env.inv(k), 9))

    # update_health
    var health = env.gi(I_HEALTH)
    if env.block(env.gi(I_PX), env.gi(I_PY)) == LAVA:
        health = 0
    health = max(0, health)
    env.si(I_HEALTH, health)

    # XLA fuses this into one FMA; spell it out so every build (we compile
    # with FMA contraction off) rounds the same way.
    var reward = fma(
        Float32(health - init_health), Float32(0.1), Float32(env.num_ach() - init_ach)
    )

    var t = env.gi(I_TIMESTEP) + 1
    env.si(I_TIMESTEP, t)
    env.sf(F_LIGHT, calculate_light_level(t))
    return reward

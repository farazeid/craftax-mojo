# Constants mirrored one-to-one from craftax/craftax_classic/constants.py and
# craftax_state.py (EnvParams / StaticEnvParams defaults).

# ---- Static params -------------------------------------------------------
comptime MAP_W = 64
comptime MAP_H = 64
comptime MAP_CELLS = MAP_W * MAP_H
comptime OBS_W = 7
comptime OBS_H = 9
comptime MAX_ZOMBIES = 3
comptime MAX_COWS = 3
comptime MAX_SKELETONS = 2
comptime MAX_ARROWS = 3
comptime MAX_PLANTS = 10

# ---- Params --------------------------------------------------------------
comptime MAX_TIMESTEPS = 10000
comptime DAY_LENGTH = 300
comptime ZOMBIE_HEALTH = 5
comptime COW_HEALTH = 3
comptime SKELETON_HEALTH = 3
comptime MOB_DESPAWN_DISTANCE = 14
comptime SPAWN_COW_CHANCE: Float32 = 0.1
comptime SPAWN_ZOMBIE_BASE_CHANCE: Float32 = 0.02
comptime SPAWN_ZOMBIE_NIGHT_CHANCE: Float32 = 0.1
comptime SPAWN_SKELETON_CHANCE: Float32 = 0.05

# ---- BlockType -----------------------------------------------------------
comptime INVALID = 0
comptime OUT_OF_BOUNDS = 1
comptime GRASS = 2
comptime WATER = 3
comptime STONE = 4
comptime TREE = 5
comptime WOOD = 6
comptime PATH = 7
comptime COAL = 8
comptime IRON = 9
comptime DIAMOND = 10
comptime CRAFTING_TABLE = 11
comptime FURNACE = 12
comptime SAND = 13
comptime LAVA = 14
comptime PLANT = 15
comptime RIPE_PLANT = 16
comptime NUM_BLOCKS = 17

# ---- Action --------------------------------------------------------------
comptime NOOP = 0
comptime LEFT = 1
comptime RIGHT = 2
comptime UP = 3
comptime DOWN = 4
comptime DO = 5
comptime SLEEP = 6
comptime PLACE_STONE = 7
comptime PLACE_TABLE = 8
comptime PLACE_FURNACE = 9
comptime PLACE_PLANT = 10
comptime MAKE_WOOD_PICKAXE = 11
comptime MAKE_STONE_PICKAXE = 12
comptime MAKE_IRON_PICKAXE = 13
comptime MAKE_WOOD_SWORD = 14
comptime MAKE_STONE_SWORD = 15
comptime MAKE_IRON_SWORD = 16
comptime NUM_ACTIONS = 17

# ---- Achievement ---------------------------------------------------------
comptime ACH_COLLECT_WOOD = 0
comptime ACH_PLACE_TABLE = 1
comptime ACH_EAT_COW = 2
comptime ACH_COLLECT_SAPLING = 3
comptime ACH_COLLECT_DRINK = 4
comptime ACH_MAKE_WOOD_PICKAXE = 5
comptime ACH_MAKE_WOOD_SWORD = 6
comptime ACH_PLACE_PLANT = 7
comptime ACH_DEFEAT_ZOMBIE = 8
comptime ACH_COLLECT_STONE = 9
comptime ACH_PLACE_STONE = 10
comptime ACH_EAT_PLANT = 11
comptime ACH_DEFEAT_SKELETON = 12
comptime ACH_MAKE_STONE_PICKAXE = 13
comptime ACH_MAKE_STONE_SWORD = 14
comptime ACH_WAKE_UP = 15
comptime ACH_PLACE_FURNACE = 16
comptime ACH_COLLECT_COAL = 17
comptime ACH_COLLECT_IRON = 18
comptime ACH_COLLECT_DIAMOND = 19
comptime ACH_MAKE_IRON_PICKAXE = 20
comptime ACH_MAKE_IRON_SWORD = 21
comptime NUM_ACHIEVEMENTS = 22

# ---- Inventory slots (order matches Inventory dataclass) -----------------
comptime INV_WOOD = 0
comptime INV_STONE = 1
comptime INV_COAL = 2
comptime INV_IRON = 3
comptime INV_DIAMOND = 4
comptime INV_SAPLING = 5
comptime INV_WOOD_PICKAXE = 6
comptime INV_STONE_PICKAXE = 7
comptime INV_IRON_PICKAXE = 8
comptime INV_WOOD_SWORD = 9
comptime INV_STONE_SWORD = 10
comptime INV_IRON_SWORD = 11
comptime NUM_INV = 12

# ---- Observation ---------------------------------------------------------
comptime OBS_CHANNELS = NUM_BLOCKS + 4
comptime OBS_MAP_SIZE = OBS_W * OBS_H * OBS_CHANNELS
comptime OBS_SIZE = OBS_MAP_SIZE + NUM_INV + 4 + 4 + 2  # 1345


# DIRECTIONS: rows 0..4 are [0,0],[0,-1],[0,1],[-1,0],[1,0]; rows 5..15 are
# zero. Cows sample from DIRECTIONS[1:9], so rows 5..8 (no-op) matter.
@inline(.always)
def dir_dx(a: Int) -> Int:
    if a == UP:
        return -1
    if a == DOWN:
        return 1
    return 0


@inline(.always)
def dir_dy(a: Int) -> Int:
    if a == LEFT:
        return -1
    if a == RIGHT:
        return 1
    return 0


# CLOSE_BLOCKS
@inline(.always)
def close_dx(k: Int) -> Int:
    # [0,-1],[0,1],[-1,0],[1,0],[-1,-1],[-1,1],[1,-1],[1,1]
    if k == 2 or k == 4 or k == 5:
        return -1
    if k == 3 or k == 6 or k == 7:
        return 1
    return 0


@inline(.always)
def close_dy(k: Int) -> Int:
    if k == 0 or k == 4 or k == 6:
        return -1
    if k == 1 or k == 5 or k == 7:
        return 1
    return 0


@inline(.always)
def is_solid(b: Int) -> Bool:
    # SOLID_BLOCKS
    return (
        b == WATER
        or b == STONE
        or b == TREE
        or b == COAL
        or b == IRON
        or b == DIAMOND
        or b == CRAFTING_TABLE
        or b == FURNACE
        or b == PLANT
        or b == RIPE_PLANT
    )

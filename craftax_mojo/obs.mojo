# Port of craftax/craftax_classic/renderer.py::render_craftax_symbolic.
# Layout (1345 float32): 7x9x21 map view (17 block one-hot + 4 mob channels,
# flattened row-major), 12 inventory, 4 intrinsics, 4 direction one-hot,
# light level, is_sleeping.

from std.memory import Pointer

from .constants import *
from .state import *


@inline(.always)
def _f16_tenth(v: Int) -> Float32:
    # JAX: x.astype(float16) / 10.0, then promoted to float32. Under jit XLA
    # rewrites the division as a multiply by the float16 reciprocal (0.1 ->
    # 0.0999756), which rounds differently for 3, 6 and 7; match jit.
    return Float32(Float16(Float32(v)) * Float16(0.1))


def write_obs[FM: Bool](env: EnvView[FM], obs: Pointer[Float32, MutUntrackedOrigin]):
    var px = env.gi(I_PX)
    var py = env.gi(I_PY)
    comptime hw = OBS_W // 2
    comptime hh = OBS_H // 2

    for i in range(OBS_W):
        for j in range(OBS_H):
            var x = px - hw + i
            var y = py - hh + j
            var b = env.cell(x * MAP_H + y) if in_bounds(x, y) else OUT_OF_BOUNDS
            var base = (i * OBS_H + j) * OBS_CHANNELS
            for k in range(OBS_CHANNELS):
                obs[unsafe_offset = base + k] = Float32(1.0) if k == b else Float32(
                    0.0
                )

    for kind in range(4):
        for m in range(mob_count(kind)):
            if not env.mob_mask(kind, m):
                continue
            var lx = env.mob_x(kind, m) - px + hw
            var ly = env.mob_y(kind, m) - py + hh
            if lx >= 0 and lx < OBS_W and ly >= 0 and ly < OBS_H:
                obs[
                    unsafe_offset = (lx * OBS_H + ly) * OBS_CHANNELS
                    + NUM_BLOCKS
                    + kind
                ] = 1.0

    var o = OBS_MAP_SIZE
    for k in range(NUM_INV):
        obs[unsafe_offset = o + k] = _f16_tenth(env.inv(k))
    o += NUM_INV
    obs[unsafe_offset = o + 0] = _f16_tenth(env.gi(I_HEALTH))
    obs[unsafe_offset = o + 1] = _f16_tenth(env.gi(I_FOOD))
    obs[unsafe_offset = o + 2] = _f16_tenth(env.gi(I_DRINK))
    obs[unsafe_offset = o + 3] = _f16_tenth(env.gi(I_ENERGY))
    o += 4
    var d = env.gi(I_DIR) - 1
    for k in range(4):
        obs[unsafe_offset = o + k] = Float32(1.0) if k == d else Float32(0.0)
    o += 4
    obs[unsafe_offset = o] = env.gf(F_LIGHT)
    obs[unsafe_offset = o + 1] = Float32(1.0) if env.gb(I_SLEEPING) else Float32(
        0.0
    )

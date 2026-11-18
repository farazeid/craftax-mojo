"""Reference side of the parity harness.

Records every random draw the JAX Craftax-Classic implementation makes, so the
Mojo port can replay identical randomness, and converts EnvState to/from the
flat layout the Mojo port uses.

Draws are captured by wrapping ``jax.random.uniform`` / ``jax.random.choice``
with an ordered ``jax.debug.callback``; ordered effects preserve program order
under jit, including inside ``lax.scan``. ``choice`` is recorded as the index
it picked (identical to sampling over ``arange(n)`` with the same key/p).
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))

import craftax_mojo_ext as ext  # noqa: E402
from craftax.craftax_classic.envs.craftax_symbolic_env import (  # noqa: E402
    CraftaxClassicSymbolicEnvNoAutoReset,
)
from craftax.craftax_classic.renderer import render_craftax_symbolic  # noqa: E402

L = ext.layout()

_tape: list[np.ndarray] = []
_orig_uniform = jax.random.uniform
_orig_choice = jax.random.choice


def _record(x):
    _tape.append(np.asarray(x, dtype=np.float32).reshape(-1))


def _uniform(key, shape=(), dtype=float, minval=0.0, maxval=1.0):
    out = _orig_uniform(key, shape, dtype, minval, maxval)
    jax.debug.callback(_record, out, ordered=True)
    return out


def _choice(key, a, shape=(), replace=True, p=None, axis=0):
    a = jnp.asarray(a)
    n = a.shape[axis]
    idx = _orig_choice(key, jnp.arange(n), shape, replace, p, axis)
    jax.debug.callback(_record, idx.astype(jnp.float32), ordered=True)
    return jnp.take(a, idx, axis=axis)


@contextlib.contextmanager
def recording():
    jax.random.uniform = _uniform
    jax.random.choice = _choice
    try:
        yield
    finally:
        jax.random.uniform = _orig_uniform
        jax.random.choice = _orig_choice


def take_tape() -> np.ndarray:
    jax.effects_barrier()
    out = np.concatenate(_tape) if _tape else np.zeros(0, np.float32)
    _tape.clear()
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# Environment wrappers (jitted, with recording installed at trace time)

env = CraftaxClassicSymbolicEnvNoAutoReset()
params = env.default_params

with recording():
    _reset = jax.jit(lambda k: env.reset_env(k, params)[1])
    _step = jax.jit(lambda k, s, a: env.step_env(k, s, a, params)[1:4])


def reset(key):
    with recording():
        s = _reset(key)
    return s, take_tape()


def step(key, state, action):
    with recording():
        s, r, d = _step(key, state, action)
    return s, float(r), bool(d), take_tape()


obs_fn = jax.jit(render_craftax_symbolic)


# --------------------------------------------------------------------------
# EnvState <-> flat layout

def _mobs(m):
    return [
        np.asarray(m.position, np.int32).reshape(-1),
        np.asarray(m.health, np.int32),
        np.asarray(m.mask, np.int32),
        np.asarray(m.attack_cooldown, np.int32),
    ]


def to_flat(s):
    inv = s.inventory
    ints = np.concatenate(
        [
            np.asarray(s.player_position, np.int32),
            np.array(
                [
                    s.player_direction,
                    s.player_health,
                    s.player_food,
                    s.player_drink,
                    s.player_energy,
                    s.is_sleeping,
                    s.timestep,
                ],
                np.int32,
            ),
            np.array(
                [
                    inv.wood, inv.stone, inv.coal, inv.iron, inv.diamond,
                    inv.sapling, inv.wood_pickaxe, inv.stone_pickaxe,
                    inv.iron_pickaxe, inv.wood_sword, inv.stone_sword,
                    inv.iron_sword,
                ],
                np.int32,
            ),
            *_mobs(s.zombies),
            *_mobs(s.cows),
            *_mobs(s.skeletons),
            *_mobs(s.arrows),
            np.asarray(s.arrow_directions, np.int32).reshape(-1),
            np.asarray(s.growing_plants_positions, np.int32).reshape(-1),
            np.asarray(s.growing_plants_age, np.int32),
            np.asarray(s.growing_plants_mask, np.int32),
            np.asarray(s.achievements, np.int32),
        ]
    )
    assert ints.shape == (L["NUM_I"],), ints.shape
    floats = np.array(
        [
            s.player_recover,
            s.player_hunger,
            s.player_thirst,
            s.player_fatigue,
            s.light_level,
        ],
        np.float32,
    )
    return {
        "ints": np.ascontiguousarray(ints),
        "floats": floats,
        "map": np.ascontiguousarray(np.asarray(s.map, np.uint8).reshape(-1)),
        "mob_map": np.ascontiguousarray(np.asarray(s.mob_map, np.uint8).reshape(-1)),
    }


def empty_flat():
    return {
        "ints": np.zeros(L["NUM_I"], np.int32),
        "floats": np.zeros(L["NUM_F"], np.float32),
        "map": np.zeros(L["MAP_CELLS"], np.uint8),
        "mob_map": np.zeros(L["MAP_CELLS"], np.uint8),
    }


def addrs(flat):
    return tuple(flat[k].ctypes.data for k in ("ints", "floats", "map", "mob_map"))


# Field names for readable diffs.
def int_field_names():
    names = ["px", "py", "dir", "health", "food", "drink", "energy",
             "sleeping", "timestep"]
    names += [f"inv[{k}]" for k in (
        "wood", "stone", "coal", "iron", "diamond", "sapling", "wood_pickaxe",
        "stone_pickaxe", "iron_pickaxe", "wood_sword", "stone_sword",
        "iron_sword")]
    for kind, c in (("zombie", 3), ("cow", 3), ("skeleton", 2), ("arrow", 3)):
        names += [f"{kind}.pos[{i // 2}][{i % 2}]" for i in range(2 * c)]
        for f in ("health", "mask", "cooldown"):
            names += [f"{kind}.{f}[{i}]" for i in range(c)]
    names += [f"arrow_dir[{i // 2}][{i % 2}]" for i in range(6)]
    names += [f"plant.pos[{i // 2}][{i % 2}]" for i in range(20)]
    names += [f"plant.age[{i}]" for i in range(10)]
    names += [f"plant.mask[{i}]" for i in range(10)]
    names += [f"ach[{i}]" for i in range(22)]
    assert len(names) == L["NUM_I"]
    return names


FLOAT_NAMES = ["recover", "hunger", "thirst", "fatigue", "light"]
INT_NAMES = int_field_names()


def diff(a, b, atol=1e-5):
    """Human-readable list of mismatches between two flat states."""
    out = []
    for i in np.nonzero(a["ints"] != b["ints"])[0]:
        out.append(f"{INT_NAMES[i]}: mojo={a['ints'][i]} jax={b['ints'][i]}")
    for i in np.nonzero(np.abs(a["floats"] - b["floats"]) > atol)[0]:
        out.append(f"{FLOAT_NAMES[i]}: mojo={a['floats'][i]} jax={b['floats'][i]}")
    for k in ("map", "mob_map"):
        bad = np.nonzero(a[k] != b[k])[0]
        if len(bad):
            c = bad[0]
            out.append(
                f"{k}: {len(bad)} cells differ, first ({c // 64},{c % 64}) "
                f"mojo={a[k][c]} jax={b[k][c]}"
            )
    return out

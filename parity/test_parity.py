"""Transition-level parity between JAX Craftax-Classic and the Mojo port.

For each transition, the JAX state s_t is loaded into Mojo, both
implementations step with the same action and the same random draws (the JAX
draws are recorded and replayed), and the resulting states, rewards, done
flags and observations must match field by field.

Episodes are a mix of plain random play and "boosted" episodes whose start
state is perturbed (inventory, tools, intrinsics, timestep, sleeping) so that
crafting, mining, placing, combat, night spawns, sleeping and the timestep
limit are all exercised. Run with:  uv run pytest parity -q
"""

from __future__ import annotations

import collections

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_ref as R

N_EPISODES = int(__import__("os").environ.get("PARITY_EPISODES", 90))
N_STEPS = int(__import__("os").environ.get("PARITY_STEPS", 300))


def boost(state, rng: np.random.Generator):
    inv = state.inventory.replace(
        **{k: int(rng.integers(0, 10)) for k in (
            "wood", "stone", "coal", "iron", "diamond", "sapling")},
        **{k: int(rng.integers(0, 2)) for k in (
            "wood_pickaxe", "stone_pickaxe", "iron_pickaxe", "wood_sword",
            "stone_sword", "iron_sword")},
    )
    return state.replace(
        inventory=inv,
        player_energy=int(rng.integers(0, 10)),
        player_food=int(rng.integers(0, 10)),
        player_drink=int(rng.integers(0, 10)),
        player_health=int(rng.integers(1, 10)),
        player_fatigue=float(rng.uniform(-12, 32)),
        player_hunger=float(rng.uniform(0, 26)),
        player_thirst=float(rng.uniform(0, 21)),
        player_recover=float(rng.uniform(-16, 26)),
        timestep=int(rng.choice([0, 150, 9900, int(rng.integers(0, 10000))])),
    )


NEAR_BLOCKS = [2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def stage(state, rng: np.random.Generator):
    """Scenario start: random blocks, mobs and ripening plants around the player.

    Exercises mining every ore, eating plants, crafting at tables/furnaces,
    lava, and fighting cows, zombies and skeletons at close range.
    """
    state = boost(state, rng)
    px, py = (int(v) for v in state.player_position)
    m = np.asarray(state.map).copy()
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            if (dx or dy) and rng.random() < 0.6:
                m[px + dx, py + dy] = rng.choice(NEAR_BLOCKS, p=None)
    m[px, py] = 2
    mob_map = np.zeros_like(np.asarray(state.mob_map))
    ring = [(dx, dy) for dx in range(-5, 6) for dy in range(-5, 6)
            if 0 < abs(dx) + abs(dy) <= 5]
    rng.shuffle(ring)
    spots = iter(ring)

    def place(mobs, n, health):
        pos = np.asarray(mobs.position).copy()
        hp = np.asarray(mobs.health).copy()
        mask = np.zeros(n, bool)
        for i in range(n):
            if rng.random() < 0.7:
                dx, dy = next(spots)
                pos[i] = (px + dx, py + dy)
                hp[i] = rng.integers(1, health + 1)
                mask[i] = True
                m[px + dx, py + dy] = 2
                mob_map[px + dx, py + dy] = True
        return mobs.replace(position=jnp.asarray(pos), health=jnp.asarray(hp),
                            mask=jnp.asarray(mask),
                            attack_cooldown=jnp.asarray(
                                rng.integers(0, 3, n), jnp.int32))

    zombies = place(state.zombies, 3, 5)
    cows = place(state.cows, 3, 3)
    skeletons = place(state.skeletons, 2, 3)

    plants_pos = np.asarray(state.growing_plants_positions).copy()
    plants_age = np.asarray(state.growing_plants_age).copy()
    plants_mask = np.zeros(10, bool)
    for i in range(int(rng.integers(0, 4))):
        dx, dy = next(spots)
        plants_pos[i] = (px + dx, py + dy)
        plants_age[i] = rng.integers(590, 610)
        plants_mask[i] = True
        m[px + dx, py + dy] = 15 if plants_age[i] < 600 else 16

    return state.replace(
        map=jnp.asarray(m), mob_map=jnp.asarray(mob_map), zombies=zombies,
        cows=cows, skeletons=skeletons,
        growing_plants_positions=jnp.asarray(plants_pos),
        growing_plants_age=jnp.asarray(plants_age),
        growing_plants_mask=jnp.asarray(plants_mask),
        is_sleeping=bool(rng.random() < 0.2),
        player_direction=int(rng.integers(1, 5)),
    )


# Action mix: uniform, plus a bias towards DO / crafting / placing so those
# code paths fire often enough to be tested.
ACTION_P = np.ones(17)
ACTION_P[5] += 6
ACTION_P[7:17] += 1
ACTION_P /= ACTION_P.sum()


def run_episode(ep: int, coverage: collections.Counter):
    rng = np.random.default_rng(ep)
    key = jax.random.PRNGKey(ep)
    key, k = jax.random.split(key)
    state, tape = R.reset(k)

    m = R.empty_flat()
    used = R.ext.reset_tape(R.addrs(m), tape.ctypes.data)
    assert used == len(tape)
    d = R.diff(m, R.to_flat(state))
    assert not d, f"ep {ep} reset: {d}"

    if ep % 3 == 1:
        state = boost(state, rng)
    elif ep % 3 == 2:
        state = stage(state, rng)

    for t in range(N_STEPS):
        action = int(rng.choice(17, p=ACTION_P))
        key, k = jax.random.split(key)
        nxt, r, done, tape = R.step(k, state, action)

        m = R.to_flat(state)
        r_m, done_m, used = R.ext.step_tape(R.addrs(m), action, tape.ctypes.data)
        ctx = f"ep {ep} t {t} action {action}"
        assert used == len(tape), f"{ctx}: draws used {used} != {len(tape)}"
        ref = R.to_flat(nxt)
        d = R.diff(m, ref)
        assert not d, f"{ctx}: {d}"
        assert np.float32(r_m) == np.float32(r), f"{ctx}: reward {r_m} vs {r}"
        assert done_m == done, f"{ctx}: done {done_m} vs {done}"

        obs_ref = np.asarray(R.obs_fn(nxt), np.float32)
        obs_m = np.zeros(R.L["OBS_SIZE"], np.float32)
        R.ext.write_obs(R.addrs(m), obs_m.ctypes.data)
        bad = np.nonzero(np.abs(obs_m - obs_ref) > 1e-6)[0]
        assert not len(bad), f"{ctx}: obs differs at {bad[:10]}"

        prev = R.to_flat(state)["ints"]
        coverage["steps"] += 1
        for i, name in enumerate(R.INT_NAMES):
            if name.startswith("ach[") and ref["ints"][i] > prev[i]:
                coverage[name] += 1
        coverage["zombie_spawn"] += int(
            ref["ints"][R.L["I_ZOMBIE"] + 9 : R.L["I_ZOMBIE"] + 12].sum()
            > prev[R.L["I_ZOMBIE"] + 9 : R.L["I_ZOMBIE"] + 12].sum())
        coverage["arrow_fired"] += int(
            ref["ints"][R.L["I_ARROW"] + 9 : R.L["I_ARROW"] + 12].sum()
            > prev[R.L["I_ARROW"] + 9 : R.L["I_ARROW"] + 12].sum())
        coverage["damage_taken"] += int(ref["ints"][3] < prev[3])
        coverage["done"] += int(done)

        if done:
            key, k = jax.random.split(key)
            state, tape = R.reset(k)
        else:
            state = nxt


COVERAGE: collections.Counter = collections.Counter()


@pytest.mark.parametrize("ep", range(N_EPISODES))
def test_episode(ep):
    run_episode(ep, COVERAGE)


def test_report_coverage():
    print("\ncoverage:", dict(sorted(COVERAGE.items())))

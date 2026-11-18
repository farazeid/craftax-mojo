"""Distributional parity: Mojo with its own RNG vs JAX with its own RNG.

The tape tests prove the transition logic is identical given identical
randomness; this checks the native generator, rejection-sampled spawns and
the batched auto-reset driver end to end, by comparing statistics:

* world generation: per-world counts of every block type (512 worlds each);
* random-policy rollouts (512 envs x 400 steps): termination rate, mean
  reward and per-achievement unlock rates.

Differences are tested with a two-sample z-score (|z| < 4.5).
  uv run pytest parity/test_distribution.py -s
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np

import jax_ref as R
from craftax.craftax_classic.envs.craftax_symbolic_env import (
    CraftaxClassicSymbolicEnvNoAutoReset,
)

N = 512
T = 400
Z_MAX = 4.5
L = R.L


class MojoBatch:
    def __init__(self, n, seed):
        self.n = n
        self.ints = np.zeros((n, L["NUM_I"]), np.int32)
        self.floats = np.zeros((n, L["NUM_F"]), np.float32)
        self.map = np.zeros((n, L["MAP_CELLS"]), np.uint8)
        self.mob = np.zeros((n, L["MAP_CELLS"]), np.uint8)
        self.rng = np.zeros((n, 2), np.uint64)
        self.obs = np.zeros((n, L["OBS_SIZE"]), np.float32)
        self.reward = np.zeros(n, np.float32)
        self.done = np.zeros(n, np.uint8)
        self.addr = tuple(a.ctypes.data for a in (
            self.ints, self.floats, self.map, self.mob, self.rng, self.obs,
            self.reward, self.done)) + (n,)
        R.ext.reset_cpu(self.addr, seed, os.cpu_count())

    def step(self):
        R.ext.step_random_cpu(self.addr, os.cpu_count())


def zscore(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    if se == 0:
        return 0.0 if a.mean() == b.mean() else np.inf
    return (a.mean() - b.mean()) / se


def report(name, z, a, b):
    flag = "" if abs(z) < Z_MAX else "  <-- FAIL"
    print(f"{name:28s} jax={np.mean(a):9.4f} mojo={np.mean(b):9.4f} z={z:+.2f}{flag}")


def test_world_statistics():
    keys = jax.random.split(jax.random.PRNGKey(123), N)
    env = CraftaxClassicSymbolicEnvNoAutoReset()
    states = jax.jit(jax.vmap(lambda k: env.reset_env(k, env.default_params)[1]))(keys)
    jmaps = np.asarray(states.map).reshape(N, -1)
    m = MojoBatch(N, seed=7)
    bad = []
    print()
    for b in range(2, 17):
        cj = (jmaps == b).sum(1)
        cm = (m.map == b).sum(1)
        if cj.sum() == 0 and cm.sum() == 0:
            continue
        z = zscore(cj, cm)
        report(f"block {b} count", z, cj, cm)
        if abs(z) >= Z_MAX:
            bad.append(b)
    assert not bad, f"block count distributions differ: {bad}"


def jax_rollout():
    env = CraftaxClassicSymbolicEnvNoAutoReset()
    p = env.default_params

    def step(carry, _):
        key, s = carry
        key, ka, ks, kr = jax.random.split(key, 4)
        a = jax.random.randint(ka, (N,), 0, 17)
        _, s2, r, d, _ = jax.vmap(env.step_env, (0, 0, 0, None))(
            jax.random.split(ks, N), s, a, p)
        unlocked = s2.achievements & ~s.achievements
        _, sr = jax.vmap(env.reset_env, (0, None))(jax.random.split(kr, N), p)
        s3 = jax.tree.map(
            lambda x, y: jnp.where(d.reshape((-1,) + (1,) * (x.ndim - 1)), x, y),
            sr, s2)
        return (key, s3), (r, d, unlocked)

    @jax.jit
    def run(key):
        _, s = jax.vmap(env.reset_env, (0, None))(jax.random.split(key, N), p)
        _, out = jax.lax.scan(step, (key, s), None, length=T)
        return out

    r, d, u = run(jax.random.PRNGKey(5))
    return np.asarray(r), np.asarray(d), np.asarray(u)


def mojo_rollout():
    m = MojoBatch(N, seed=11)
    a0 = L["I_ACH"]
    rs, ds, us = [], [], []
    for _ in range(T):
        before = m.ints[:, a0:a0 + 22].copy()
        m.step()
        after = m.ints[:, a0:a0 + 22]
        done = m.done.astype(bool)
        # On done the env already holds a fresh world, so terminal-step
        # unlocks aren't observable; both sides exclude them (see below).
        unlocked = (after > before) & ~done[:, None]
        rs.append(m.reward.copy())
        ds.append(done)
        us.append(unlocked)
    return np.stack(rs), np.stack(ds), np.stack(us)


def test_rollout_statistics():
    rj, dj, uj = jax_rollout()
    rm, dm, um = mojo_rollout()
    # Per-env aggregates over the rollout are the independent samples.
    print()
    checks = {
        "terminations per env": (dj.sum(0), dm.sum(0)),
        "reward per env": (rj.sum(0), rm.sum(0)),
    }
    # Unlocks on terminal steps aren't visible in Mojo's post-reset state;
    # compare non-terminal unlocks on both sides.
    uj = uj & ~dj[:, :, None]
    for k in range(22):
        if uj[..., k].sum() + um[..., k].sum() >= 20:
            checks[f"ach[{k}] unlocks per env"] = (uj[..., k].sum(0), um[..., k].sum(0))
    bad = []
    for name, (a, b) in checks.items():
        z = zscore(a, b)
        report(name, z, a, b)
        if abs(z) >= Z_MAX:
            bad.append(name)
    assert not bad, f"rollout statistics differ: {bad}"

"""Throughput benchmark for the JAX reference: random-policy, auto-reset steps
per second (observations included) for a sweep of batch sizes.

Two reset strategies:
  autoreset   Craftax's own CraftaxClassicSymbolicEnv: every step also
              generates a fresh world for every env and selects it on done
              (what `env.step` does out of the box).
  optimistic  craftax_baselines' OptimisticResetVecEnvWrapper: per step only
              `reset_ratio`-many worlds are generated and handed to the envs
              that finished. This is what Craftax's PPO baselines use and the
              faster, fairer baseline.

The whole rollout (action sampling, step, reset) is one jitted lax.scan, as
in the Craftax speed tests.

  uv run python bench/bench_jax.py --steps 200 --sizes 1,16,256,4096
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp

from craftax.craftax_classic.envs.craftax_symbolic_env import (
    CraftaxClassicSymbolicEnv,
    CraftaxClassicSymbolicEnvNoAutoReset,
)


def make_autoreset(n: int):
    env = CraftaxClassicSymbolicEnv()
    params = env.default_params

    def init(key):
        keys = jax.random.split(key, n)
        _, state = jax.vmap(env.reset, (0, None))(keys, params)
        return state

    def step(carry, _):
        key, state = carry
        key, ka, ks = jax.random.split(key, 3)
        actions = jax.random.randint(ka, (n,), 0, 17)
        obs, state, reward, done, _ = jax.vmap(env.step, (0, 0, 0, None))(
            jax.random.split(ks, n), state, actions, params
        )
        return (key, state), (obs.sum(), reward.sum(), done.sum())

    return init, step


def make_optimistic(n: int, reset_ratio: int):
    """Port of craftax_baselines.wrappers.OptimisticResetVecEnvWrapper."""
    env = CraftaxClassicSymbolicEnvNoAutoReset()
    params = env.default_params
    ratio = min(reset_ratio, n)
    assert n % ratio == 0

    def init(key):
        keys = jax.random.split(key, n)
        _, state = jax.vmap(env.reset, (0, None))(keys, params)
        return state

    def step(carry, _):
        key, state = carry
        key, ka, ks, kr, kc = jax.random.split(key, 5)
        actions = jax.random.randint(ka, (n,), 0, 17)
        obs_st, state_st, reward, done, _ = jax.vmap(
            env.step, (0, 0, 0, None)
        )(jax.random.split(ks, n), state, actions, params)
        obs_re, state_re = jax.vmap(env.reset, (0, None))(
            jax.random.split(kr, n // ratio), params
        )
        # Each finished env picks one of the n/ratio fresh worlds.
        idx = jax.random.randint(kc, (n,), 0, n // ratio)
        pick = lambda re, st: jax.lax.select(  # noqa: E731
            jnp.broadcast_to(done.reshape((-1,) + (1,) * (st.ndim - 1)), st.shape),
            re[idx],
            st,
        )
        state = jax.tree.map(pick, state_re, state_st)
        obs = pick(obs_re, obs_st)
        return (key, state), (obs.sum(), reward.sum(), done.sum())

    return init, step


def bench(n: int, steps: int, mode: str, reset_ratio: int, repeats: int):
    if mode == "autoreset":
        init, step = make_autoreset(n)
    else:
        init, step = make_optimistic(n, reset_ratio)

    @jax.jit
    def rollout(key, state):
        (key, state), out = jax.lax.scan(step, (key, state), None, length=steps)
        return key, state, out

    key = jax.random.PRNGKey(0)
    state = jax.jit(init)(key)
    t0 = time.perf_counter()
    key, state, out = rollout(key, state)  # compile + first run
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        key, state, out = rollout(key, state)
        jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)
    dev = jax.devices()[0]
    return {
        "impl": f"jax-{mode}",
        "device": dev.platform,
        "device_name": getattr(dev, "device_kind", str(dev)),
        "num_envs": n,
        "steps": steps,
        "seconds": best,
        "steps_per_second": n * steps / best,
        "compile_seconds": compile_s,
        "jax_version": jax.__version__,
        **({"reset_ratio": min(reset_ratio, n)} if mode == "optimistic" else {}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=0, help="0: scale with n")
    ap.add_argument("--sizes", default="1,16,256,4096")
    ap.add_argument("--mode", choices=["autoreset", "optimistic"], default="autoreset")
    ap.add_argument("--reset-ratio", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    for n in (int(s) for s in args.sizes.split(",")):
        steps = args.steps if args.steps > 0 else max(20, min(1000, 2**22 // n))
        try:
            out = bench(n, steps, args.mode, args.reset_ratio, args.repeats)
        except Exception as e:  # typically device OOM at large n
            out = {"impl": f"jax-{args.mode}", "num_envs": n, "error": str(e)[:200]}
        print(json.dumps(out), flush=True)
        if "error" in out:
            break


if __name__ == "__main__":
    main()

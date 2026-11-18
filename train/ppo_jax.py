"""End-to-end jitted PPO on Craftax-Classic-Symbolic (JAX reference).

Algorithm and hyperparameters follow baxtub's ppo.py / ppo.crafter.yaml
(github.com/farazeid/baxtub): separate actor and critic MLPs (3 x 512, tanh,
orthogonal init), optimistic resets (ratio 16), 512 envs x 64 steps per
batch, 4 epochs x 8 minibatches, Adam(2e-4, eps 1e-5) with linear annealing
and global-norm clipping at 1.0, clip 0.2, vf 0.5, ent 0.01, advantage
normalisation over the batch. Two deliberate differences:

* GAE masks with the transition's own done flag and bootstraps the last step
  from V(next obs), as in PureJaxRL / Craftax's baselines. baxtub's version
  masks with the *next* transition's done flag and bootstraps from the last
  stored value, which leaks value across episode boundaries.
* Written functionally (params pytree + optax) instead of flax.nnx; the
  computation is the same.

Training runs as a single jitted lax.scan over batches; per-batch metrics
are returned from the scan rather than logged through host callbacks, so
nothing leaves the device until training ends.

  uv run python train/ppo_jax.py --total-steps 5e6 --out results/train/jax.json
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from craftax.craftax_classic.envs.craftax_symbolic_env import (
    CraftaxClassicSymbolicEnvNoAutoReset,
)


@dataclass(frozen=True)
class Config:
    num_envs: int = 512
    num_steps: int = 64  # per env per batch
    total_steps: int = 1_000_000
    reset_ratio: int = 16
    layer_width: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.8
    clip_coef: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0
    num_epochs: int = 4
    num_minibatches: int = 8
    lr: float = 2e-4
    anneal_lr: bool = True
    seed: int = 42

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def num_batches(self) -> int:
        return self.total_steps // self.batch_size


# ---- Network ---------------------------------------------------------------

def init_mlp(key, sizes, last_scale):
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for i, (k, din, dout) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        scale = last_scale if i == len(sizes) - 2 else np.sqrt(2)
        w = jax.nn.initializers.orthogonal(scale)(k, (din, dout), jnp.float32)
        params.append({"w": w, "b": jnp.zeros((dout,), jnp.float32)})
    return params


def init_params(key, cfg: Config, obs_dim: int, num_actions: int):
    ka, kc = jax.random.split(key)
    w = cfg.layer_width
    return {
        "actor": init_mlp(ka, [obs_dim, w, w, w, num_actions], 0.01),
        "critic": init_mlp(kc, [obs_dim, w, w, w, 1], 1.0),
    }


def mlp(params, x):
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


def policy(params, obs):
    logits = mlp(params["actor"], obs)
    value = mlp(params["critic"], obs)[..., 0]
    return logits, value


# ---- Loss and advantages ----------------------------------------------------

def ppo_loss(params, cfg: Config, obs, action, old_logp, adv, ret):
    logits, value = policy(params, obs)
    logp_all = jax.nn.log_softmax(logits)
    logp = jnp.take_along_axis(logp_all, action[:, None], axis=1)[:, 0]
    entropy = -(jnp.exp(logp_all) * logp_all).sum(-1)
    ratio = jnp.exp(logp - old_logp)
    pg = jnp.maximum(
        -adv * ratio,
        -adv * jnp.clip(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
    vf = 0.5 * ((value - ret) ** 2).mean()
    ent = entropy.mean()
    return pg + cfg.vf_coef * vf - cfg.ent_coef * ent, (pg, vf, ent)


def gae(cfg: Config, values, rewards, dones, last_value):
    """Standard GAE over [T, N] arrays; dones mask the transition's own step."""
    def body(carry, x):
        gae_, next_value = carry
        value, reward, done = x
        delta = reward + cfg.gamma * next_value * (1 - done) - value
        gae_ = delta + cfg.gamma * cfg.gae_lambda * (1 - done) * gae_
        return (gae_, value), gae_

    _, adv = jax.lax.scan(
        body, (jnp.zeros_like(last_value), last_value),
        (values, rewards, dones), reverse=True, unroll=16)
    return adv, adv + values


def make_optimizer(cfg: Config, lr_schedule):
    return optax.chain(
        optax.clip_by_global_norm(cfg.max_grad_norm),
        optax.adam(lr_schedule if cfg.anneal_lr else cfg.lr, eps=1e-5),
    )


# ---- Environment: optimistic resets + episode logging ----------------------

class Env:
    def __init__(self, cfg: Config):
        self.env = CraftaxClassicSymbolicEnvNoAutoReset()
        self.params = self.env.default_params
        self.cfg = cfg
        self.num_resets = cfg.num_envs // cfg.reset_ratio

    def reset(self, key):
        keys = jax.random.split(key, self.cfg.num_envs)
        obs, state = jax.vmap(self.env.reset, (0, None))(keys, self.params)
        return obs, state

    def step(self, key, state, action):
        """OptimisticResetVecEnvWrapper.step (baxtub / craftax_baselines)."""
        n, nr = self.cfg.num_envs, self.num_resets
        k1, k2, k3 = jax.random.split(key, 3)
        obs_st, state_st, reward, done, info = jax.vmap(
            self.env.step, (0, 0, 0, None))(jax.random.split(k1, n), state, action, self.params)
        obs_re, state_re = jax.vmap(self.env.reset, (0, None))(
            jax.random.split(k2, nr), self.params)
        reset_idx = jnp.arange(nr).repeat(self.cfg.reset_ratio)
        being_reset = jax.random.choice(k3, jnp.arange(n), shape=(nr,), p=done, replace=False)
        reset_idx = reset_idx.at[being_reset].set(jnp.arange(nr))
        pick = lambda re, st: jnp.where(  # noqa: E731
            done.reshape((-1,) + (1,) * (st.ndim - 1)), re[reset_idx], st)
        state = jax.tree.map(pick, state_re, state_st)
        obs = pick(obs_re, obs_st)
        return obs, state, reward, done, state_st.achievements


# ---- Training --------------------------------------------------------------

def make_train(cfg: Config):
    env = Env(cfg)
    obs_dim = env.env.observation_space(env.params).shape[0]
    num_actions = env.env.action_space(env.params).n
    updates_per_batch = cfg.num_epochs * cfg.num_minibatches

    def lr_schedule(count):
        # baxtub's schedule: constant within a batch, linear decay over batches.
        frac = 1.0 - (count // updates_per_batch) / cfg.num_batches
        return cfg.lr * frac

    tx = make_optimizer(cfg, lr_schedule)

    def env_step(carry, _):
        params, obs, state, ep_ret, ep_len, key = carry
        key, ka, ks = jax.random.split(key, 3)
        logits, value = policy(params, obs)
        action = jax.random.categorical(ka, logits)
        logp = jax.nn.log_softmax(logits)[jnp.arange(cfg.num_envs), action]
        next_obs, state, reward, done, ach = env.step(ks, state, action)
        ep_ret = ep_ret + reward
        ep_len = ep_len + 1
        stats = (done, ep_ret * done, ep_len * done, ach * done[:, None])
        ep_ret = ep_ret * (1 - done)
        ep_len = ep_len * (1 - done)
        transition = (obs, action, logp, value, reward, done)
        return (params, next_obs, state, ep_ret, ep_len, key), (transition, stats)

    def update_minibatch(carry, mb):
        params, opt_state = carry
        (loss, aux), grads = jax.value_and_grad(ppo_loss, has_aux=True)(params, cfg, *mb)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), (loss, *aux)

    def update_epoch(carry, _):
        params, opt_state, data, key = carry
        key, kp = jax.random.split(key)
        perm = jax.random.permutation(kp, cfg.batch_size)
        mbs = jax.tree.map(
            lambda x: x.reshape((cfg.batch_size,) + x.shape[2:])[perm].reshape(
                (cfg.num_minibatches, -1) + x.shape[2:]), data)
        (params, opt_state), losses = jax.lax.scan(update_minibatch, (params, opt_state), mbs)
        return (params, opt_state, data, key), losses

    def batch_step(carry, _):
        params, opt_state, obs, state, ep_ret, ep_len, key = carry
        (params, obs, state, ep_ret, ep_len, key), (traj, stats) = jax.lax.scan(
            env_step, (params, obs, state, ep_ret, ep_len, key), None, cfg.num_steps)
        t_obs, t_action, t_logp, t_value, t_reward, t_done = traj
        _, last_value = policy(params, obs)
        adv, ret = gae(cfg, t_value, t_reward, t_done.astype(jnp.float32), last_value)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        key, ku = jax.random.split(key)
        (params, opt_state, _, _), losses = jax.lax.scan(
            update_epoch, (params, opt_state, (t_obs, t_action, t_logp, adv, ret), ku),
            None, cfg.num_epochs)
        done, rets, lens, achs = stats
        n_done = done.sum()
        metrics = {
            "episodes": n_done,
            "episode_return": rets.sum() / jnp.maximum(n_done, 1),
            "episode_length": lens.sum() / jnp.maximum(n_done, 1),
            "achievements": achs.sum((0, 1)) / jnp.maximum(n_done, 1),
            "loss": losses[0].mean(),
            "policy_loss": losses[1].mean(),
            "value_loss": losses[2].mean(),
            "entropy": losses[3].mean(),
        }
        return (params, opt_state, obs, state, ep_ret, ep_len, key), metrics

    def train(key):
        key, kp, ke = jax.random.split(key, 3)
        params = init_params(kp, cfg, obs_dim, num_actions)
        opt_state = tx.init(params)
        obs, state = env.reset(ke)
        zeros = jnp.zeros(cfg.num_envs)
        carry = (params, opt_state, obs, state, zeros, zeros, key)
        carry, metrics = jax.lax.scan(batch_step, carry, None, cfg.num_batches)
        return carry[0], metrics

    return train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=float, default=1e6)
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--precision", choices=["default", "highest"], default="default",
                    help="JAX matmul precision (default allows TF32 on NVIDIA)")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    jax.config.update("jax_default_matmul_precision", args.precision)

    cfg = Config(total_steps=int(args.total_steps), num_envs=args.num_envs, seed=args.seed)
    train = jax.jit(make_train(cfg))
    key = jax.random.PRNGKey(cfg.seed)

    t0 = time.perf_counter()
    compiled = train.lower(key).compile()
    compile_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    params, metrics = compiled(key)
    jax.block_until_ready(metrics)
    train_s = time.perf_counter() - t0

    steps = cfg.num_batches * cfg.batch_size
    metrics = jax.tree.map(np.asarray, metrics)
    dev = jax.devices()[0]
    out = {
        "impl": "jax",
        "device_name": getattr(dev, "device_kind", str(dev)),
        "config": asdict(cfg),
        "precision": args.precision,
        "env_steps": steps,
        "compile_seconds": compile_s,
        "train_seconds": train_s,
        "steps_per_second": steps / train_s,
        "metrics": {k: v.tolist() for k, v in metrics.items()},
    }
    last = slice(-max(1, cfg.num_batches // 10), None)
    print(json.dumps({k: out[k] for k in (
        "impl", "device_name", "env_steps", "compile_seconds", "train_seconds",
        "steps_per_second")}))
    print("final episode_return %.2f, episode_length %.1f" % (
        np.nanmean(metrics["episode_return"][last]), np.nanmean(metrics["episode_length"][last])))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()

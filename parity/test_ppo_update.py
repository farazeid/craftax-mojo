"""PPO update parity: Mojo kernels vs jax.grad + optax on identical inputs.

Builds a minibatch whose old log-probs are perturbed so that some ratios fall
outside the clip range, then compares the gradient of the PPO loss, the
parameters after one clipped-Adam step, the loss sums and a GAE pass.
Needs build/check_ppo (see tests/check_ppo.mojo).

  uv run pytest parity/test_ppo_update.py -s
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
import ppo_jax as P  # noqa: E402

ROWS = 256
T, N = 16, 32
LR = 2e-4


def flatten(params):
    """Mojo layout: actor then critic; per layer W [in, out] row-major, then b."""
    parts = []
    for net in ("actor", "critic"):
        for layer in params[net]:
            parts += [np.asarray(layer["w"]).ravel(), np.asarray(layer["b"])]
    return np.concatenate(parts).astype(np.float32)


def test_update_and_gae():
    jax.config.update("jax_default_matmul_precision", "highest")
    cfg = P.Config()
    rng = np.random.default_rng(0)
    params = P.init_params(jax.random.PRNGKey(1), cfg, 1345, 17)
    # Larger final-layer weights than at init so the policy is not uniform.
    params["actor"][-1]["w"] = params["actor"][-1]["w"] * 50

    obs = (rng.random((ROWS, 1345)) < 0.1).astype(np.float32)
    logits, _ = P.policy(params, obs)
    act = np.asarray(jax.random.categorical(jax.random.PRNGKey(2), logits)).astype(np.int32)
    logp_now = np.asarray(jax.nn.log_softmax(logits))[np.arange(ROWS), act]
    old_logp = (logp_now + rng.normal(0, 0.3, ROWS)).astype(np.float32)
    adv = rng.normal(size=ROWS).astype(np.float32)
    ret = rng.normal(size=ROWS).astype(np.float32)

    (loss, (pg, vf, ent)), grads = jax.value_and_grad(P.ppo_loss, has_aux=True)(
        params, cfg, obs, act, old_logp, adv, ret)
    tx = P.make_optimizer(cfg, LR)
    updates, _ = tx.update(grads, tx.init(params), params)
    after = jax.tree.map(lambda p, u: p + u, params, updates)

    values = rng.normal(size=(T, N)).astype(np.float32)
    rewards = (rng.random((T, N)) < 0.1).astype(np.float32)
    dones = rng.random((T, N)) < 0.05
    last_value = rng.normal(size=N).astype(np.float32)
    g_adv, g_ret = P.gae(cfg, values, rewards, dones.astype(np.float32), last_value)

    with tempfile.TemporaryDirectory() as d:
        dd = Path(d)
        flatten(params).tofile(dd / "params.bin")
        obs.tofile(dd / "obs.bin")
        act.tofile(dd / "act.bin")
        old_logp.tofile(dd / "logp.bin")
        adv.tofile(dd / "adv.bin")
        ret.tofile(dd / "ret.bin")
        values.tofile(dd / "values.bin")
        rewards.tofile(dd / "rewards.bin")
        dones.astype(np.uint8).tofile(dd / "dones.bin")
        last_value.tofile(dd / "last_value.bin")
        subprocess.run(
            [str(ROOT / "build/check_ppo"), d, str(ROWS), str(LR), str(T), str(N)],
            check=True, capture_output=True)
        m_grads = np.fromfile(dd / "grads.bin", np.float32)
        m_after = np.fromfile(dd / "params_after.bin", np.float32)
        m_stats = np.fromfile(dd / "stats.bin", np.float32)
        m_adv = np.fromfile(dd / "gae_adv.bin", np.float32)
        m_ret = np.fromfile(dd / "gae_ret.bin", np.float32)

    ratio = np.exp(logp_now - old_logp)
    clipped = np.mean((ratio < 0.8) | (ratio > 1.2))
    j_grads = flatten(grads)
    rel = np.abs(m_grads - j_grads).max() / np.abs(j_grads).max()
    upd_j = flatten(after) - flatten(params)
    upd_m = m_after - flatten(params)
    rel_upd = np.abs(upd_m - upd_j).max() / np.abs(upd_j).max()
    print(f"\nclipped fraction {clipped:.2f}; global grad norm {np.linalg.norm(j_grads):.3f}")
    print(f"grad   max rel err {rel:.2e}")
    print(f"update max rel err {rel_upd:.2e}")
    print(f"losses jax ({pg:.6f}, {vf:.6f}, {ent:.6f}) "
          f"mojo ({m_stats[0] / ROWS:.6f}, {m_stats[1] / ROWS:.6f}, {m_stats[2] / ROWS:.6f})")
    print(f"gae    max abs err adv {np.abs(m_adv - np.asarray(g_adv).ravel()).max():.2e} "
          f"ret {np.abs(m_ret - np.asarray(g_ret).ravel()).max():.2e}")

    assert 0.05 < clipped < 0.95  # the clipped branch is exercised
    assert rel < 1e-4
    assert rel_upd < 1e-3
    np.testing.assert_allclose(m_stats / ROWS, [pg, vf, ent], rtol=1e-4)
    np.testing.assert_allclose(m_adv, np.asarray(g_adv).ravel(), atol=1e-5)
    np.testing.assert_allclose(m_ret, np.asarray(g_ret).ravel(), atol=1e-5)

"""Summarise end-to-end PPO runs (results/train/*.json) into
results/train_summary.md and results/train_summary.json.

  uv run python results/summarize_train.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ACH = [
    "collect_wood", "place_table", "eat_cow", "collect_sapling", "collect_drink",
    "make_wood_pickaxe", "make_wood_sword", "place_plant", "defeat_zombie",
    "collect_stone", "place_stone", "eat_plant", "defeat_skeleton",
    "make_stone_pickaxe", "make_stone_sword", "wake_up", "place_furnace",
    "collect_coal", "collect_iron", "collect_diamond", "make_iron_pickaxe",
    "make_iron_sword",
]


def load():
    runs = []
    for f in sorted((HERE / "train").glob("*.json")):
        m = re.match(r"(?P<machine>[^-]+)-(?P<impl>mojo|jax)-(?P<variant>[^-]+)-(?P<run>.+)", f.stem)
        if not m:
            continue
        d = json.loads(f.read_text())
        d.update(m.groupdict())
        if d["run"] == "fp32":  # full-precision matmuls
            d["variant"] += "-fp32"
        runs.append(d)
    return runs


def tail_mean(x, frac=0.1):
    x = np.asarray(x, float)
    k = max(1, int(len(x) * frac))
    return float(np.nanmean(x[-k:]))


def score(ach_rates):
    """Crafter score: geometric mean of per-achievement success rates (%)."""
    r = np.asarray(ach_rates, float) * 100
    return float(np.exp(np.mean(np.log(1 + r))) - 1)


def main():
    runs = load()
    groups = defaultdict(list)
    for r in runs:
        groups[(r["machine"], r["variant"], r["impl"])].append(r)

    out = ["# End-to-end PPO: Craftax-Classic-Symbolic", "",
           "Steps/s is wall time for the whole run after compilation (JAX) / "
           "start-up (Mojo). Return, length and score are means over the last "
           "10% of batches; score is the Crafter geometric mean of "
           "achievement success rates. ± is the std over seeds.", "",
           "| machine | config | impl | runs | env steps | steps/s | episode return | episode length | score |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    summary = {}
    for (machine, variant, impl), rs in sorted(groups.items()):
        sps = np.array([r["steps_per_second"] for r in rs])
        ret = np.array([tail_mean(r["metrics"]["episode_return"]) for r in rs])
        length = np.array([tail_mean(r["metrics"]["episode_length"]) for r in rs])
        sc = np.array([score(np.nanmean(np.asarray(r["metrics"]["achievements"])[
            -max(1, len(r["metrics"]["achievements"]) // 10):], 0)) for r in rs])

        def pm(x, f):
            return f.format(x.mean()) + (" ± " + f.format(x.std()) if len(x) > 1 else "")

        out.append(f"| {machine} | {variant} | {impl} | {len(rs)} | {rs[0]['env_steps']:,} | "
                   f"{pm(sps / 1e3, '{:.0f}')}k | {pm(ret, '{:.2f}')} | {pm(length, '{:.0f}')} | "
                   f"{pm(sc, '{:.2f}')} |")
        summary[f"{machine}/{variant}/{impl}"] = {
            "steps_per_second": sps.tolist(), "episode_return": ret.tolist(),
            "score": sc.tolist(),
            "curves": [{"episode_return": r["metrics"]["episode_return"],
                        "batch_size": r["config"]["num_envs"] * r["config"]["num_steps"]}
                       for r in rs],
        }
    out.append("")
    for machine, variant in sorted({(m, v) for m, v, _ in groups}):
        a = groups.get((machine, variant, "mojo"))
        b = groups.get((machine, variant, "jax"))
        if a and b:
            ratio = np.mean([r["steps_per_second"] for r in a]) / np.mean(
                [r["steps_per_second"] for r in b])
            out.append(f"- {machine} {variant}: Mojo / JAX training throughput = **{ratio:.2f}x**")
    (HERE / "train_summary.md").write_text("\n".join(out) + "\n")
    (HERE / "train_summary.json").write_text(json.dumps(summary))
    print("\n".join(out))


if __name__ == "__main__":
    main()

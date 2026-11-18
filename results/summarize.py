"""Summarise results/raw/*.jsonl into results/summary.md and results/summary.json.

  uv run python results/summarize.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent

# label -> human-readable machine name
MACHINES = {
    "m4pro": "MacBook Pro, Apple M4 Pro (14-core CPU, 20-core GPU)",
    "l40s": "flair-node-13, NVIDIA L40S (48 GB)",
}

IMPL_ORDER = [
    ("mojo", "gpu", "Mojo GPU"),
    ("mojo", "gpu-naive", "Mojo GPU (naive fused resets)"),
    ("jax-optimistic", "gpu", "JAX GPU (optimistic resets)"),
    ("jax-autoreset", "gpu", "JAX GPU (auto-reset)"),
    ("mojo", "cpu", "Mojo CPU"),
    ("jax-optimistic", "cpu", "JAX CPU (optimistic resets)"),
    ("jax-autoreset", "cpu", "JAX CPU (auto-reset)"),
]


def load():
    rows = []
    # "<machine>-<suffix>.jsonl" adds to <machine>; newer files take
    # precedence for the same (impl, device, num_envs).
    for f in sorted((HERE / "raw").glob("*.jsonl"), key=lambda p: p.stat().st_mtime):
        label = f.stem.split("-")[0]
        for line in f.read_text().splitlines():
            if not line.startswith("{"):
                continue
            r = json.loads(line)
            if "error" in r:
                r["oom"] = True
            r["machine"] = label
            if r.get("device") == "cuda":
                r["device"] = "gpu"
            rows.append(r)
    return rows


def fmt(sps):
    if sps >= 1e6:
        return f"{sps / 1e6:.2f}M"
    if sps >= 1e3:
        return f"{sps / 1e3:.1f}k"
    return f"{sps:.0f}"


def main():
    rows = load()
    table = defaultdict(dict)  # (machine, impl, device) -> n -> sps
    for r in rows:
        if r.get("oom"):
            table[(r["machine"], r["impl"], "oom")][r["num_envs"]] = None
            continue
        table[(r["machine"], r["impl"], r["device"])][r["num_envs"]] = r[
            "steps_per_second"
        ]

    out = ["# Throughput: Craftax-Classic symbolic, random policy, auto-reset",
           "",
           "Environment steps per second (observations included). Best of "
           "repeats for JAX; one timed run after warm-up for Mojo.", ""]
    summary = {}
    for machine in sorted({r["machine"] for r in rows}):
        out.append(f"## {MACHINES.get(machine, machine)}")
        out.append("")
        impls = [(i, d, name) for i, d, name in IMPL_ORDER
                 if (machine, i, d) in table]
        sizes = sorted({n for (m, i, d), v in table.items() if m == machine
                        for n in v})
        out.append("| num_envs | " + " | ".join(n for _, _, n in impls) + " |")
        out.append("|---:|" + "---:|" * len(impls))
        for n in sizes:
            cells = []
            for i, d, _ in impls:
                v = table[(machine, i, d)].get(n)
                if v is None and n in table.get((machine, i, "oom"), {}):
                    cells.append("OOM")
                else:
                    cells.append(fmt(v) if v else "–")
            out.append(f"| {n:,} | " + " | ".join(cells) + " |")
        out.append("")

        def peak(i, d):
            v = table.get((machine, i, d), {})
            return max(v.values()) if v else None

        peaks = {name: peak(i, d) for i, d, name in impls}
        summary[machine] = {k: v for k, v in peaks.items() if v}
        out.append("Peak throughput: " + ", ".join(
            f"{k} {fmt(v)}" for k, v in peaks.items() if v) + ".")
        for dev in ("GPU", "CPU"):
            mojo = peaks.get(f"Mojo {dev}")
            jax_best = max(filter(None, [v for k, v in peaks.items()
                                         if k.startswith(f"JAX {dev}")]), default=None)
            if mojo and jax_best:
                out.append(f"{dev}: peak Mojo / peak JAX = **{mojo / jax_best:.1f}x**.")
        out.append("")

    (HERE / "summary.md").write_text("\n".join(out))
    (HERE / "summary.json").write_text(json.dumps(
        {"rows": rows, "peaks": summary}, indent=1))
    print("\n".join(out))


if __name__ == "__main__":
    main()

"""Draw the README figures from results/summary.json and
results/train_summary.json into results/figures/.

  uv run --with matplotlib python results/plot.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

from summarize import IMPL_ORDER, load  # noqa: E402

HERE = Path(__file__).parent
FIG = HERE / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# One colour and marker per implementation, the same in every figure.
STYLE = {
    "Mojo GPU": ("#2a78d6", "o"),
    "JAX GPU (optimistic resets)": ("#eb6834", "s"),
    "JAX CPU (optimistic resets)": ("#eb6834", "s"),
    "Mojo CPU": ("#1baf7a", "^"),
    "JAX GPU (auto-reset)": ("#eda100", "D"),
    "JAX CPU (auto-reset)": ("#eda100", "D"),
    "mojo": ("#2a78d6", "o"),
    "jax": ("#eb6834", "s"),
}
PANELS = [("l40s", "NVIDIA L40S"), ("m4pro", "Apple M4 Pro")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
    "axes.edgecolor": GRID, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False,
})


def human(x, _=None):
    for div, suf in ((1e6, "M"), (1e3, "k")):
        if x >= div:
            return f"{x / div:.3g}{suf}"
    return f"{x:.3g}"


def throughput():
    rows = [r for r in load() if not r.get("oom")]
    table = {}
    for r in rows:  # later files override earlier ones, as in summarize.py
        table[(r["machine"], r["impl"], r["device"], r["num_envs"])] = r["steps_per_second"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (machine, title) in zip(axes, PANELS):
        for impl, device, name in IMPL_ORDER:
            if name not in STYLE:  # the naive-reset variant is in summary.md
                continue
            pts = sorted((n, v) for (m, i, d, n), v in table.items()
                         if (m, i, d) == (machine, impl, device))
            if not pts:
                continue
            n, v = map(np.array, zip(*pts))
            colour, marker = STYLE[name]
            ax.plot(n, v, color=colour, marker=marker, markersize=6, linewidth=2,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
            ax.annotate(human(v.max()), (n[v.argmax()], v.max()), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=9, color=INK)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_xlabel("parallel environments")
        ax.set_xticks([1, 16, 256, 4096, 65536, 1048576])
        ax.xaxis.set_major_formatter(human)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_major_formatter(human)
        ax.legend(loc="lower right", fontsize=9)
    axes[0].set_ylabel("environment steps / s")
    fig.suptitle("Environment throughput (random policy, auto-reset, observations included)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "throughput.png", dpi=150)


def training():
    t = json.loads((HERE / "train_summary.json").read_text())
    configs = [("512", "512 envs\n(TF32)"), ("4096", "4,096 envs\n(TF32)"),
               ("512-fp32", "512 envs\n(full FP32)")]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2),
                                   gridspec_kw={"width_ratios": [1, 1.3]})
    x = np.arange(len(configs))
    w = 0.36
    for k, impl in enumerate(["mojo", "jax"]):
        colour, _ = STYLE[impl]
        vals = [np.mean(t[f"l40s/{c}/{impl}"]["steps_per_second"]) for c, _ in configs]
        bars = ax1.bar(x + (k - 0.5) * (w + 0.02), vals, w, color=colour,
                       label="Mojo/MAX" if impl == "mojo" else "JAX")
        ax1.bar_label(bars, [human(round(v, -3)) for v in vals], padding=3,
                      fontsize=9, color=INK)
    ax1.set_xticks(x, [label for _, label in configs])
    ax1.set_ylabel("environment steps / s")
    ax1.yaxis.set_major_formatter(human)
    ax1.grid(axis="x", visible=False)
    ax1.set_title("PPO training throughput, L40S", loc="left", fontsize=11)
    ax1.legend(loc="upper right", fontsize=9)

    for impl in ["mojo", "jax"]:
        colour, _ = STYLE[impl]
        curves = t[f"l40s/512/{impl}"]["curves"]
        n = min(len(c["episode_return"]) for c in curves)
        y = np.array([np.asarray(c["episode_return"][:n], float) for c in curves])
        k = 25  # moving average over batches
        y = np.array([np.convolve(r, np.ones(k) / k, mode="valid") for r in y])
        steps = (np.arange(y.shape[1]) + k) * curves[0]["batch_size"]
        mean, lo, hi = y.mean(0), y.min(0), y.max(0)
        name = "Mojo/MAX" if impl == "mojo" else "JAX"
        ax2.fill_between(steps, lo, hi, color=colour, alpha=0.18, linewidth=0)
        ax2.plot(steps, mean, color=colour, linewidth=2, label=name)
    ax2.set_xlabel("environment steps")
    ax2.set_ylabel("episode return")
    ax2.xaxis.set_major_formatter(human)
    ax2.set_title("Learning curve, 512 envs (3 seeds, min–max band)", loc="left",
                  fontsize=11)
    ax2.legend(loc="lower right", fontsize=9)
    ax2.margins(x=0)
    fig.suptitle("End-to-end PPO on the GPU: Craftax-Classic-Symbolic",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "training.png", dpi=150)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    throughput()
    training()

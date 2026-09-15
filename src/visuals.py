from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch

STATE_BYTES = {
    "FP16 params": 2.0,
    "FP16 grads": 2.0,
    "FP32 master": 4.0,
    "Adam m": 4.0,
    "Adam v": 4.0,
}


def _save(fig, path):
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig


def plot_virtual_cluster(world_size=32, local_batch=4, path=None):
    cols = 8
    rows = math.ceil(world_size / cols)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, cols + 3.2)
    ax.set_ylim(-0.8, rows + 1.3)
    ax.axis("off")

    for rank in range(world_size):
        r = rows - 1 - rank // cols
        c = rank % cols
        box = FancyBboxPatch((c + 0.08, r + 0.10), 0.82, 0.72,
                             boxstyle="round,pad=0.02,rounding_size=0.06",
                             linewidth=1.2)
        ax.add_patch(box)
        ax.text(c + 0.49, r + 0.55, f"Rank {rank}", ha="center", va="center", fontsize=8)
        ax.text(c + 0.49, r + 0.31, f"{local_batch} samples", ha="center", va="center", fontsize=7)

    sync = FancyBboxPatch((cols + 0.55, 1.2), 2.0, 1.55,
                          boxstyle="round,pad=0.05,rounding_size=0.08",
                          linewidth=1.5)
    ax.add_patch(sync)
    ax.text(cols + 1.55, 2.22, "Collective\nsynchronization", ha="center", va="center", fontsize=11)
    ax.text(cols + 1.55, 1.52, "one global-batch\noptimizer step", ha="center", va="center", fontsize=9)
    ax.add_patch(FancyArrowPatch((cols - 0.05, rows/2), (cols + 0.50, 1.98),
                                arrowstyle="->", mutation_scale=18, linewidth=1.5))

    ax.text(0, rows + 0.95, f"32 virtual GPUs / ranks → global batch = {world_size * local_batch}",
            fontsize=15, fontweight="bold", va="center")
    ax.text(0, rows + 0.48,
            "Each rank sees different data; synchronization is what makes them one data-parallel training run.",
            fontsize=10, va="center")
    return _save(fig, path)


def plot_collective_flow(world_size=32, path=None):
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    ax.text(0.2, 5.05, "Collective identity used by Data Parallel and ZeRO", fontsize=15, fontweight="bold")
    ax.text(0.2, 4.62, "32 local gradients → reduce-scatter → owned shards → all-gather → full averaged gradient",
            fontsize=10)

    for i in range(world_size):
        x = 0.55 + (i % 8) * 0.36
        y = 3.9 - (i // 8) * 0.42
        circ = plt.Circle((x, y), 0.11, fill=True)
        ax.add_patch(circ)
    ax.text(1.8, 1.85, "32 different\nlocal gradients", ha="center", fontsize=10)

    rs = FancyBboxPatch((3.65, 2.35), 2.15, 1.15, boxstyle="round,pad=0.04", linewidth=1.5)
    ax.add_patch(rs)
    ax.text(4.72, 2.93, "Reduce-Scatter\n(average + shard)", ha="center", va="center", fontsize=10)
    ax.add_patch(FancyArrowPatch((3.15, 3.05), (3.60, 3.05), arrowstyle="->", mutation_scale=18, linewidth=1.5))

    start = 6.35
    width = 3.5 / world_size
    for i in range(world_size):
        ax.add_patch(Rectangle((start + i*width, 2.62), width*0.92, 0.62, linewidth=0.4))
    ax.text(start + 1.75, 1.85, "32 owned averaged-gradient shards", ha="center", fontsize=10)
    ax.add_patch(FancyArrowPatch((5.83, 3.05), (6.25, 3.05), arrowstyle="->", mutation_scale=18, linewidth=1.5))

    ag = FancyBboxPatch((10.1, 2.35), 1.75, 1.15, boxstyle="round,pad=0.04", linewidth=1.5)
    ax.add_patch(ag)
    ax.text(10.98, 2.93, "All-Gather", ha="center", va="center", fontsize=10)
    ax.add_patch(FancyArrowPatch((9.92, 3.05), (10.05, 3.05), arrowstyle="->", mutation_scale=18, linewidth=1.5))

    full = FancyBboxPatch((12.25, 2.35), 1.35, 1.15, boxstyle="round,pad=0.04", linewidth=1.5)
    ax.add_patch(full)
    ax.text(12.92, 2.93, "Full\naverage", ha="center", va="center", fontsize=10)
    ax.add_patch(FancyArrowPatch((11.90, 3.05), (12.20, 3.05), arrowstyle="->", mutation_scale=18, linewidth=1.5))

    ax.text(7.05, 0.78,
            "Numerical notebook check: max |all-reduce − (reduce-scatter + all-gather)| = 0.0",
            ha="center", fontsize=11, fontweight="bold")
    return _save(fig, path)


def plot_zero_storyboard(world_size=32, model_params=30_000_000_000, card_gib=74.5, path=None):
    strategies = [
        ("Data Parallel", 16.0, 2.0, 32.0, "P, G, O all replicated"),
        ("ZeRO-1", 4.0 + 12.0/world_size, 2.0, 8.75, "optimizer state sharded"),
        ("ZeRO-2", 2.0 + 14.0/world_size, 2.0, 4.875, "gradients + optimizer sharded"),
        ("ZeRO-3", 16.0/world_size, 3.0, 1.0, "params + grads + optimizer sharded"),
    ]
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    ax.text(0.2, 7.0, "ZeRO progression at 32 ranks — one-page result storyboard", fontsize=16, fontweight="bold")
    ax.text(0.2, 6.57,
            "Same optimizer mathematics, progressively less persistent redundancy, higher Stage-3 communication.",
            fontsize=10)

    x_positions = [0.35, 4.25, 8.15, 12.05]
    for x, (name, bpp, comm, redundancy, ownership) in zip(x_positions, strategies):
        gib = bpp * model_params / (1024**3)
        box = FancyBboxPatch((x, 1.0), 3.35, 4.95, boxstyle="round,pad=0.06", linewidth=1.5, fill=False)
        ax.add_patch(box)
        ax.text(x+1.675, 5.55, name, ha="center", fontsize=13, fontweight="bold")
        ax.text(x+1.675, 5.05, ownership, ha="center", fontsize=8.5, wrap=True)

        gauge_x, gauge_y, gauge_w, gauge_h = x+0.42, 3.80, 2.50, 0.55
        ax.add_patch(Rectangle((gauge_x, gauge_y), gauge_w, gauge_h, fill=False, linewidth=1.1))
        fill_w = min(gauge_w, gauge_w * gib/card_gib)
        ax.add_patch(Rectangle((gauge_x, gauge_y), fill_w, gauge_h, linewidth=0))
        if gib > card_gib:
            ax.text(x+1.675, 3.50, f"{gib:.1f} GiB/rank  ✕ > 74.5", ha="center", fontsize=10, fontweight="bold")
        else:
            ax.text(x+1.675, 3.50, f"{gib:.1f} GiB/rank  ✓ fits", ha="center", fontsize=10, fontweight="bold")

        ax.text(x+0.40, 2.80, f"{bpp:.4f} B/param/rank", fontsize=9.5)
        ax.text(x+0.40, 2.30, f"Communication ≈ {comm:.0f}P/step", fontsize=9.5)
        ax.text(x+0.40, 1.80, f"Cluster redundancy = {redundancy:g}×", fontsize=9.5)
        ax.text(x+0.40, 1.30,
                "Optimizer work/rank: " + ("100%" if name == "Data Parallel" else f"{100/world_size:.3g}%"),
                fontsize=9.5)

    ax.text(8.0, 0.40,
            "Key result: ZeRO-2 is the first stage that fits the 30B state at 32 ranks; ZeRO-3 removes persistent cluster-wide redundancy.",
            ha="center", fontsize=10.5, fontweight="bold")
    return _save(fig, path)


def plot_memory_composition(world_size=32, path=None):
    strategies = ["Data Parallel", "ZeRO-1", "ZeRO-2", "ZeRO-3"]
    state_names = list(STATE_BYTES.keys())
    fractions = {
        "Data Parallel": [1, 1, 1, 1, 1],
        "ZeRO-1": [1, 1, 1/world_size, 1/world_size, 1/world_size],
        "ZeRO-2": [1, 1/world_size, 1/world_size, 1/world_size, 1/world_size],
        "ZeRO-3": [1/world_size]*5,
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    bottoms = np.zeros(len(strategies))
    for idx, state in enumerate(state_names):
        vals = np.array([STATE_BYTES[state] * fractions[s][idx] for s in strategies])
        ax.bar(strategies, vals, bottom=bottoms, label=state)
        bottoms += vals
    for i, total in enumerate(bottoms):
        ax.text(i, total + 0.28, f"{total:.4g} B", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Persistent bytes per parameter on one rank")
    ax.set_title("Where the 16 bytes/parameter go as ZeRO removes redundancy")
    ax.legend(ncol=3)
    ax.set_ylim(0, max(bottoms)*1.14)
    fig.tight_layout()
    return _save(fig, path)


def plot_fit_matrix(model_params=30_000_000_000, card_gib=74.5, path=None):
    strategies = [("Data Parallel","dp"),("ZeRO-1","zero1"),("ZeRO-2","zero2"),("ZeRO-3","zero3")]
    world_sizes = [1,2,4,8,16,32,64,128]
    def bpp(key,n):
        if key=="dp": return 16
        if key=="zero1": return 4+12/n
        if key=="zero2": return 2+14/n
        return 16/n
    vals = np.array([[bpp(k,n)*model_params/(1024**3) for n in world_sizes] for _,k in strategies])
    fits = vals <= card_gib

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.imshow(fits.astype(float), aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(world_sizes)), world_sizes)
    ax.set_yticks(range(len(strategies)), [s[0] for s in strategies])
    ax.set_xlabel("World size")
    ax.set_title("30B state-only fit matrix on a 74.5 GiB GPU")
    for i in range(len(strategies)):
        for j in range(len(world_sizes)):
            txt = f"{'FIT' if fits[i,j] else 'NO'}\n{vals[i,j]:.1f} GiB"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_correctness_trajectory(history_df, path=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in ["Data Parallel","ZeRO-1","ZeRO-2","ZeRO-3"]:
        ax.plot(history_df["step"], history_df[name], marker="o", label=name)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean local loss before update")
    ax.set_title("Correctness proof: all sharding strategies follow the same Adam trajectory")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_communication_pressure(path=None):
    rows = [
        ("H100 · 2P", 7.10, 2.40),
        ("H100 · 3P", 7.10, 3.60),
        ("B200 · 2P", 3.12, 2.40),
        ("B200 · 3P", 3.12, 3.60),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    labels = [r[0] for r in rows]
    compute = np.array([r[1] for r in rows])
    comm = np.array([r[2] for r in rows])
    ratios = 100*comm/compute
    x = np.arange(len(labels))
    ax.bar(x-0.18, compute, width=0.36, label="Compute window (s)")
    ax.bar(x+0.18, comm, width=0.36, label="Communication time (s)")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Seconds")
    ax.set_title("Communication becomes harder to hide as compute gets faster")
    ax.legend()
    for i, ratio in enumerate(ratios):
        ax.text(i, max(compute[i], comm[i]) + 0.18, f"{ratio:.0f}% of compute", ha="center", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def plot_decision_map(card_gib=74.5, path=None):
    points = pd.DataFrame([
        {"candidate":"ZeRO-2 · 32 ranks","comm_gb":120.0,"state_gib":68.1},
        {"candidate":"ZeRO-3 · 8 ranks","comm_gb":180.0,"state_gib":55.9},
    ])
    points["headroom_gib"] = card_gib - points["state_gib"]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(points["comm_gb"], points["headroom_gib"], s=170)
    for _, r in points.iterrows():
        ax.annotate(r["candidate"], (r["comm_gb"], r["headroom_gib"]),
                    xytext=(8, 8), textcoords="offset points", fontsize=10)
    ax.set_xlabel("Approx. communication per rank per step (GB)")
    ax.set_ylabel("State headroom on 74.5 GiB GPU (GiB)")
    ax.set_title("The real choice is a trade-off, not 'highest ZeRO stage wins'")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return _save(fig, path)

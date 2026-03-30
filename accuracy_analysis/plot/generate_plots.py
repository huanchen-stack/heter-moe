import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 200,
})

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(PLOT_DIR), "data")

COLORS = {"bf16": "#4C72B0", "fp8": "#55A868", "nvfp4": "#C44E52"}
LABELS = {"bf16": "BF16 (Baseline)", "fp8": "FP8", "nvfp4": "NVFP4"}


# ── Perplexity plots ─────────────────────────────────────────────────────────

def plot_ppl(data, title, filename):
    configs = [d["quant_mode"] for d in data]
    ppls = [d["perplexity"] for d in data]
    colors = [COLORS[c] for c in configs]
    labels = [LABELS[c] for c in configs]

    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(labels, ppls, color=colors, width=0.5, edgecolor="white", linewidth=0.8)

    # value labels on bars
    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{ppl:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # y-axis: start slightly below the minimum to show differences clearly
    y_min = min(ppls) * 0.995
    y_max = max(ppls) * 1.008
    ax.set_ylim(y_min, y_max)

    ax.set_ylabel("Perplexity")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")


with open(os.path.join(DATA_DIR, "ppl", "ppl_memory_bound.json")) as f:
    ppl_mem = json.load(f)
with open(os.path.join(DATA_DIR, "ppl", "ppl_compute_bound.json")) as f:
    ppl_comp = json.load(f)

plot_ppl(ppl_mem, "Perplexity — Memory-Bound (seq 512)", "ppl_memory_bound.png")
plot_ppl(ppl_comp, "Perplexity — Compute-Bound (seq 8192)", "ppl_compute_bound.png")


# ── Downstream table (LaTeX → image) ─────────────────────────────────────────

with open(os.path.join(DATA_DIR, "downstream", "downstream_results.json")) as f:
    ds = json.load(f)

# organise by config, preserving order: bf16, fp8, nvfp4
order = ["bf16", "fp8", "nvfp4"]
by_mode = {d["quant_mode"]: d for d in ds}

# collect all tasks (union across configs) keeping display names
all_tasks = {}
for cfg in ds:
    for key, val in cfg.get("tasks", {}).items():
        all_tasks[key] = val["display_name"]

task_keys = list(all_tasks.keys())
task_names = [all_tasks[k] for k in task_keys]

# build rows: one per quantization mode
rows = []
for mode in order:
    entry = by_mode.get(mode)
    if entry is None:
        continue
    row = [LABELS[mode]]
    for tk in task_keys:
        if tk in entry["tasks"]:
            row.append(f'{entry["tasks"][tk]["primary_metric"] * 100:.2f}')
        else:
            row.append("—")
    rows.append(row)

# compute degradation row (relative to bf16 baseline)
baseline = by_mode["bf16"]
for mode in ["fp8", "nvfp4"]:
    entry = by_mode.get(mode)
    if entry is None:
        continue
    row = [f"{LABELS[mode]} $\\Delta$"]
    for tk in task_keys:
        if tk in entry["tasks"] and tk in baseline["tasks"]:
            diff = (entry["tasks"][tk]["primary_metric"] - baseline["tasks"][tk]["primary_metric"]) * 100
            sign = "+" if diff >= 0 else ""
            row.append(f"{sign}{diff:.2f}")
        else:
            row.append("—")
    rows.append(row)

# render with matplotlib table
n_cols = 1 + len(task_names)
n_rows = len(rows) + 1  # +1 for header

fig, ax = plt.subplots(figsize=(max(9, 2.2 * n_cols), 0.6 * n_rows + 1.2))
ax.axis("off")
ax.set_title("Downstream Task Accuracy — Qwen3-30B-A3B", fontsize=15, pad=16)

header = ["Precision"] + task_names
cell_text = [r for r in rows]

table = ax.table(cellText=cell_text, colLabels=header, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.8)

# auto-size columns
table.auto_set_column_width(col=list(range(n_cols)))

# style header
for j in range(n_cols):
    cell = table[0, j]
    cell.set_facecolor("#2C3E50")
    cell.set_text_props(color="white", fontweight="bold")

# style data rows
for i in range(1, n_rows):
    for j in range(n_cols):
        cell = table[i, j]
        # delta rows get light gray bg
        if "Δ" in rows[i - 1][0] or "Delta" in rows[i - 1][0]:
            cell.set_facecolor("#F0F0F0")
            cell.set_text_props(fontstyle="italic")
        # highlight baseline row
        if i == 1:
            cell.set_facecolor("#EBF5FB")

fig.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "downstream_table.png"), bbox_inches="tight")
plt.close(fig)
print("Saved downstream_table.png")

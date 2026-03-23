"""
Generate heatmaps from FP8 sensitivity analysis results.

Produces two heatmaps:
  1. Token routing distribution (how many tokens each expert processes)
  2. Quantization error per expert (L2 norm of layer output difference)

Usage:
    python heatmap.py                                    # defaults
    python heatmap.py --input sensitivity.json           # custom input
    python heatmap.py --input sensitivity.json --linear  # linear color scale
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm, LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def load_sensitivity(path: str) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load sensitivity JSON → (error_matrix, tokens_matrix, num_layers, num_experts)."""
    with open(path) as f:
        data = json.load(f)

    num_layers = len(data)
    num_experts = max(len(v) for v in data.values())

    error_matrix = np.full((num_layers, num_experts), np.nan)
    tokens_matrix = np.full((num_layers, num_experts), np.nan)

    for layer_str, experts in data.items():
        li = int(layer_str)
        for expert_str, info in experts.items():
            ei = int(expert_str)
            if ei >= num_experts:
                continue
            tokens_matrix[li, ei] = info["tokens"]
            err = info["error"]
            error_matrix[li, ei] = err[0] if isinstance(err, list) else err

    return error_matrix, tokens_matrix, num_layers, num_experts


def plot_heatmap(
    matrix: np.ndarray,
    title: str,
    cbar_label: str,
    save_path: str,
    num_layers: int,
    num_experts: int,
    use_log: bool = True,
    cmap: str = "viridis",
    symlog: bool = False,
):
    """Plot a single heatmap and save to file."""
    fig, ax = plt.subplots(figsize=(28, 14))

    plot_data = matrix.copy()
    if symlog:
        # SymLogNorm: linear near 0, log for large values — proper for integer counts
        plot_data = np.nan_to_num(plot_data, nan=0.0)
        norm = SymLogNorm(linthresh=10, vmin=0, vmax=np.max(plot_data))
    elif use_log:
        plot_data = np.where(np.isnan(plot_data), 1e-10, plot_data)
        plot_data = np.maximum(plot_data, 1e-10)
        norm = LogNorm(vmin=np.nanmin(plot_data[plot_data > 0]), vmax=np.nanmax(plot_data))
    else:
        norm = None

    im = ax.imshow(
        plot_data, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest",
        origin="lower",
    )

    ax.set_xlabel("Expert Index", fontsize=14)
    ax.set_ylabel("Layer Index", fontsize=14)
    ax.set_title(title, fontsize=18, fontweight="bold")

    xtick_step = 8 if num_experts > 32 else 4
    ax.set_xticks(range(0, num_experts, xtick_step))
    ax.set_xticklabels(range(0, num_experts, xtick_step), fontsize=7)

    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(range(num_layers), fontsize=7)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(cbar_label, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_split_heatmap(
    tokens_matrix: np.ndarray,
    error_matrix: np.ndarray,
    save_path: str,
    num_layers: int,
    num_experts: int,
):
    """Each cell split diagonally: upper-left = tokens (green), lower-right = error (red)."""
    cmap_green = LinearSegmentedColormap.from_list("w2g", ["#ffffff", "#2ca02c"])
    cmap_red = LinearSegmentedColormap.from_list("w2r", ["#ffffff", "#d62728"])

    tokens_clean = np.nan_to_num(tokens_matrix, nan=0.0)
    error_clean = np.nan_to_num(error_matrix, nan=0.0)

    token_norm = SymLogNorm(linthresh=10, vmin=0, vmax=np.max(tokens_clean))
    error_pos = error_clean[error_clean > 0]
    error_norm = LogNorm(
        vmin=error_pos.min() if len(error_pos) else 1e-10,
        vmax=error_clean.max() if error_clean.max() > 0 else 1,
    )

    upper_patches, upper_vals = [], []
    lower_patches, lower_vals = [], []

    for row in range(num_layers):
        for col in range(num_experts):
            upper_patches.append(Polygon(
                [(col, row), (col + 1, row), (col, row + 1)], closed=True,
            ))
            upper_vals.append(tokens_clean[row, col])

            lower_patches.append(Polygon(
                [(col + 1, row), (col + 1, row + 1), (col, row + 1)], closed=True,
            ))
            lower_vals.append(max(error_clean[row, col], 1e-10))

    fig, ax = plt.subplots(figsize=(28, 14))

    upper_pc = PatchCollection(upper_patches, cmap=cmap_green, norm=token_norm, edgecolors="none")
    upper_pc.set_array(np.array(upper_vals))
    ax.add_collection(upper_pc)

    lower_pc = PatchCollection(lower_patches, cmap=cmap_red, norm=error_norm, edgecolors="none")
    lower_pc.set_array(np.array(lower_vals))
    ax.add_collection(lower_pc)

    ax.set_xlim(0, num_experts)
    ax.set_ylim(0, num_layers)
    ax.set_xlabel("Expert Index", fontsize=14)
    ax.set_ylabel("Layer Index", fontsize=14)
    ax.set_title("Token Count (green) / Quantization Error (red)", fontsize=18, fontweight="bold")

    xtick_step = 8 if num_experts > 32 else 4
    ax.set_xticks(range(0, num_experts, xtick_step))
    ax.set_xticklabels(range(0, num_experts, xtick_step), fontsize=7)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(range(num_layers), fontsize=7)

    cbar_g = fig.colorbar(upper_pc, ax=ax, shrink=0.4, pad=0.02, anchor=(0.0, 1.0))
    cbar_g.set_label("Token Count", fontsize=11)
    cbar_r = fig.colorbar(lower_pc, ax=ax, shrink=0.4, pad=0.02, anchor=(0.0, 0.0))
    cbar_r.set_label("Error (L2 norm)", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Heatmaps from sensitivity JSON")
    parser.add_argument("--input", type=str, default="../calib/fp8/qwen3-30b-a3b-fp8-layer_out_norm.json", help="Input JSON path")
    parser.add_argument("--linear", action="store_true", help="Use linear scale instead of log")
    args = parser.parse_args()

    error_matrix, tokens_matrix, num_layers, num_experts = load_sensitivity(args.input)
    use_log = not args.linear

    print(f"Loaded: {num_layers} layers x {num_experts} experts")
    print(f"  Error  range: [{np.nanmin(error_matrix):.2f}, {np.nanmax(error_matrix):.2f}]")
    print(f"  Tokens range: [{np.nanmin(tokens_matrix):.0f}, {np.nanmax(tokens_matrix):.0f}]")
    print(f"  Scale: {'log' if use_log else 'linear'}")

    # ── Heatmap 1: Token routing ─────────────────────────────────────
    plot_heatmap(
        tokens_matrix,
        title="Token Routing Distribution per Expert",
        cbar_label="Token Count",
        save_path="heatmap_tokens.png",
        num_layers=num_layers,
        num_experts=num_experts,
        use_log=use_log,
        cmap="YlOrRd",
        symlog=True,
    )

    # ── Heatmap 2: Error per token (error / num_tokens) ────────────
    error_per_token = error_matrix / np.maximum(tokens_matrix, 1)
    error_per_token = np.where(tokens_matrix == 0, np.nan, error_per_token)
    plot_heatmap(
        error_per_token,
        title="Quantization Error per Token",
        cbar_label="Error / Token Count",
        save_path="heatmap_error_per_token.png",
        num_layers=num_layers,
        num_experts=num_experts,
        use_log=use_log,
        cmap="YlOrRd",
    )

    # ── Heatmap 3: Split diagonal (tokens green / error red) ────────
    plot_split_heatmap(
        tokens_matrix, error_matrix,
        save_path="heatmap_split.png",
        num_layers=num_layers,
        num_experts=num_experts,
    )

    # ── Heatmap 4: Quantization error ────────────────────────────────
    plot_heatmap(
        error_matrix,
        title="Quantization Sensitivity (L2 Error)",
        cbar_label="Error (L2 norm)",
        save_path="heatmap_error.png",
        num_layers=num_layers,
        num_experts=num_experts,
        use_log=use_log,
        cmap="YlOrRd",
    )


if __name__ == "__main__":
    main()

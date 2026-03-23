import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

with open(SCRIPT_DIR / "../calib/fp8/qwen3-30b-a3b-fp8-layerwise_perplexity.json") as f:
    data = json.load(f)

layers = sorted(data.keys(), key=lambda x: int(x))
ref_ppl = data[layers[0]]["ref_perplexity"]
ppl_increase = [data[l]["ppl_increase"] for l in layers]
ppl_actual = [data[l]["perplexity"] for l in layers]
layer_ids = [int(l) for l in layers]

colors = ["#ff9999" if v >= 0 else "#90ee90" for v in ppl_increase]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(layer_ids, ppl_increase, bottom=ref_ppl, color=colors, edgecolor="none", width=0.8)
ax.axhline(ref_ppl, color="black", linewidth=0.5)
ax.set_xlabel("Layer")
ax.set_ylabel("Perplexity")
ax.set_title("Qwen3-30B-A3B FP8 Layerwise Sensitivity")
ax.set_xticks(range(0, len(layers), 2))
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "fp8/layerwise_sensitivity.png", dpi=150)
plt.show()

import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

base_dir = Path("output/FNSPID")
versions = [
    "ver_camf",
    "ver_gen1",
    "ver_gen2",
    "ver_gen9",
    "ver_gen4"
]

models = ["MultiModal_Baseline"]

results = {model: [] for model in models}
manifest_paths = {}

for ver in versions:
    for model in models:
        model_dir = base_dir / ver / model
        mse = math.nan
        manifest_path = None

        if model_dir.exists():
            manifests = [p for p in model_dir.glob("*/manifest.json") if p.is_file()]
            if manifests:
                manifests.sort(key=lambda p: p.stat().st_mtime)
                manifest_path = manifests[-1]
                with manifest_path.open() as f:
                    payload = json.load(f)
                metrics = payload.get("best_metrics") or payload.get("best_test_metrics") or {}
                mse = metrics.get("MSE", math.nan)

        results[model].append(mse)
        if manifest_path:
            manifest_paths[(ver, model)] = manifest_path

bar_centers = list(range(len(versions)))
num_models = len(models)
bar_width = 0.6 / max(num_models, 1)

fig, ax = plt.subplots(figsize=(7, 4))
for idx, (model, y_values) in enumerate(results.items()):
    offset = (idx - (num_models - 1) / 2) * bar_width
    positions = [x + offset for x in bar_centers]
    ax.bar(positions, y_values, width=bar_width, label=model)

ax.set_xticks(bar_centers)
ax.set_xticklabels(["Original text", "Analysis Intensity Trend", "Intensity Trend", "Trend", "Overall trend"],fontsize=8)
ax.set_xlabel("dataset alias")
ax.set_ylabel("MSE (best_metrics)")
ax.set_title("FNSPID best MSE")
ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
ax.legend()
fig.tight_layout()

output_path = base_dir / "mse_ver_Authentic.png"
fig.savefig(output_path, dpi=200)

print("Bar plot saved to", output_path)
for (ver, model), path in sorted(manifest_paths.items()):
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    print(f"{ver} | {model} | {mtime:%Y-%m-%d %H:%M:%S} | {path}")

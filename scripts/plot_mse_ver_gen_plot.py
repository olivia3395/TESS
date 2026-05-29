import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

base_dir = Path("output/FNSPID")
versions=["ver_camf", "ver_gen1","ver_gen2", "ver_gen3", "ver_gen4" ,"ver_gen5" ,"ver_gen6" ,   "ver_gen7" ,"ver_gen8" ,"ver_gen9"]


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

x = list(range(1, len(versions) + 1))
plt.figure(figsize=(7, 4))
for model, y in results.items():
    plt.plot(x, y, marker="o", label=model)
plt.xticks(x, [v.replace("ver_", "") for v in versions])
plt.xlabel("dataset alias")
plt.ylabel("MSE (best_metrics)")
plt.title("FNSPID best MSE across ver_gen1-ver_gen8")
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig(base_dir / "mse_ver_gen_plot2.png", dpi=200)

print("Plot saved to", base_dir / "mse_ver_gen_plot1.png")
for (ver, model), path in sorted(manifest_paths.items()):
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    print(f"{ver} | {model} | {mtime:%Y-%m-%d %H:%M:%S} | {path}")
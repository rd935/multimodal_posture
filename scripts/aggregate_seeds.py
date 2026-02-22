import json
import glob
import numpy as np

def load_metrics(pattern):
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No files match pattern: {pattern}")

    accs, f1s = [], []
    for fp in files:
        with open(fp, "r") as f:
            d = json.load(f)

        # adjust keys if yours are named slightly differently
        accs.append(d["test"]["acc"])
        f1s.append(d["test"]["f1_macro"])

    accs = np.array(accs, dtype=float)
    f1s  = np.array(f1s, dtype=float)

    return {
        "n": len(files),
        "acc_mean": accs.mean(), "acc_std": accs.std(ddof=0),
        "f1_mean":  f1s.mean(),  "f1_std":  f1s.std(ddof=0),
        "files": files,
    }

models = {
    "RGB-only":        "results/rgb_baseline_results_seed*.json",
    "Depth-only":      "results/depth_baseline_results_seed*.json",
    "Early Fusion":    "results/fusion_early_results_seed*.json",
    "Attention Fusion":"results/fusion_attention_results_seed*.json",
    "Core Fusion":     "results/fusion_core_results_seed*.json",
}

for name, pat in models.items():
    s = load_metrics(pat)
    print(
        f"{name:16s} | "
        f"Acc {s['acc_mean']:.4f} ± {s['acc_std']:.4f} | "
        f"Macro-F1 {s['f1_mean']:.4f} ± {s['f1_std']:.4f} | "
        f"n={s['n']}"
    )
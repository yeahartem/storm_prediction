"""
Bootstrap 95% confidence intervals for AUROC and AP.

Usage:
    python bootstrap_ci.py predictions.csv [--n_boot 2000] [--seed 42]

Input CSV must have columns: prediction (probability), target (0/1).

Outputs CI values to stdout and saves bootstrap_ci_results.json.
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def bootstrap_ci(y_true, y_prob, n_boot=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aurocs, aps = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        aurocs.append(roc_auc_score(yt, yp))
        aps.append(average_precision_score(yt, yp))
    aurocs = np.array(aurocs)
    aps = np.array(aps)
    lo, hi = alpha / 2, 1 - alpha / 2
    return {
        "AUROC": {
            "point": float(roc_auc_score(y_true, y_prob)),
            "ci_lo": float(np.quantile(aurocs, lo)),
            "ci_hi": float(np.quantile(aurocs, hi)),
            "n_boot": len(aurocs),
        },
        "AP": {
            "point": float(average_precision_score(y_true, y_prob)),
            "ci_lo": float(np.quantile(aps, lo)),
            "ci_hi": float(np.quantile(aps, hi)),
            "n_boot": len(aps),
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", help="CSV with 'prediction' and 'target' columns")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    y_true = df["target"].values.astype(np.float32)
    y_prob = df["prediction"].values.astype(np.float32)

    print(f"N={len(df)}, pos_rate={y_true.mean():.4f}, n_boot={args.n_boot}")
    results = bootstrap_ci(y_true, y_prob, n_boot=args.n_boot, seed=args.seed)

    for metric, d in results.items():
        print(
            f"{metric}: {d['point']:.4f}  "
            f"95% CI [{d['ci_lo']:.4f}, {d['ci_hi']:.4f}]  "
            f"width={d['ci_hi']-d['ci_lo']:.4f}"
        )

    out_path = args.csv.replace(".csv", "_bootstrap_ci.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

"""
Bootstrap 95% CI for AUROC and AP on ALL test_predictions.csv files.

Use proper columns: 'score' (sigmoid probability) and 'binary_target'.
The old bootstrap_ci.py used 'prediction' (raw float) and 'target' (raw m/s) which is wrong.

Usage:
    python bootstrap_ci_all.py
    python bootstrap_ci_all.py --root out/ --n_boot 2000 --pattern "test_predictions.csv"

Output: prints a markdown table to stdout, also saves to bootstrap_summary.csv
"""
import argparse, glob, os, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def bootstrap_ci(y_true, y_score, n_boot=2000, alpha=0.05, seed=42, do_brier=True):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aurocs, aps, briers = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        s = yt.sum()
        if s == 0 or s == n:
            continue
        aurocs.append(roc_auc_score(yt, ys))
        aps.append(average_precision_score(yt, ys))
        if do_brier:
            briers.append(brier_score_loss(yt, ys))
    aurocs = np.array(aurocs); aps = np.array(aps)
    lo, hi = alpha / 2, 1 - alpha / 2
    out = {
        "AUROC_pt":  float(roc_auc_score(y_true, y_score)),
        "AUROC_lo":  float(np.quantile(aurocs, lo)),
        "AUROC_hi":  float(np.quantile(aurocs, hi)),
        "AP_pt":     float(average_precision_score(y_true, y_score)),
        "AP_lo":     float(np.quantile(aps, lo)),
        "AP_hi":     float(np.quantile(aps, hi)),
        "n_boot":    int(len(aurocs)),
        "n_samples": int(n),
        "pos_rate":  float(y_true.mean()),
    }
    if do_brier:
        briers = np.array(briers)
        out.update({
            "BS_pt": float(brier_score_loss(y_true, y_score)),
            "BS_lo": float(np.quantile(briers, lo)),
            "BS_hi": float(np.quantile(briers, hi)),
        })
    else:
        out.update({"BS_pt": float('nan'), "BS_lo": float('nan'), "BS_hi": float('nan')})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="out", help="directory to scan recursively")
    p.add_argument("--pattern", default="test_predictions.csv")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="bootstrap_summary.csv")
    p.add_argument("--since", default=None,
                   help="only process subdirs named >= this date prefix, e.g. 2026-05-02")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, "**", args.pattern), recursive=True))
    if args.since:
        # Filter by date prefix in path: e.g. out/2026-05-05/...
        files = [f for f in files if any(part >= args.since for part in f.split(os.sep))]
    if not files:
        print(f"No files found at {args.root}/**/{args.pattern}")
        return
    print(f"Found {len(files)} files. Bootstrapping with n_boot={args.n_boot}...\n")

    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[skip] {f}: {e}"); continue
        # Pick the right columns. Newer logs have score/binary_target.
        do_brier = True
        if "score" in df.columns and "binary_target" in df.columns:
            y_true  = df["binary_target"].values.astype(np.int32)
            y_score = df["score"].values.astype(np.float32)
        elif "prediction" in df.columns and "target" in df.columns:
            # Legacy CSV: 'prediction' is raw logit (or raw m/s for old regression runs).
            # Skip Brier (out of [0,1] range), AUROC/AP still valid (rank-only).
            print(f"  [legacy CSV] {f}: AUROC/AP only, no Brier")
            y_true  = (df["target"].values >= 15.0).astype(np.int32)
            y_score = df["prediction"].values.astype(np.float32)
            do_brier = False
        else:
            print(f"  [skip] {f}: no usable columns ({df.columns.tolist()[:5]})")
            continue

        if y_true.sum() == 0:
            print(f"  [skip] {f}: no positives in test")
            continue

        r = bootstrap_ci(y_true, y_score, n_boot=args.n_boot, seed=args.seed, do_brier=do_brier)
        # Try to infer experiment label from path
        rel = os.path.relpath(f, args.root)
        rows.append({"path": rel, **r})

        print(
            f"{rel}\n"
            f"  N={r['n_samples']:>7d}  pos_rate={r['pos_rate']:.3f}\n"
            f"  AUROC = {r['AUROC_pt']:.4f}  [{r['AUROC_lo']:.4f}, {r['AUROC_hi']:.4f}]  "
            f"width={r['AUROC_hi']-r['AUROC_lo']:.4f}\n"
            f"  AP    = {r['AP_pt']:.4f}  [{r['AP_lo']:.4f}, {r['AP_hi']:.4f}]  "
            f"width={r['AP_hi']-r['AP_lo']:.4f}\n"
            f"  Brier = {r['BS_pt']:.4f}  [{r['BS_lo']:.4f}, {r['BS_hi']:.4f}]\n"
        )

    if rows:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(args.out, index=False)
        print(f"Saved -> {args.out}")

        # Markdown table for paper
        print("\n=== Markdown table for paper ===")
        print("| Run | N | AUROC | AP | Brier |")
        print("|---|---:|---|---|---|")
        for r in rows:
            label = r["path"].replace("/test_predictions.csv", "")
            print(
                f"| `{label}` | {r['n_samples']} | "
                f"{r['AUROC_pt']:.3f} [{r['AUROC_lo']:.3f}, {r['AUROC_hi']:.3f}] | "
                f"{r['AP_pt']:.3f} [{r['AP_lo']:.3f}, {r['AP_hi']:.3f}] | "
                f"{r['BS_pt']:.3f} |"
            )


if __name__ == "__main__":
    main()

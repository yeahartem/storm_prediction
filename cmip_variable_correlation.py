"""
EDA: pairwise correlations between all CMIP6 variables + elevation
at station locations (train period).

Variables: pr, tasmax, tasmin, sfcWindmax, elevation
Outputs:   eda_output/cmip_correlation/  (PNG + CSV)

Run locally:
    python cmip_variable_correlation.py
"""
import logging
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

DATA_DIR  = "data/cmip6_world"
OUT_DIR   = "eda_output/cmip_correlation"
HALF_SIDE = 47          # padding offset — indices in train_data_idxs are into padded arrays
MAX_ROWS  = 300_000     # subsample for speed; set to None to use all ~2M

# Variable display names for plots
VAR_LABELS = {
    "pr":         "Precip (pr)",
    "tasmax":     "Max Temp (tasmax)",
    "tasmin":     "Min Temp (tasmin)",
    "sfcWindmax": "Wind (sfcWindmax)",
    "elevation":  "Elevation",
}
VARIABLES = list(VAR_LABELS.keys())

os.makedirs(OUT_DIR, exist_ok=True)


def load_center_pixels():
    """Extract center-pixel values for all variables at every station-time sample."""
    logging.info("Loading train_data_idxs ...")
    idxs = np.load(os.path.join(DATA_DIR, "train_data_idxs.npy"))
    # rows: lat_idx(pad), lon_idx(pad), time_idx, time_pos, time_pos_m, lat_pos, lon_pos, target, threshold
    lat_pad  = idxs[0].astype(int)
    lon_pad  = idxs[1].astype(int)
    time_idx = idxs[2].astype(int)
    target   = idxs[7].astype(np.float32)
    threshold = idxs[8].astype(np.float32)
    lat_raw  = lat_pad - HALF_SIDE   # unpadded CMIP6 grid index
    lon_raw  = lon_pad - HALF_SIDE

    N = len(target)
    logging.info(f"Total train samples: {N:,}")

    # Subsample for speed
    rng = np.random.default_rng(42)
    if MAX_ROWS and N > MAX_ROWS:
        sel = rng.choice(N, size=MAX_ROWS, replace=False)
        sel.sort()
    else:
        sel = np.arange(N)

    lat_s  = lat_raw[sel]
    lon_s  = lon_raw[sel]
    t_s    = time_idx[sel]
    tgt_s  = target[sel]
    thr_s  = threshold[sel]
    positive = (tgt_s >= thr_s).astype(int)

    logging.info(f"Using {len(sel):,} samples  (positive rate: {positive.mean():.3f})")

    # Load CMIP6 time-varying variables one at a time (center pixel only)
    data = {}
    for var in ["pr", "tasmax", "tasmin", "sfcWindmax"]:
        path = os.path.join(DATA_DIR, f"{var}_16.npy")
        logging.info(f"Loading {var} from {path} ...")
        arr = np.load(path)                    # float16, shape (time, lat, lon)
        data[var] = arr[t_s, lat_s, lon_s].astype(np.float32)
        del arr

    # Elevation: no time dimension, shape (lat, lon)
    elev_path = os.path.join(DATA_DIR, "elev_16.npy")
    logging.info(f"Loading elevation from {elev_path} ...")
    elev_arr = np.load(elev_path)
    if elev_arr.ndim == 3:
        elev_arr = elev_arr[0]                 # drop dummy time dim if present

    # elev_16 might be on a different grid (elev_lat/elev_lon vs lat/lon)
    # Use the closest lat/lon mapping
    try:
        data["elevation"] = elev_arr[lat_s, lon_s].astype(np.float32)
    except IndexError:
        logging.warning("Elevation index out of bounds — using lat/lon clipped indices")
        ls = np.clip(lat_s, 0, elev_arr.shape[0]-1)
        lo = np.clip(lon_s, 0, elev_arr.shape[1]-1)
        data["elevation"] = elev_arr[ls, lo].astype(np.float32)

    df = pd.DataFrame(data)
    df["positive"] = positive
    df["target_mps"] = tgt_s
    return df


def plot_correlation_heatmap(df, out_dir):
    """Pearson + Spearman correlation heatmaps side by side."""
    feat = df[VARIABLES]
    pearson  = feat.corr(method="pearson")
    spearman = feat.corr(method="spearman")

    labels = [VAR_LABELS[v] for v in VARIABLES]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, corr_mat, title in zip(axes,
                                    [pearson, spearman],
                                    ["Pearson correlation", "Spearman correlation"]):
        im = ax.imshow(corr_mat.values, vmin=-1, vmax=1, cmap="RdBu_r")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(VARIABLES)))
        ax.set_yticks(range(len(VARIABLES)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        for i in range(len(VARIABLES)):
            for j in range(len(VARIABLES)):
                val = corr_mat.values[i, j]
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color, fontweight="bold")
        ax.set_title(title, fontsize=12)

    fig.suptitle("CMIP6 variable pairwise correlations at station locations (train set)",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "01_correlation_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved {path}")

    pearson.to_csv(os.path.join(out_dir, "pearson_correlation.csv"))
    spearman.to_csv(os.path.join(out_dir, "spearman_correlation.csv"))
    return pearson


def plot_storm_vs_nostorm(df, out_dir):
    """Mean and distribution of each variable split by storm label."""
    storm    = df[df["positive"] == 1]
    no_storm = df[df["positive"] == 0]

    fig, axes = plt.subplots(1, len(VARIABLES), figsize=(3.5 * len(VARIABLES), 5), sharey=False)
    for ax, var in zip(axes, VARIABLES):
        s_vals = storm[var].values
        n_vals = no_storm[var].values
        # downsample for violin speed
        step = max(1, len(n_vals) // 20000)
        parts = ax.violinplot(
            [n_vals[::step], s_vals[::step]],
            positions=[0, 1],
            showmedians=True,
            showextrema=True,
        )
        parts["bodies"][0].set_facecolor("steelblue")
        parts["bodies"][0].set_alpha(0.7)
        parts["bodies"][1].set_facecolor("tomato")
        parts["bodies"][1].set_alpha(0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No storm", "Storm"], fontsize=9)
        ax.set_title(VAR_LABELS[var], fontsize=10)
        ax.set_ylabel("Normalized value", fontsize=8)

        # annotate means
        ax.axhline(np.median(n_vals), color="steelblue", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.axhline(np.median(s_vals), color="tomato",    linestyle="--", linewidth=0.8, alpha=0.8)

        # t-test p-value
        _, pval = stats.ttest_ind(s_vals, n_vals, equal_var=False)
        sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
        ax.set_xlabel(f"p={pval:.1e} {sig}", fontsize=8)

    fig.suptitle("Variable distributions: storm (red) vs no-storm (blue) — train set",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(out_dir, "02_storm_vs_nostorm_violin.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved {path}")

    # Summary table
    means = df.groupby("positive")[VARIABLES].agg(["mean", "median", "std"])
    means.to_csv(os.path.join(out_dir, "storm_vs_nostorm_stats.csv"))
    logging.info("Storm vs no-storm means:\n" +
                 df.groupby("positive")[VARIABLES].mean().round(3).to_string())


def plot_scatter_matrix(df, out_dir):
    """Pairwise scatter plots, coloured by storm label."""
    n = len(VARIABLES)
    subsample = df.sample(n=min(8000, len(df)), random_state=1)
    colors = ["steelblue" if p == 0 else "tomato" for p in subsample["positive"]]

    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    for i, v1 in enumerate(VARIABLES):
        for j, v2 in enumerate(VARIABLES):
            ax = axes[i][j]
            if i == j:
                # Diagonal: histogram
                ax.hist(df[v1][::10], bins=40, color="slategray", alpha=0.7, density=True)
                ax.set_title(VAR_LABELS[v1], fontsize=8, pad=2)
            else:
                ax.scatter(subsample[v2], subsample[v1],
                           c=colors, alpha=0.25, s=3, linewidths=0)
                r, p = stats.pearsonr(df[v1], df[v2])
                ax.set_title(f"r={r:.2f}", fontsize=7, color="darkred" if abs(r) > 0.5 else "black")
            if j == 0:
                ax.set_ylabel(VAR_LABELS[v1], fontsize=7)
            if i == n - 1:
                ax.set_xlabel(VAR_LABELS[v2], fontsize=7)
            ax.tick_params(labelsize=6)

    fig.suptitle("Pairwise scatter: red = storm event, blue = no storm (train set)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(out_dir, "03_scatter_matrix.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved {path}")


def plot_pointbiserial(df, out_dir):
    """Point-biserial correlation of each variable with the binary storm label.
    This directly answers: 'how well does each variable alone predict storms?'
    """
    results = []
    for var in VARIABLES:
        r, p = stats.pointbiserialr(df["positive"], df[var])
        results.append({"variable": VAR_LABELS[var], "r_pb": r, "p_value": p, "abs_r": abs(r)})
    res_df = pd.DataFrame(results).sort_values("abs_r", ascending=False)
    logging.info(f"\nPoint-biserial correlation with storm label:\n{res_df.to_string(index=False)}")
    res_df.to_csv(os.path.join(out_dir, "pointbiserial_with_storm.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["tomato" if r > 0 else "steelblue" for r in res_df["r_pb"]]
    bars = ax.barh(res_df["variable"], res_df["r_pb"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Point-biserial correlation with storm label", fontsize=10)
    ax.set_title("How strongly each variable correlates with storm occurrence\n(train set, center pixel)",
                 fontsize=10)
    for bar, val in zip(bars, res_df["r_pb"]):
        x = bar.get_width()
        ax.text(x + 0.003 * np.sign(x), bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", va="center", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "04_pointbiserial_storm.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved {path}")
    return res_df


def main():
    df = load_center_pixels()

    logging.info("Plotting correlation heatmap ...")
    plot_correlation_heatmap(df, OUT_DIR)

    logging.info("Plotting storm vs no-storm distributions ...")
    plot_storm_vs_nostorm(df, OUT_DIR)

    logging.info("Plotting scatter matrix ...")
    plot_scatter_matrix(df, OUT_DIR)

    logging.info("Plotting point-biserial correlations with storm label ...")
    plot_pointbiserial(df, OUT_DIR)

    logging.info(f"\nAll outputs saved to {OUT_DIR}/")
    logging.info("Files:")
    for f in sorted(os.listdir(OUT_DIR)):
        logging.info(f"  {f}")


if __name__ == "__main__":
    main()

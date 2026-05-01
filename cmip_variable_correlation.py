"""
EDA: correlation between CMIP6 variables at storm vs non-storm station-days.

Loads the normalized CMIP6 .npy files and the station targets,
then for each station-day extracts the center-pixel value of each variable
and computes:
  1. Pearson correlation matrix across all variables (center pixel)
  2. Mean value of each variable conditioned on storm / no-storm label
  3. Violin / distribution plots

Usage:
    python cmip_variable_correlation.py

Output: eda_output/cmip_correlation/  (PNG figures + CSV)
"""
import logging
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

DATA_DIR   = "data/cmip6_world"
PARQUET    = "data/weatherstation_data/world_stations_2000_2025_25_days_6_months_fixed.parquet"
OUT_DIR    = "eda_output/cmip_correlation"
PRECISION  = "float16"
VARIABLES  = ["pr", "tasmax", "tasmin"]   # sfcWindmax excluded (not in best model)
TRAIN_END  = "2022-12-31"
TEST_START = "2023-01-01"
THRESHOLD  = 15.0
MAX_SAMPLES = 200_000   # subsample to keep memory manageable

os.makedirs(OUT_DIR, exist_ok=True)

def load_variable(var):
    path = os.path.join(DATA_DIR, f"{var}_{PRECISION}.npy")
    logging.info(f"Loading {path} ...")
    arr = np.load(path).astype(np.float32)   # (time, lat, lon)
    return arr

def main():
    # Load time/lat/lon coords
    time_coords = np.load(os.path.join(DATA_DIR, "time.npy")).astype("datetime64[D]")
    lat_coords  = np.load(os.path.join(DATA_DIR, "lat.npy"))
    lon_coords  = np.load(os.path.join(DATA_DIR, "lon.npy"))

    # Load all variables: shape (n_vars, time, lat, lon)
    var_data = np.stack([load_variable(v) for v in VARIABLES], axis=0)

    # Load station targets
    logging.info("Loading station targets...")
    df = pd.read_parquet(PARQUET, columns=["station_id", "date", "MXWDSP", "lat", "lon"])
    df["date"] = pd.to_datetime(df["date"])

    # Use train+val period only for correlation analysis (no test leakage)
    df = df[df["date"] <= TRAIN_END].copy()
    logging.info(f"Train+val rows: {len(df):,}")

    # Per-station threshold
    def sthresh(s):
        return max(s.quantile(0.95), THRESHOLD)
    thresholds = df.groupby("station_id")["MXWDSP"].apply(sthresh)
    df = df.join(thresholds.rename("threshold"), on="station_id")
    df["positive"] = (df["MXWDSP"] >= df["threshold"]).astype(int)

    # Map station lat/lon to nearest CMIP6 pixel
    def nearest_idx(coords, val):
        return int(np.argmin(np.abs(coords - val)))

    logging.info("Mapping stations to CMIP6 grid pixels...")
    station_info = df.groupby("station_id")[["lat", "lon"]].first().reset_index()
    station_info["lat_idx"] = station_info["lat"].apply(lambda x: nearest_idx(lat_coords, x))
    station_info["lon_idx"] = station_info["lon"].apply(lambda x: nearest_idx(lon_coords, x))
    df = df.merge(station_info[["station_id", "lat_idx", "lon_idx"]], on="station_id")

    # Map dates to time indices
    logging.info("Mapping dates to time indices...")
    time_dt64 = time_coords
    date_np = df["date"].values.astype("datetime64[D]")
    time_idx_map = {t: i for i, t in enumerate(time_dt64)}
    df["time_idx"] = [time_idx_map.get(d, -1) for d in date_np]
    df = df[df["time_idx"] >= 0]

    # Subsample for speed
    if len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42)
    logging.info(f"Extracting {len(df):,} center-pixel values...")

    # Extract center-pixel CMIP6 values for each sample
    t_idx = df["time_idx"].values.astype(int)
    la_idx = df["lat_idx"].values.astype(int)
    lo_idx = df["lon_idx"].values.astype(int)

    extracted = {}
    for i, var in enumerate(VARIABLES):
        extracted[var] = var_data[i, t_idx, la_idx, lo_idx]

    feat_df = pd.DataFrame(extracted)
    feat_df["positive"] = df["positive"].values
    feat_df["lat"] = df["lat"].values

    # 1. Overall Pearson correlation matrix
    corr = feat_df[VARIABLES].corr()
    logging.info(f"\nPearson correlation matrix (center pixel, train+val):\n{corr.round(3)}")
    corr.to_csv(os.path.join(OUT_DIR, "variable_pearson_corr.csv"))

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(VARIABLES)))
    ax.set_yticks(range(len(VARIABLES)))
    ax.set_xticklabels(VARIABLES, rotation=45, ha="right")
    ax.set_yticklabels(VARIABLES)
    for i in range(len(VARIABLES)):
        for j in range(len(VARIABLES)):
            ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("CMIP6 variable correlation (center pixel, train+val)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "variable_corr_heatmap.png"), dpi=150)
    plt.close(fig)
    logging.info("Saved variable_corr_heatmap.png")

    # 2. Correlation conditioned on storm / no-storm
    storm    = feat_df[feat_df["positive"] == 1][VARIABLES]
    no_storm = feat_df[feat_df["positive"] == 0][VARIABLES]
    corr_storm    = storm.corr()
    corr_nostorm  = no_storm.corr()
    logging.info(f"\nCorrelation | storm=1:\n{corr_storm.round(3)}")
    logging.info(f"\nCorrelation | storm=0:\n{corr_nostorm.round(3)}")

    # 3. Mean value per variable, storm vs no-storm
    means = feat_df.groupby("positive")[VARIABLES].mean()
    logging.info(f"\nMean variable values by label:\n{means.round(3)}")
    means.to_csv(os.path.join(OUT_DIR, "variable_means_by_label.csv"))

    fig, axes = plt.subplots(1, len(VARIABLES), figsize=(4*len(VARIABLES), 4), sharey=False)
    for ax, var in zip(axes, VARIABLES):
        s_vals = storm[var].values
        n_vals = no_storm[var].values
        ax.violinplot([n_vals[::10], s_vals[::10]], positions=[0, 1],
                      showmedians=True, showextrema=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No storm", "Storm"])
        ax.set_title(var)
        ax.set_ylabel("Normalized value")
    fig.suptitle("CMIP6 variable distribution: storm vs no-storm (center pixel)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "variable_violin_by_label.png"), dpi=150)
    plt.close(fig)
    logging.info("Saved variable_violin_by_label.png")

    # 4. Pairwise scatter (storm colored)
    if len(VARIABLES) >= 2:
        fig, axes = plt.subplots(len(VARIABLES), len(VARIABLES),
                                  figsize=(3*len(VARIABLES), 3*len(VARIABLES)))
        subsample = feat_df.sample(n=min(5000, len(feat_df)), random_state=1)
        colors = ["steelblue" if p == 0 else "tomato" for p in subsample["positive"]]
        for i, v1 in enumerate(VARIABLES):
            for j, v2 in enumerate(VARIABLES):
                ax = axes[i][j]
                if i == j:
                    ax.hist(feat_df[v1], bins=40, color="gray", alpha=0.7)
                    ax.set_xlabel(v1)
                else:
                    ax.scatter(subsample[v2], subsample[v1], c=colors, alpha=0.3, s=3)
                if j == 0:
                    ax.set_ylabel(v1)
                if i == len(VARIABLES)-1:
                    ax.set_xlabel(v2)
        fig.suptitle("Pairwise CMIP6 scatter (red=storm, blue=no-storm)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "variable_pairplot.png"), dpi=120)
        plt.close(fig)
        logging.info("Saved variable_pairplot.png")

    logging.info(f"\nAll outputs saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()

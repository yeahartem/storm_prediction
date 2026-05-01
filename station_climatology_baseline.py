"""
Per-station climatological baseline.

For each station pixel, compute the training-period positive rate
(P(storm) based on historical frequency) and use it as a constant
prediction for all test samples from that station.

If this achieves ~0.84 AUROC -> the models are mostly doing
climatological prediction from lat/lon. If it achieves ~0.7 or lower,
the CNN genuinely learns beyond raw location statistics.

Usage:
    python station_climatology_baseline.py

Reads: data/weatherstation_data/world_stations_2000_2025_25_days_6_months.parquet
"""
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

PARQUET = "data/weatherstation_data/world_stations_2000_2025_25_days_6_months.parquet"
TRAIN_END  = "2022-12-31"  # training + validation
TEST_START = "2023-01-01"
TEST_END   = "2024-12-31"
THRESHOLD  = 15.0          # abs lower bound on per-station threshold

def main():
    logging.info("Loading parquet...")
    df = pd.read_parquet(PARQUET, columns=["STATION", "DATE", "MXWDSP", "LATITUDE", "LONGITUDE"])
    df = df.rename(columns={"STATION": "station_id", "DATE": "date", "LATITUDE": "lat", "LONGITUDE": "lon"})
    df["date"] = pd.to_datetime(df["date"])

    # Split
    train = df[df["date"] <= TRAIN_END].copy()
    test  = df[(df["date"] >= TEST_START) & (df["date"] <= TEST_END)].copy()
    logging.info(f"Train rows: {len(train):,}  Test rows: {len(test):,}")

    # Per-station threshold: max(p95, 15 m/s) on train
    def station_threshold(s):
        q95 = s.quantile(0.95)
        return max(q95, THRESHOLD)

    logging.info("Computing per-station thresholds on train set...")
    thresholds = train.groupby("station_id")["MXWDSP"].apply(station_threshold)
    thresholds.name = "threshold"

    # Per-station positive rate on train (= climatological probability)
    logging.info("Computing per-station storm frequency on train set...")
    train = train.join(thresholds, on="station_id")
    train["positive"] = (train["MXWDSP"] >= train["threshold"]).astype(int)
    clim_rate = train.groupby("station_id")["positive"].mean()
    clim_rate.name = "clim_rate"

    # Evaluate on test
    test = test.join(thresholds, on="station_id")
    test = test.join(clim_rate, on="station_id")
    test = test.dropna(subset=["threshold", "clim_rate"])
    test["positive"] = (test["MXWDSP"] >= test["threshold"]).astype(int)

    y_true = test["positive"].values
    y_pred = test["clim_rate"].values   # constant per-station probability

    pos_rate = y_true.mean()
    logging.info(f"Test samples: {len(test):,}  positive rate: {pos_rate:.3f}")

    auroc = roc_auc_score(y_true, y_pred)
    ap    = average_precision_score(y_true, y_pred)
    bs    = brier_score_loss(y_true, y_pred)
    bs_clim = pos_rate * (1 - pos_rate)
    bss   = 1.0 - bs / bs_clim

    print("\n=== Per-station climatological baseline ===")
    print(f"  AUROC : {auroc:.4f}")
    print(f"  AP    : {ap:.4f}")
    print(f"  BS    : {bs:.4f}")
    print(f"  BSS   : {bss:.4f}")
    print()
    print("Interpretation:")
    print(f"  Best CNN AUROC = 0.851 -> CNN adds {0.851 - auroc:.3f} AUROC above climatology")
    print(f"  elev-only      = 0.844 -> elev adds {0.844 - auroc:.3f} AUROC above climatology")
    print(f"  pr-only        = 0.802 -> pr   adds {0.802 - auroc:.3f} AUROC above climatology")

    # Save per-station rates for inspection
    out = clim_rate.reset_index()
    out.columns = ["station_id", "clim_positive_rate"]
    out.to_csv("station_climatology_rates.csv", index=False)
    logging.info("Saved station_climatology_rates.csv")

if __name__ == "__main__":
    main()

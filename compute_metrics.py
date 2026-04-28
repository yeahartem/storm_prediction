"""
compute_metrics.py -- evaluate a checkpoint and compute all test metrics.

Computes: AUROC, AP, BSS, F1 (threshold calibrated on 2021-2022),
Precision, Recall -- globally + per-region + per-season.

Also computes AP and BSS for the two baselines (CMIP threshold + LR)
that are still TBD in the paper table.

Usage (on server):
  # Best model (pr+tasmax+tasmin+elev)
  python compute_metrics.py \
      train=train_cmip6_world_noprecip_1gpu \
      ckpt_path="out/2025-XX-XX/XX-XX-XX/epoch=22-step=XXXX.ckpt" \
      eval_out="eval_results/best_model"

  # Ablation: +sfcWindmax
  python compute_metrics.py \
      train=train_cmip6_world_sfcwind_1gpu \
      ckpt_path="out/.../epoch=16.ckpt" \
      eval_out="eval_results/sfcwind"

  # Baselines only (no checkpoint needed)
  python compute_metrics.py \
      train=train_cmip6_world_noprecip_1gpu \
      ckpt_path=null \
      eval_out="eval_results/baselines"
"""
import sys, os
sys.path.append(os.getcwd())

# Monkey-patch torch.load for PL 2.0.2 + PyTorch 2.6+ compatibility
import torch
_orig = torch.load
def _patched(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig(*a, **kw)
torch.load = _patched

import warnings
warnings.filterwarnings("ignore")
import logging
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             brier_score_loss)
from sklearn.linear_model import LogisticRegression
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.regression.models.pl_module import WindNetPL
from src.regression.datamodule import XarrayDataset, XarrayDatasetElev
from src.regression.data_load import DataPreLoader


# ---------------------------------------------------------------------------
# Region bounding boxes: (lat_min, lat_max, lon_min, lon_max)
# ---------------------------------------------------------------------------
REGIONS = {
    "russia_europe":       ( 50,  70,  20,  60),
    "russia_west_siberia": ( 50,  70,  60,  90),
    "russia_east_siberia": ( 50,  70,  90, 130),
    "russia_far_east":     ( 40,  65, 130, 170),
    "africa_equatorial":   (-10,  10,   0,  50),
    "africa_south":        (-35, -10,  15,  45),
    "africa_north_east":   ( 10,  35,  25,  55),
    "africa_sahel_east":   ( 10,  20,  10,  40),
}

SEASON_OF = {12: "DJF", 1: "DJF",  2: "DJF",
              3: "MAM", 4: "MAM",  5: "MAM",
              6: "JJA", 7: "JJA",  8: "JJA",
              9: "SON", 10: "SON", 11: "SON"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def binary_labels(idxs):
    """Compute binary labels: 1 if y_max >= per-station threshold."""
    return (idxs[7, :] >= idxs[8, :]).astype(np.int32)


def get_meta(idxs):
    """Return lat (deg), lon (deg), month (1-12) from data_idxs."""
    lat   = idxs[5, :] * 90.0
    lon   = idxs[6, :] * 180.0
    month = np.clip(np.round(idxs[4, :] * 12).astype(int), 1, 12)
    return lat, lon, month


def compute_metrics(probs, labels, threshold, clim_rate=None):
    if labels.sum() == 0:
        return dict(AUROC=float('nan'), AP=float('nan'), BSS=float('nan'),
                    F1=float('nan'), Prec=float('nan'), Rec=float('nan'),
                    n_pos=0, n_total=int(len(labels)), pos_rate=0.0,
                    threshold=float(threshold))
    preds = (probs >= threshold).astype(int)
    bs    = brier_score_loss(labels, probs)
    cr    = float(labels.mean()) if clim_rate is None else clim_rate
    bs_c  = cr * (1.0 - cr)
    bss   = float(1.0 - bs / bs_c) if bs_c > 1e-9 else float('nan')
    return dict(
        AUROC    = float(roc_auc_score(labels, probs)),
        AP       = float(average_precision_score(labels, probs)),
        BSS      = bss,
        F1       = float(f1_score(labels, preds, zero_division=0)),
        Prec     = float(precision_score(labels, preds, zero_division=0)),
        Rec      = float(recall_score(labels, preds, zero_division=0)),
        n_pos    = int(labels.sum()),
        n_total  = int(len(labels)),
        pos_rate = float(labels.mean()),
        threshold= float(threshold),
    )


def calibrate_threshold(probs, labels):
    """Find p* in [0.05, 0.95] that maximises F1 on val data."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.005):
        f1 = f1_score(labels, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    logging.info(f"Calibrated threshold: {best_t:.3f}  (val F1={best_f1:.4f})")
    return best_t


# ---------------------------------------------------------------------------
# CNN inference
# ---------------------------------------------------------------------------
def _make_dataset(DPL, cfg, data_idxs):
    """Create a Dataset from arbitrary data_idxs without triggering the full DPL pipeline."""
    Cls = XarrayDatasetElev if cfg.train.use_elevation else XarrayDataset
    # Create instance bypassing __init__ (all needed attrs set manually)
    ds = object.__new__(Cls)
    ds.cfg           = cfg
    ds.dataset_torch = DPL.dataset_torch
    ds.data_idxs     = data_idxs
    ds._DPL          = None
    ds.dtype         = torch.float32
    ds.dense_weighter = None
    if cfg.train.use_elevation:
        ds.elevation_torch = DPL.elevation_torch
    return ds


@torch.no_grad()
def cnn_inference(model, DPL, cfg, data_idxs, device, batch_size=256):
    ds     = _make_dataset(DPL, cfg, data_idxs)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=4,
                        pin_memory=True, shuffle=False, drop_last=False)
    all_probs = []
    for batch in tqdm(loader, desc="Inference", unit="batch"):
        X_list = [t.to(device) for t in batch[0]]
        logits = model(X_list)
        probs  = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------
def baseline_cmip_threshold(DPL, cfg, val_idxs, test_idxs):
    """
    CMIP threshold baseline: use raw CMIP6 sfcWindmax at the co-located cell
    as the anomaly score. sfcWindmax is variable index 0 in the dataset.

    We need the sfcWindmax value at (lat_idx, lon_idx, time_idx) for each sample.
    """
    sfcwind_ch = None
    for i, var in enumerate(cfg.train.variables):
        if 'sfcWind' in var or 'wind' in var.lower():
            sfcwind_ch = i
            break

    if sfcwind_ch is None:
        logging.warning("sfcWindmax not in this config's variables -- skipping CMIP baseline")
        return None, None

    logging.info(f"CMIP baseline: sfcWindmax channel {sfcwind_ch}")

    def extract_wind(idxs):
        # data_idxs rows: 0=lat_idx, 1=lon_idx, 2=time_idx (all shifted by padding)
        lat_idxs  = idxs[0, :].astype(int)
        lon_idxs  = idxs[1, :].astype(int)
        time_idxs = idxs[2, :].astype(int)
        # dataset_torch shape: (C, T, lat_padded, lon_padded)
        scores = DPL.dataset_torch[sfcwind_ch, time_idxs, lat_idxs, lon_idxs].numpy()
        return scores.astype(np.float32)

    scores_val  = extract_wind(val_idxs)
    labels_val  = binary_labels(val_idxs)
    scores_test = extract_wind(test_idxs)
    labels_test = binary_labels(test_idxs)

    # Min-max scale to [0,1] using val range (CMIP values are normalized, not probs)
    s_min, s_max = float(scores_val.min()), float(scores_val.max())
    scores_val  = (scores_val  - s_min) / (s_max - s_min + 1e-8)
    scores_test = np.clip((scores_test - s_min) / (s_max - s_min + 1e-8), 0.0, 1.0)

    # Calibrate threshold on val by maximising F1
    threshold = calibrate_threshold(scores_val, labels_val)

    m = compute_metrics(scores_test, labels_test, threshold)
    m["method"] = "CMIP_threshold_baseline"
    logging.info(f"CMIP baseline: AUROC={m['AUROC']:.4f}  AP={m['AP']:.4f}  BSS={m['BSS']:.4f}")
    return m, threshold


def baseline_logistic(DPL, cfg, train_idxs, val_idxs, test_idxs):
    """
    Logistic regression on: [sfcWindmax_local, sin(2pi*month/12), cos(2pi*month/12)].
    Trained on train (2000-2020), threshold calibrated on val (2021-2022).
    """
    sfcwind_ch = None
    for i, var in enumerate(cfg.train.variables):
        if 'sfcWind' in var or 'wind' in var.lower():
            sfcwind_ch = i
            break

    if sfcwind_ch is None:
        logging.warning("sfcWindmax not in this config's variables -- skipping LR baseline")
        return None

    def extract_features(idxs):
        lat_idxs  = idxs[0, :].astype(int)
        lon_idxs  = idxs[1, :].astype(int)
        time_idxs = idxs[2, :].astype(int)
        wind = DPL.dataset_torch[sfcwind_ch, time_idxs, lat_idxs, lon_idxs].numpy().astype(np.float32)
        month = np.clip(np.round(idxs[4, :] * 12).astype(int), 1, 12)
        sin_m = np.sin(2 * np.pi * month / 12).astype(np.float32)
        cos_m = np.cos(2 * np.pi * month / 12).astype(np.float32)
        return np.stack([wind, sin_m, cos_m], axis=1)

    # Use only train period 2000-2020 for fitting
    val_start_idx = np.searchsorted(DPL.time_coords, np.datetime64('2021-01-01', 'D'))
    pure_train_idxs = train_idxs[:, train_idxs[2, :] < val_start_idx]

    X_train = extract_features(pure_train_idxs)
    y_train = binary_labels(pure_train_idxs)
    X_val   = extract_features(val_idxs)
    y_val   = binary_labels(val_idxs)
    X_test  = extract_features(test_idxs)
    y_test  = binary_labels(test_idxs)

    logging.info(f"LR baseline: fitting on {len(X_train)} train samples...")
    lr = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr.fit(X_train, y_train)

    probs_val  = lr.predict_proba(X_val)[:, 1]
    probs_test = lr.predict_proba(X_test)[:, 1]

    threshold = calibrate_threshold(probs_val, y_val)
    m = compute_metrics(probs_test, y_test, threshold)
    m["method"] = "LogisticRegression_baseline"
    logging.info(f"LR baseline: AUROC={m['AUROC']:.4f}  AP={m['AP']:.4f}  BSS={m['BSS']:.4f}")
    return m


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def run_eval(cfg: DictConfig, ckpt_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"Output dir: {out_dir}")

    # ── Load data ─────────────────────────────────────────────────────────
    logging.info("Loading data (DataPreLoader)...")
    DPL = DataPreLoader(cfg)

    # Split train_data_idxs into pure train (2000-2020) and val (2021-2022)
    val_start = np.searchsorted(DPL.time_coords, np.datetime64('2021-01-01', 'D'))
    all_train = DPL.train_data_idxs  # everything before 2023-01-01
    val_mask  = all_train[2, :] >= val_start
    val_idxs  = all_train[:,  val_mask]   # 2021-2022
    test_idxs = DPL.test_data_idxs        # 2023-2024

    logging.info(f"Val samples  (2021-2022): {val_idxs.shape[1]}")
    logging.info(f"Test samples (2023-2024): {test_idxs.shape[1]}")

    # ── Baselines (always computed, no checkpoint needed) ──────────────────
    baseline_rows = []
    m_cmip, _ = baseline_cmip_threshold(DPL, cfg, val_idxs, test_idxs)
    if m_cmip:
        baseline_rows.append(m_cmip)

    m_lr = baseline_logistic(DPL, cfg, all_train, val_idxs, test_idxs)
    if m_lr:
        baseline_rows.append(m_lr)

    if baseline_rows:
        pd.DataFrame(baseline_rows).to_csv(
            os.path.join(out_dir, "baseline_metrics.csv"), index=False)
        logging.info("Saved baseline_metrics.csv")

    # ── CNN evaluation (skipped if ckpt_path is null) ─────────────────────
    if not ckpt_path or ckpt_path.lower() in ("null", "none", ""):
        logging.info("No checkpoint provided -- baselines only.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")
    logging.info(f"Loading checkpoint: {ckpt_path}")

    # Infer in_chans from checkpoint to avoid size mismatch when config differs from training config
    _ckpt_tmp = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_in_chans = _ckpt_tmp["state_dict"]["net.ghostnetv2.conv_stem.weight"].shape[1]
    del _ckpt_tmp
    if ckpt_in_chans != cfg.train.in_chans:
        logging.warning(f"in_chans mismatch: config={cfg.train.in_chans}, checkpoint={ckpt_in_chans} -- overriding with checkpoint value")
        cfg.train.in_chans = ckpt_in_chans

    model = WindNetPL.load_from_checkpoint(
        ckpt_path, cfg=cfg, run_dir=out_dir, strict=False).eval().to(device)

    # Val inference → threshold calibration
    logging.info("Val inference (2021-2022) for threshold calibration...")
    probs_val  = cnn_inference(model, DPL, cfg, val_idxs, device)
    labels_val = binary_labels(val_idxs)
    threshold  = calibrate_threshold(probs_val, labels_val)

    # Test inference
    logging.info("Test inference (2023-2024)...")
    probs_test  = cnn_inference(model, DPL, cfg, test_idxs, device)
    labels_test = binary_labels(test_idxs)
    lats, lons, months = get_meta(test_idxs)

    # ── Global metrics ─────────────────────────────────────────────────────
    gm = compute_metrics(probs_test, labels_test, threshold)
    gm["split"] = "global"
    logging.info(f"GLOBAL: AUROC={gm['AUROC']:.4f}  AP={gm['AP']:.4f}  "
                 f"BSS={gm['BSS']:.4f}  F1={gm['F1']:.4f}@{threshold:.3f}")
    pd.DataFrame([gm]).to_csv(os.path.join(out_dir, "global_metrics.csv"), index=False)

    # Save raw test predictions for bootstrap CI (run: python bootstrap_ci.py test_predictions.csv)
    pd.DataFrame({"prediction": probs_test, "target": labels_test}).to_csv(
        os.path.join(out_dir, "test_predictions.csv"), index=False)
    logging.info("Saved test_predictions.csv for bootstrap CI")

    # ── Regional metrics ───────────────────────────────────────────────────
    reg_rows = []
    for reg, (la0, la1, lo0, lo1) in REGIONS.items():
        mask = (lats >= la0) & (lats <= la1) & (lons >= lo0) & (lons <= lo1)
        n = mask.sum()
        if n < 30:
            logging.warning(f"  {reg}: {n} samples -- skipped")
            continue
        rm = compute_metrics(probs_test[mask], labels_test[mask], threshold)
        rm["region"] = reg
        reg_rows.append(rm)
        logging.info(f"  {reg:25s}: AUROC={rm['AUROC']:.3f}  AP={rm['AP']:.3f}  n={n}")
    pd.DataFrame(reg_rows).to_csv(os.path.join(out_dir, "regional_metrics.csv"), index=False)

    # ── Seasonal metrics ───────────────────────────────────────────────────
    seas_rows = []
    for season in ["DJF", "MAM", "JJA", "SON"]:
        mask = np.array([SEASON_OF.get(m, "?") == season for m in months])
        n = mask.sum()
        if n < 30:
            continue
        sm = compute_metrics(probs_test[mask], labels_test[mask], threshold)
        sm["season"] = season
        seas_rows.append(sm)
        logging.info(f"  {season}: AUROC={sm['AUROC']:.3f}  AP={sm['AP']:.3f}  n={n}")
    pd.DataFrame(seas_rows).to_csv(os.path.join(out_dir, "seasonal_metrics.csv"), index=False)

    logging.info(f"All results saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------
@hydra.main(version_base=None,
            config_path=os.path.join(os.getcwd(), "configs"),
            config_name="cmip6_world.yaml")
def main(cfg: DictConfig):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[logging.StreamHandler()])

    # Accept ckpt_path / eval_out via env vars (avoids Hydra grammar issues
    # when checkpoint filenames contain '=' like epoch=22-step=115000.ckpt)
    ckpt_path = os.environ.get("CKPT_PATH") or cfg.get("ckpt_path", None)
    out_dir   = os.environ.get("EVAL_OUT")  or cfg.get("eval_out", "eval_results/default")
    run_eval(cfg, ckpt_path, out_dir)


if __name__ == "__main__":
    sys.argv.append('hydra.run.dir=out/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()

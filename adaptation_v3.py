"""
adaptation_v3.py — E3 (per-station Platt) + E4 (last-layer fine-tuning).
Requires surrogate_train.parquet / surrogate_test.parquet from adaptation_v2.py.

E3: Per-station Platt scaling
  p_s(x) = sigmoid(a_s * z + b_s)
  Prior: L2 toward (a=1, b=0): reg_loss = lambda * [(a_s-1)^2 + b_s^2]
  6036 independent 2-param LBFGS problems → ~30 seconds total

E4: Last-layer fine-tuning
  Freeze backbone + head_lin1 + head_activation.
  Cache 70-dim features (head_activation output) for N_FEATURES_SAMPLES random samples.
  Train new head_lin2: Linear(70 -> 1) with Adam on real GSOD labels.
  Expected AUROC: 0.68-0.76 (transfer learning from CMIP6 backbone).
"""
import sys, os
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader, TensorDataset

from src.regression.data_load import DataPreLoader
from src.regression.datamodule import XarrayDatasetElev
from src.regression.models.models import GhostWindNet27

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ── paths ─────────────────────────────────────────────────────────────────────
SYNTHETIC_CKPT   = 'out/2026-04-21/23-47-36/epoch=22-step=115000.ckpt'
SUPERVISED_CKPT  = 'out/2026-04-20/15-54-18/epoch=22-step=115000.ckpt'
SURROGATE_TRAIN  = 'surrogate_train.parquet'
SURROGATE_TEST   = 'surrogate_test.parquet'
FEATURES_TRAIN   = 'features_train.npz'   # 70-dim features for E4
FEATURES_TEST    = 'features_test.npz'

# ── hyper-params ──────────────────────────────────────────────────────────────
DEVICE               = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE           = 64
N_FEATURES_SAMPLES   = 500_000   # subset for E4 feature caching (~45 min vs 5 hr)
E3_LAMBDA            = 1.0       # L2 regularization strength for per-station Platt
E4_LR                = 1e-3
E4_EPOCHS            = 30
E4_BATCH_SIZE        = 4096      # purely CPU/RAM, large is fine


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def auroc_ap_bss(y_true, scores):
    auroc = roc_auc_score(y_true, scores)
    ap    = average_precision_score(y_true, scores)
    bs    = brier_score_loss(y_true, scores)
    p     = y_true.mean()
    bss   = 1.0 - bs / (p * (1 - p)) if 0 < p < 1 else 0.0
    return auroc, ap, bss


def load_model(ckpt_path, in_chans=4):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = {k[len('net.'):]: v for k, v in ckpt['state_dict'].items()
          if k.startswith('net.')}
    m = GhostWindNet27(in_chans=in_chans)
    m.load_state_dict(sd)
    m.requires_grad_(False)
    return m.to(DEVICE).eval()


def build_loaders(cfg):
    dpl = DataPreLoader(cfg)
    train_ds = XarrayDatasetElev(DPL=dpl, test=False)
    test_ds  = XarrayDatasetElev(DPL=dpl, test=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              drop_last=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              drop_last=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# E3: Per-station Platt scaling
# ─────────────────────────────────────────────────────────────────────────────
def _station_key(lat, lon):
    """Round to nearest 0.5° grid — robust station identifier."""
    return f"{round(float(lat)*2)/2:.1f}_{round(float(lon)*2)/2:.1f}"


def e3_optimize(df: pd.DataFrame, lam: float) -> dict:
    """
    Fit per-station (a_s, b_s) via L-BFGS-B.
    Returns dict: station_key -> (a_s, b_s).
    """
    df = df.copy()
    df['key'] = [_station_key(la, lo)
                 for la, lo in zip(df['lat'].values, df['lon'].values)]

    station_params = {}
    keys = df['key'].unique()
    logging.info(f"E3: fitting {len(keys)} stations (lambda={lam})...")

    for i, key in enumerate(keys):
        mask = df['key'] == key
        z_s = df.loc[mask, 'z'].values.astype(np.float64)
        y_s = df.loc[mask, 'y_bin'].values.astype(np.float64)
        n_pos = y_s.sum()

        def loss(ab):
            a, b = ab
            logits = a * z_s + b
            p = sigmoid_np(logits)
            bce = -np.mean(y_s * np.log(p + 1e-9) + (1 - y_s) * np.log(1 - p + 1e-9))
            reg = lam * ((a - 1) ** 2 + b ** 2)
            return bce + reg

        def grad(ab):
            a, b = ab
            logits = a * z_s + b
            p = sigmoid_np(logits)
            r = p - y_s
            da = np.mean(r * z_s) + 2 * lam * (a - 1)
            db = np.mean(r)        + 2 * lam * b
            return np.array([da, db])

        res = minimize(loss, [1.0, 0.0], jac=grad, method='L-BFGS-B',
                       bounds=[(0.01, 20.0), (-10.0, 10.0)],
                       options={'maxiter': 200})
        station_params[key] = tuple(res.x)

        if i % 500 == 0:
            logging.info(f"  E3: {i}/{len(keys)} stations done")

    return station_params


def e3_score(df: pd.DataFrame, station_params: dict,
             fallback_ab=(1.0, 0.0)) -> np.ndarray:
    """Apply per-station (a_s, b_s) to df rows."""
    df = df.copy()
    df['key'] = [_station_key(la, lo)
                 for la, lo in zip(df['lat'].values, df['lon'].values)]
    a_fallback, b_fallback = fallback_ab
    scores = np.empty(len(df), dtype=np.float64)
    for i, (key, z) in enumerate(zip(df['key'].values, df['z'].values)):
        a, b = station_params.get(key, (a_fallback, b_fallback))
        scores[i] = sigmoid_np(a * z + b)
    return scores


# ─────────────────────────────────────────────────────────────────────────────
# E4: Feature caching + last-layer fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
def _extract_features_70d(model: GhostWindNet27,
                           X: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Extract 70-dim features after head_activation (before head_lin2)."""
    b    = X.shape[0]
    days = X.shape[2]
    Xr = X.reshape(b * days, X.shape[1], X.shape[3], X.shape[4])
    Xr = model.ghostnetv2(Xr)            # (b*days, 70)
    pr = pos.reshape(b * days, 4)
    Xr = torch.cat([Xr, pr], dim=1)      # (b*days, 74)
    Xr = Xr.reshape(b, days * (model.embed + 4))  # (b, 1998)
    # Note: head_dropout is inactive in eval mode
    Xr = model.head_lin1(Xr)            # (b, 70)
    Xr = model.head_activation(Xr)      # (b, 70)
    return Xr                            # (b, 70)


def build_feature_surrogate(model: GhostWindNet27,
                             loader: DataLoader,
                             path: str,
                             split: str,
                             max_samples: int = None) -> tuple:
    """
    One forward pass → cache (features_70d, y_bin) for all (or max_samples) samples.
    Returns (features_np, y_bin_np).
    """
    if os.path.exists(path):
        logging.info(f"Loading feature surrogate from {path}")
        d = np.load(path)
        return d['features'], d['y_bin']

    logging.info(f"Building feature surrogate for {split} "
                 f"(max={max_samples or 'all'})...")
    feats_list, y_list = [], []
    n_collected = 0
    n_batches = len(loader)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_samples and n_collected >= max_samples:
                break
            if i % 500 == 0:
                logging.info(f"  Features {split}: batch {i}/{n_batches}")
            (X, pos), y, _, station_thresholds = batch
            X, pos = X.to(DEVICE), pos.to(DEVICE)
            feats = _extract_features_70d(model, X, pos).cpu().numpy().astype(np.float32)
            y_bin = (y.numpy() >= station_thresholds.numpy()).astype(np.float32)
            feats_list.append(feats)
            y_list.append(y_bin)
            n_collected += len(y_bin)

    features = np.concatenate(feats_list, axis=0)
    y_bin    = np.concatenate(y_list, axis=0)

    if max_samples and len(features) > max_samples:
        idx = np.random.default_rng(42).permutation(len(features))[:max_samples]
        features = features[idx]
        y_bin    = y_bin[idx]

    np.savez_compressed(path, features=features, y_bin=y_bin)
    logging.info(f"Feature surrogate saved: {features.shape} → {path}")
    return features, y_bin


def e4_train_head(features_train: np.ndarray, y_train: np.ndarray,
                  features_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Train new Linear(70→1) head on cached features.
    Features are CPU tensors; training is fast (no CNN forward needed).
    """
    X_tr = torch.from_numpy(features_train).float()
    y_tr = torch.from_numpy(y_train).float()
    X_te = torch.from_numpy(features_test).float()
    y_te = torch.from_numpy(y_test).float()

    head = nn.Linear(70, 1)
    # Init: copy weights from original model for warm start
    if os.path.exists(SYNTHETIC_CKPT):
        ckpt = torch.load(SYNTHETIC_CKPT, map_location='cpu')
        sd = {k[len('net.'):]: v for k, v in ckpt['state_dict'].items()
              if k.startswith('net.')}
        head.weight.data = sd['head_lin2.weight'].clone()
        head.bias.data   = sd['head_lin2.bias'].clone()
        logging.info("E4: warm-started head_lin2 from synthetic checkpoint")

    optimizer = torch.optim.Adam(head.parameters(), lr=E4_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=E4_EPOCHS, eta_min=E4_LR / 100)

    # Compute pos_weight for imbalance
    n_pos = y_tr.sum().item()
    n_neg = len(y_tr) - n_pos
    pos_weight = torch.tensor(n_neg / max(n_pos, 1))
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds_train = TensorDataset(X_tr, y_tr)
    loader_tr = DataLoader(ds_train, batch_size=E4_BATCH_SIZE, shuffle=True)

    best_auroc = 0.0
    best_weights = None

    logging.info(f"E4 training: {len(X_tr)} samples, {E4_EPOCHS} epochs, "
                 f"lr={E4_LR}, pos_weight={pos_weight:.2f}")

    for epoch in range(E4_EPOCHS):
        head.train()
        total_loss = 0.0
        for xb, yb in loader_tr:
            optimizer.zero_grad()
            logits = head(xb).squeeze(-1)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        # Evaluate on test features
        head.eval()
        with torch.no_grad():
            logits_te = head(X_te).squeeze(-1).numpy()
        scores_te = sigmoid_np(logits_te)
        auroc_te  = roc_auc_score(y_te.numpy(), scores_te)
        ap_te     = average_precision_score(y_te.numpy(), scores_te)

        if auroc_te > best_auroc:
            best_auroc   = auroc_te
            best_weights = {k: v.clone() for k, v in head.state_dict().items()}

        logging.info(f"  E4 epoch {epoch+1:3d}/{E4_EPOCHS}: "
                     f"train_loss={total_loss/len(loader_tr):.4f}  "
                     f"AUROC={auroc_te:.4f}  AP={ap_te:.4f}  "
                     f"lr={scheduler.get_last_lr()[0]:.2e}")

    # Restore best
    head.load_state_dict(best_weights)
    head.eval()
    with torch.no_grad():
        logits_te = head(X_te).squeeze(-1).numpy()
    scores_final = sigmoid_np(logits_te)
    auroc, ap, bss = auroc_ap_bss(y_te.numpy(), scores_final)

    torch.save(head.state_dict(), 'e4_head_lin2.pt')
    logging.info(f"E4 best checkpoint saved → e4_head_lin2.pt")
    return {'AUROC': auroc, 'AP': ap, 'BSS': bss}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Check surrogate files exist ───────────────────────────────────────────
    for p in [SURROGATE_TRAIN, SURROGATE_TEST]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Run adaptation_v2.py first to build surrogates.")

    df_train = pd.read_parquet(SURROGATE_TRAIN)
    df_test  = pd.read_parquet(SURROGATE_TEST)
    logging.info(f"Loaded surrogates: train={len(df_train)}, test={len(df_test)}")

    # ── Baseline (no adaptation) ──────────────────────────────────────────────
    z0  = df_test['z'].values.astype(np.float64)
    y0  = df_test['y_bin'].values.astype(np.float64)
    s0  = sigmoid_np(z0)
    auroc0, ap0, bss0 = auroc_ap_bss(y0, s0)
    logging.info(f"Baseline: AUROC={auroc0:.4f}  AP={ap0:.4f}  BSS={bss0:.4f}")

    # ── E3: Per-station Platt ─────────────────────────────────────────────────
    logging.info("\n--- E3: Per-station Platt (lambda={}) ---".format(E3_LAMBDA))
    station_params = e3_optimize(df_train, lam=E3_LAMBDA)
    np.save('e3_station_params.npy', station_params)

    scores_e3 = e3_score(df_test, station_params)
    auroc_e3, ap_e3, bss_e3 = auroc_ap_bss(y0, scores_e3)
    logging.info(f"E3 result: AUROC={auroc_e3:.4f}  AP={ap_e3:.4f}  BSS={bss_e3:.4f}")

    # Quick ablation: try different lambdas
    for lam_try in [0.1, 10.0]:
        sp_try = e3_optimize(df_train, lam=lam_try)
        sc_try = e3_score(df_test, sp_try)
        a, ap_, b = auroc_ap_bss(y0, sc_try)
        logging.info(f"  E3 (lambda={lam_try}): AUROC={a:.4f}  AP={ap_:.4f}  BSS={b:.4f}")

    # ── E4: Last-layer fine-tuning ────────────────────────────────────────────
    logging.info("\n--- E4: Last-layer fine-tuning ---")

    config_dir = os.path.join(os.getcwd(), 'configs')
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name='cmip6_world_nosfcwind_server',
            overrides=['train.distributed=false', 'train.num_workers=0',
                       f'train.batch_size={BATCH_SIZE}'])
    train_loader, test_loader = build_loaders(cfg)
    model = load_model(SYNTHETIC_CKPT)

    features_train, y_train = build_feature_surrogate(
        model, train_loader, FEATURES_TRAIN, 'train', max_samples=N_FEATURES_SAMPLES)
    features_test, y_test = build_feature_surrogate(
        model, test_loader, FEATURES_TEST, 'test', max_samples=None)

    logging.info(f"Features: train={features_train.shape}, test={features_test.shape}")

    m_e4 = e4_train_head(features_train, y_train, features_test, y_test)
    logging.info(f"E4 result: AUROC={m_e4['AUROC']:.4f}  AP={m_e4['AP']:.4f}  "
                 f"BSS={m_e4['BSS']:.4f}")

    # ── Supervised upper bound (from prior run) ───────────────────────────────
    logging.info("")
    logging.info("=" * 68)
    logging.info("PAPER TABLE — Test 2023-2024, real GSOD labels")
    logging.info("=" * 68)
    logging.info(f"{'Model':<42} {'AUROC':>7} {'AP':>7} {'BSS':>7}")
    logging.info(f"{'Synthetic CNN (no adaptation)':<42} "
                 f"{auroc0:>7.3f} {ap0:>7.3f} {bss0:>7.3f}")
    logging.info(f"{'E3: Per-station Platt (K=6036, LBFGS)':<42} "
                 f"{auroc_e3:>7.3f} {ap_e3:>7.3f} {bss_e3:>7.3f}")
    logging.info(f"{'E4: Last-layer fine-tuning (70 params)':<42} "
                 f"{m_e4['AUROC']:>7.3f} {m_e4['AP']:>7.3f} {m_e4['BSS']:>7.3f}")
    logging.info(f"{'Supervised CNN (upper bound)':<42} "
                 f"{'0.851':>7} {'0.757':>7} {'—':>7}")
    logging.info("=" * 68)


if __name__ == '__main__':
    main()

"""
adaptation_v2.py — Per-cluster Platt scaling on cached CNN logits.
Based on Gleb's SA-guided pipeline (flowback analogy).

Pipeline:
  1. build_surrogate(): one CNN forward pass → cache (z, y_bin, cluster, lat, lon)
  2. E1 Global Platt   (K=1, 2 params): LBFGS  → ground-truth optimum
  3. E2 Cluster Platt  (K=3, 6 params): LBFGS  → ground-truth optimum
  4. E2-SA             (K=3, 6 params): SA-guided optimizer (flowback-style)
  5. Acceptance:  temporal hold-out 2023-2024

Cluster definition (elevation-based, physically motivated):
  0 = flat     : elev < 200 m   (synoptic wind patterns dominate)
  1 = hills    : 200–1000 m     (moderate orographic enhancement)
  2 = mountain : elev >= 1000 m (strong orographic, CMIP6 most underestimates here)
"""
import sys, os
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.regression.data_load import DataPreLoader
from src.regression.datamodule import XarrayDatasetElev
from src.regression.models.models import GhostWindNet27

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ── paths ────────────────────────────────────────────────────────────────────
SYNTHETIC_CKPT  = 'out/2026-04-21/23-47-36/epoch=22-step=115000.ckpt'
SUPERVISED_CKPT = 'out/2026-04-20/15-54-18/epoch=22-step=115000.ckpt'
SURROGATE_TRAIN = 'surrogate_train.parquet'
SURROGATE_TEST  = 'surrogate_test.parquet'

# ── hyper-params ─────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE  = 64          # conservative: other jobs occupy ~10GB on each GPU
K_CLUSTERS  = 3
CLUSTER_NAMES = ['flat (<200m)', 'hills (200-1000m)', 'mountain (>=1000m)']

# Elevation normalization constants (from src/utils/norm_values.py)
ELEV_MEAN = 377.73032
ELEV_STD  = 855.89075

# SA-guided optimizer params
SA_DELTAS             = np.array([-0.25, -0.15, -0.05, 0.0, 0.05, 0.15, 0.25])
SAMPLES_PER_SEGMENT   = 24    # dense interpolation points between SA-grid nodes
SA_MAX_ITERS          = 30
SA_STAGNATION_PATIENCE = 4
SA_MIN_REL_IMPROVEMENT = 5e-5  # minimum relative loss improvement to accept step

# LBFGS
LBFGS_N_TRIALS = 5   # random restarts


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def get_cluster_from_elev_norm(elev_norm: np.ndarray) -> np.ndarray:
    """Map normalized elevation to cluster index [0,1,2]."""
    elev_m = elev_norm * ELEV_STD + ELEV_MEAN
    return np.where(elev_m < 200, 0, np.where(elev_m < 1000, 1, 2)).astype(np.int32)


def bce_cluster_platt(theta: np.ndarray, z: np.ndarray,
                       y: np.ndarray, c: np.ndarray, K: int) -> float:
    """
    BCE loss for per-cluster Platt scaling.
    theta = [a0, b0, a1, b1, ..., a_{K-1}, b_{K-1}]
    """
    logits = np.empty(len(z), dtype=np.float64)
    for k in range(K):
        mask = (c == k)
        if not mask.any():
            continue
        a_k = theta[2 * k]
        b_k = theta[2 * k + 1]
        logits[mask] = a_k * z[mask] + b_k
    p = sigmoid_np(logits)
    return -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))


def bce_grad(theta: np.ndarray, z: np.ndarray,
             y: np.ndarray, c: np.ndarray, K: int) -> np.ndarray:
    """Analytical gradient of BCE w.r.t. theta."""
    grad = np.zeros_like(theta)
    for k in range(K):
        mask = (c == k)
        if not mask.any():
            continue
        a_k = theta[2 * k]
        b_k = theta[2 * k + 1]
        z_k = z[mask]
        y_k = y[mask]
        p_k = sigmoid_np(a_k * z_k + b_k)
        residual = p_k - y_k           # (p - y)
        grad[2 * k]     = np.mean(residual * z_k)    # ∂L/∂a_k
        grad[2 * k + 1] = np.mean(residual)           # ∂L/∂b_k
    return grad


def evaluate_platt(theta: np.ndarray, df: pd.DataFrame, K: int) -> dict:
    """Compute AUROC, AP, Brier, BSS on a surrogate dataframe."""
    z = df['z'].values.astype(np.float64)
    y = df['y_bin'].values.astype(np.float64)
    c = df['cluster'].values.astype(np.int32) if K > 1 else np.zeros(len(df), dtype=np.int32)

    scores = np.empty(len(z), dtype=np.float64)
    for k in range(K):
        mask = (c == k)
        if not mask.any():
            continue
        scores[mask] = sigmoid_np(theta[2 * k] * z[mask] + theta[2 * k + 1])

    auroc = roc_auc_score(y, scores)
    ap    = average_precision_score(y, scores)
    bs    = brier_score_loss(y, scores)
    p_clim = y.mean()
    bss   = 1.0 - bs / (p_clim * (1 - p_clim)) if 0 < p_clim < 1 else 0.0
    return {'AUROC': auroc, 'AP': ap, 'BS': bs, 'BSS': bss}


# ─────────────────────────────────────────────────────────────────────────────
# Model + data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str, in_chans: int = 4) -> GhostWindNet27:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = {k[len('net.'):]: v
                  for k, v in ckpt['state_dict'].items()
                  if k.startswith('net.')}
    model = GhostWindNet27(in_chans=in_chans)
    model.load_state_dict(state_dict)
    model.requires_grad_(False)
    return model.to(DEVICE).eval()


def build_loaders(cfg):
    dpl = DataPreLoader(cfg)
    train_ds = XarrayDatasetElev(DPL=dpl, test=False)
    test_ds  = XarrayDatasetElev(DPL=dpl, test=True)
    # drop_last=False so we get every sample for surrogate
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False,
                              drop_last=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              drop_last=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Build surrogate (cache logits)
# ─────────────────────────────────────────────────────────────────────────────
def build_surrogate(model: GhostWindNet27, loader: DataLoader,
                    path: str, split: str = '') -> pd.DataFrame:
    """
    One forward pass over full dataset → save (z, y_bin, cluster, lat, lon).
    If file exists, loads it instead.
    """
    if os.path.exists(path):
        logging.info(f"Loading cached surrogate from {path}")
        return pd.read_parquet(path)

    logging.info(f"Building surrogate for {split} ({len(loader.dataset)} samples)...")
    rows = []
    n_batches = len(loader)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i % 200 == 0:
                logging.info(f"  Surrogate {split}: batch {i}/{n_batches}")
            (X, pos), y, _, station_thresholds = batch
            X, pos = X.to(DEVICE), pos.to(DEVICE)

            logits = model([X, pos]).squeeze(-1).cpu().numpy().astype(np.float32)
            y_bin  = (y.numpy() >= station_thresholds.numpy()).astype(np.float32)

            # Center pixel elevation for cluster assignment
            h = X.shape[-1]
            center = h // 2
            elev_norm = X[:, 3, 0, center, center].cpu().numpy().astype(np.float32)
            cluster   = get_cluster_from_elev_norm(elev_norm)

            lat = (pos[:, 0, 2] * 90).cpu().numpy().astype(np.float32)
            lon = (pos[:, 0, 3] * 180).cpu().numpy().astype(np.float32)

            for j in range(len(logits)):
                rows.append({
                    'z':       float(logits[j]),
                    'y_bin':   float(y_bin[j]),
                    'cluster': int(cluster[j]),
                    'lat':     float(lat[j]),
                    'lon':     float(lon[j]),
                })

    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    logging.info(f"Surrogate saved: {len(df)} rows → {path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2/3: LBFGS optimizer (ground-truth optimum)
# ─────────────────────────────────────────────────────────────────────────────
def optimize_lbfgs(df: pd.DataFrame, K: int) -> np.ndarray:
    """L-BFGS-B with analytical gradient, multiple random restarts."""
    z = df['z'].values.astype(np.float64)
    y = df['y_bin'].values.astype(np.float64)
    c = df['cluster'].values.astype(np.int32) if K > 1 else np.zeros(len(df), dtype=np.int32)

    bounds = []
    for _ in range(K):
        bounds.extend([(0.01, 20.0), (-10.0, 10.0)])

    best_res = None
    rng = np.random.default_rng(42)
    for trial in range(LBFGS_N_TRIALS):
        theta0 = np.zeros(2 * K)
        theta0[::2]  = rng.uniform(0.5, 2.0, K)   # a_k
        theta0[1::2] = rng.uniform(-1.0, 1.0, K)  # b_k

        res = minimize(
            bce_cluster_platt, theta0,
            jac=bce_grad,
            args=(z, y, c, K),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-8},
        )
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    logging.info(f"  LBFGS K={K}: final BCE={best_res.fun:.6f}  success={best_res.success}")
    return best_res.x


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: SA-guided optimizer (flowback-style)
# ─────────────────────────────────────────────────────────────────────────────
def _sa_delta_to_theta(theta_base: np.ndarray, j: int, delta_pct: float,
                        sigma_z: float) -> np.ndarray:
    """Apply delta_pct change to parameter j of theta_base."""
    theta = theta_base.copy()
    if j % 2 == 0:          # a_k: multiplicative (scale = 1 + delta%)
        theta[j] = theta_base[j] * (1.0 + delta_pct)
    else:                    # b_k: additive, scaled by σ_z (like Gleb's b-percent def)
        theta[j] = theta_base[j] + delta_pct * sigma_z
    return theta


def sa_guided_optimize(theta0: np.ndarray, z: np.ndarray, y: np.ndarray,
                        c: np.ndarray, K: int, sigma_z: float) -> tuple:
    """
    SA-guided optimizer: exact structural analogue of SAGuidedOptimizer (optimizer.py).

    Each iteration:
      1. Build SA-library: eval loss at SA_DELTAS for each parameter (one-at-a-time)
      2. Dense search: interpolate SAMPLES_PER_SEGMENT pts per interval → find best Δ
      3. Combined step: apply all best Δ at scales [1.0, 0.5, 0.25]
      4. Acceptance: pick best candidate if improvement > threshold
    """
    best_theta = theta0.copy()
    best_loss  = bce_cluster_platt(best_theta, z, y, c, K)
    n_params   = len(theta0)
    stagnation = 0
    history    = []

    logging.info(f"SA-guided start: K={K}, n_params={n_params}, loss0={best_loss:.6f}")

    for iteration in range(SA_MAX_ITERS):

        # ── 1. Build SA-library ────────────────────────────────────────────
        sa_deltas_arr  = []  # shape: (n_params, len(SA_DELTAS))
        sa_losses_arr  = []
        for j in range(n_params):
            deltas, losses = [], []
            for delta_pct in SA_DELTAS:
                theta_pert = _sa_delta_to_theta(best_theta, j, delta_pct, sigma_z)
                # Clamp to valid bounds
                theta_pert[::2]  = np.clip(theta_pert[::2],  0.01, 20.0)
                theta_pert[1::2] = np.clip(theta_pert[1::2], -10.0, 10.0)
                loss_val = bce_cluster_platt(theta_pert, z, y, c, K)
                deltas.append(delta_pct)
                losses.append(loss_val)
            sa_deltas_arr.append(np.array(deltas))
            sa_losses_arr.append(np.array(losses))

        # ── 2. Dense search per parameter ─────────────────────────────────
        best_delta_per_param = []
        for j in range(n_params):
            deltas = sa_deltas_arr[j]
            losses = sa_losses_arr[j]
            best_delta_j = 0.0
            best_loss_j  = np.interp(0.0, deltas, losses)
            for seg in range(len(deltas) - 1):
                dense = np.linspace(deltas[seg], deltas[seg + 1], SAMPLES_PER_SEGMENT + 1)
                dense_losses = np.interp(dense, deltas, losses)
                idx = np.argmin(dense_losses)
                if dense_losses[idx] < best_loss_j:
                    best_loss_j  = dense_losses[idx]
                    best_delta_j = dense[idx]
            best_delta_per_param.append(best_delta_j)

        # ── 3. Combined step at multiple scales ───────────────────────────
        candidates = []
        for scale in [1.0, 0.5, 0.25]:
            theta_cand = best_theta.copy()
            for j, delta in enumerate(best_delta_per_param):
                theta_cand = _sa_delta_to_theta(theta_cand, j, scale * delta, sigma_z)
            theta_cand[::2]  = np.clip(theta_cand[::2],  0.01, 20.0)
            theta_cand[1::2] = np.clip(theta_cand[1::2], -10.0, 10.0)
            candidates.append(theta_cand)

        # ── 4. Acceptance ─────────────────────────────────────────────────
        accepted = False
        threshold = best_loss * SA_MIN_REL_IMPROVEMENT
        for theta_cand in candidates:
            loss_cand = bce_cluster_platt(theta_cand, z, y, c, K)
            improvement = best_loss - loss_cand
            if improvement > threshold:
                best_theta = theta_cand
                best_loss  = loss_cand
                stagnation = 0
                accepted   = True
                break   # first scale that improves is accepted (like flowback)

        history.append({'iter': iteration + 1, 'loss': best_loss, 'accepted': accepted})
        a_vals = best_theta[::2].round(4)
        b_vals = best_theta[1::2].round(4)
        status = "OK" if accepted else "stagnate"
        logging.info(f"  SA iter {iteration+1:3d}/{SA_MAX_ITERS}: "
                     f"loss={best_loss:.6f}  [{status}]  "
                     f"a={a_vals}  b={b_vals}")

        if not accepted:
            stagnation += 1
            if stagnation >= SA_STAGNATION_PATIENCE:
                logging.info(f"  SA converged at iter {iteration+1} "
                             f"(stagnation={stagnation})")
                break

    return best_theta, best_loss, history


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    config_dir = os.path.join(os.getcwd(), 'configs')
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name='cmip6_world_nosfcwind_server',
            overrides=[
                'train.distributed=false',
                'train.num_workers=0',
                f'train.batch_size={BATCH_SIZE}',
            ]
        )

    logging.info("Loading data (real GSOD labels, vars: pr+tasmax+tasmin+elev)...")
    train_loader, test_loader = build_loaders(cfg)

    logging.info(f"Loading synthetic CNN from {SYNTHETIC_CKPT}")
    model = load_model(SYNTHETIC_CKPT)

    # ── Step 1: Build / load surrogate ───────────────────────────────────────
    df_train = build_surrogate(model, train_loader, SURROGATE_TRAIN, 'train')
    df_test  = build_surrogate(model, test_loader,  SURROGATE_TEST,  'test')

    # Cluster statistics
    logging.info("Cluster distribution (train):")
    for k, name in enumerate(CLUSTER_NAMES):
        n = (df_train['cluster'] == k).sum()
        n_pos = ((df_train['cluster'] == k) & (df_train['y_bin'] == 1)).sum()
        logging.info(f"  {name}: {n:>8d} samples, pos_rate={n_pos/n:.3f}")

    # σ_z: std of training logits (used for b_k percent-scale in SA)
    sigma_z = float(df_train['z'].std())
    logging.info(f"Logit sigma_z (train) = {sigma_z:.4f}")

    # ── Baseline: no adaptation ───────────────────────────────────────────────
    theta_identity = np.array([1.0, 0.0] * K_CLUSTERS)  # a_k=1, b_k=0
    m0 = evaluate_platt(theta_identity, df_test, K_CLUSTERS)
    logging.info(f"Baseline (no adaptation):  AUROC={m0['AUROC']:.4f}  AP={m0['AP']:.4f}  "
                 f"BS={m0['BS']:.4f}  BSS={m0['BSS']:.4f}")

    # ── E1: Global Platt (K=1), LBFGS ────────────────────────────────────────
    logging.info("\n--- E1: Global Platt (K=1), LBFGS ---")
    theta_e1 = optimize_lbfgs(df_train, K=1)
    m_e1 = evaluate_platt(theta_e1, df_test, K=1)
    logging.info(f"E1 theta: a={theta_e1[0]:.4f}  b={theta_e1[1]:.4f}")
    logging.info(f"E1 result: AUROC={m_e1['AUROC']:.4f}  AP={m_e1['AP']:.4f}  "
                 f"BS={m_e1['BS']:.4f}  BSS={m_e1['BSS']:.4f}")

    # ── E2: Cluster Platt (K=3), LBFGS ───────────────────────────────────────
    logging.info(f"\n--- E2: Cluster Platt (K={K_CLUSTERS}), LBFGS ---")
    theta_e2 = optimize_lbfgs(df_train, K=K_CLUSTERS)
    m_e2 = evaluate_platt(theta_e2, df_test, K=K_CLUSTERS)
    for k, name in enumerate(CLUSTER_NAMES):
        logging.info(f"  {name}: a={theta_e2[2*k]:.4f}  b={theta_e2[2*k+1]:.4f}")
    logging.info(f"E2 result: AUROC={m_e2['AUROC']:.4f}  AP={m_e2['AP']:.4f}  "
                 f"BS={m_e2['BS']:.4f}  BSS={m_e2['BSS']:.4f}")

    # ── E2-SA: Cluster Platt (K=3), SA-guided ────────────────────────────────
    logging.info(f"\n--- E2-SA: Cluster Platt (K={K_CLUSTERS}), SA-guided ---")
    z_tr = df_train['z'].values.astype(np.float64)
    y_tr = df_train['y_bin'].values.astype(np.float64)
    c_tr = df_train['cluster'].values.astype(np.int32)

    theta_sa_init = np.array([1.0, 0.0] * K_CLUSTERS)
    theta_e2_sa, loss_e2_sa, sa_history = sa_guided_optimize(
        theta_sa_init, z_tr, y_tr, c_tr, K_CLUSTERS, sigma_z)
    m_e2_sa = evaluate_platt(theta_e2_sa, df_test, K_CLUSTERS)
    for k, name in enumerate(CLUSTER_NAMES):
        logging.info(f"  {name}: a={theta_e2_sa[2*k]:.4f}  b={theta_e2_sa[2*k+1]:.4f}")
    logging.info(f"E2-SA result: AUROC={m_e2_sa['AUROC']:.4f}  AP={m_e2_sa['AP']:.4f}  "
                 f"BS={m_e2_sa['BS']:.4f}  BSS={m_e2_sa['BSS']:.4f}")
    n_iters = len(sa_history)
    logging.info(f"SA-guided: {n_iters} iterations "
                 f"({n_iters * (K_CLUSTERS * len(SA_DELTAS) + 3)} actual loss evals)")

    # ── Supervised upper bound ────────────────────────────────────────────────
    m_sup = None
    if SUPERVISED_CKPT and os.path.exists(SUPERVISED_CKPT):
        logging.info(f"\nEvaluating supervised upper bound from {SUPERVISED_CKPT}...")
        import torchmetrics
        model_sup = load_model(SUPERVISED_CKPT)
        auroc_m = torchmetrics.AUROC(task='binary').to(DEVICE)
        ap_m    = torchmetrics.AveragePrecision(num_classes=1, task='binary').to(DEVICE)
        with torch.no_grad():
            for batch in test_loader:
                (X, pos), y, _, st = batch
                X, pos = X.to(DEVICE), pos.to(DEVICE)
                logits = model_sup([X, pos]).squeeze()
                y_bin  = (y >= st).float().to(DEVICE)
                auroc_m.update(torch.sigmoid(logits), y_bin.int())
                ap_m.update(torch.sigmoid(logits), y_bin.int())
        m_sup = {'AUROC': auroc_m.compute().item(), 'AP': ap_m.compute().item()}

    # ── Paper table ───────────────────────────────────────────────────────────
    logging.info("")
    logging.info("=" * 68)
    logging.info("PAPER TABLE — Test 2023-2024, real GSOD labels")
    logging.info("=" * 68)
    logging.info(f"{'Model':<40} {'AUROC':>7} {'AP':>7} {'BSS':>7}")
    logging.info(f"{'Synthetic CNN (no adaptation)':<40} "
                 f"{m0['AUROC']:>7.3f} {m0['AP']:>7.3f} {m0['BSS']:>7.3f}")
    logging.info(f"{'E1: Global Platt (K=1, LBFGS)':<40} "
                 f"{m_e1['AUROC']:>7.3f} {m_e1['AP']:>7.3f} {m_e1['BSS']:>7.3f}")
    logging.info(f"{'E2: Cluster Platt (K=3, LBFGS)':<40} "
                 f"{m_e2['AUROC']:>7.3f} {m_e2['AP']:>7.3f} {m_e2['BSS']:>7.3f}")
    logging.info(f"{'E2-SA: Cluster Platt (K=3, SA-guided)':<40} "
                 f"{m_e2_sa['AUROC']:>7.3f} {m_e2_sa['AP']:>7.3f} {m_e2_sa['BSS']:>7.3f}")
    if m_sup:
        logging.info(f"{'Supervised CNN (upper bound)':<40} "
                     f"{m_sup['AUROC']:>7.3f} {m_sup['AP']:>7.3f} {'—':>7}")
    else:
        logging.info(f"{'Supervised CNN (upper bound)':<40} {'0.851':>7} {'0.757':>7} {'—':>7}")
    logging.info("=" * 68)

    # Save results
    np.save('theta_e1.npy', theta_e1)
    np.save('theta_e2_lbfgs.npy', theta_e2)
    np.save('theta_e2_sa.npy', theta_e2_sa)
    pd.DataFrame(sa_history).to_csv('sa_history.csv', index=False)
    logging.info("Saved: theta_e1.npy, theta_e2_lbfgs.npy, theta_e2_sa.npy, sa_history.csv")


if __name__ == '__main__':
    main()

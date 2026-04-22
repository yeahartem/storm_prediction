"""
Sensitivity analysis + adaptation of synthetic CNN to real GSOD labels.

Algorithm (Gleb's sensitivity-guided adaptation):
  1. Load synthetic CNN (trained on CMIP6 sfcWindmax pseudo-labels)
  2. Sensitivity: compute |dp/d_alpha_i| for each input channel at alpha=[1,1,1,1]
  3. Adaptation: gradient descent on alpha to minimise BCE vs real GSOD labels
  4. Evaluate and print paper comparison table

Alpha interpretation: X_adapted[i] = alpha[i] * X_original[i]
  alpha > 1: CMIP6 underestimates this variable relative to what the model needs
  alpha < 1: CMIP6 overestimates
"""
import sys, os
sys.path.append(os.getcwd())

import torch
import torch.nn.functional as F
import numpy as np
import logging
import torchmetrics
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.regression.data_load import DataPreLoader
from src.regression.datamodule import XarrayDatasetElev
from src.regression.models.models import GhostWindNet27

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# ── paths ────────────────────────────────────────────────────────────────────
SYNTHETIC_CKPT   = 'out/2026-04-21/23-47-36/epoch=22-step=115000.ckpt'
SUPERVISED_CKPT  = 'out/2026-04-20/15-54-18/epoch=22-step=115000.ckpt'
CHANNEL_NAMES    = ['pr', 'tasmax', 'tasmin', 'elevation']

# ── hyper-params ─────────────────────────────────────────────────────────────
DEVICE                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE              = 256
SENSITIVITY_BATCHES     = 100   # batches used to estimate sensitivity
ADAPT_LR                = 0.05
ADAPT_EPOCHS            = 40
ADAPT_BATCHES_PER_EPOCH = 500   # limit per epoch for speed (~3 min/epoch on server)
ALPHA_MIN, ALPHA_MAX    = 0.1, 5.0


# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str, in_chans: int = 4) -> GhostWindNet27:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = {k[len('net.'):]: v
                  for k, v in ckpt['state_dict'].items()
                  if k.startswith('net.')}
    model = GhostWindNet27(in_chans=in_chans)
    model.load_state_dict(state_dict)
    model.requires_grad_(False)   # freeze: only alpha trains
    return model.to(DEVICE).eval()


def build_loaders(cfg):
    dpl = DataPreLoader(cfg)
    train_ds = XarrayDatasetElev(DPL=dpl, test=False)
    test_ds  = XarrayDatasetElev(DPL=dpl, test=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              drop_last=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              drop_last=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader


def scaled_forward(model, X, pos, alpha):
    """Forward pass with per-channel alpha scaling."""
    X_scaled = X * alpha.view(1, -1, 1, 1, 1)
    return model([X_scaled, pos]).squeeze()


# ─────────────────────────────────────────────────────────────────────────────
def sensitivity_analysis(model, train_loader, alpha: torch.Tensor) -> torch.Tensor:
    """
    Finite-difference sensitivity: S_i = E[|p(alpha_i + eps) - p(alpha_i)| / eps]
    Returns tensor of shape (n_channels,).
    """
    eps = 0.1
    S = torch.zeros(len(CHANNEL_NAMES), device=DEVICE)
    n = 0
    for batch in train_loader:
        if n >= SENSITIVITY_BATCHES:
            break
        (X, pos), y, _, _ = batch
        X, pos = X.to(DEVICE), pos.to(DEVICE)
        with torch.no_grad():
            p_base = torch.sigmoid(scaled_forward(model, X, pos, alpha))
            for i in range(len(CHANNEL_NAMES)):
                a_pert = alpha.clone()
                a_pert[i] += eps
                p_pert = torch.sigmoid(scaled_forward(model, X, pos, a_pert))
                S[i] += (p_pert - p_base).abs().mean()
        n += 1
    S /= n
    return S


# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, alpha: torch.Tensor) -> dict:
    """Evaluate on real GSOD labels: AUROC, AP, BCE."""
    auroc_m = torchmetrics.AUROC(task='binary').to(DEVICE)
    ap_m    = torchmetrics.AveragePrecision(num_classes=1, task='binary').to(DEVICE)
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            (X, pos), y, _, station_thresholds = batch
            X, pos = X.to(DEVICE), pos.to(DEVICE)
            y_bin  = (y >= station_thresholds).float().to(DEVICE)
            logits = scaled_forward(model, X, pos, alpha)
            total_loss += F.binary_cross_entropy_with_logits(logits, y_bin).item()
            probs = torch.sigmoid(logits)
            auroc_m.update(probs, y_bin.int())
            ap_m.update(probs, y_bin.int())
            n += 1
    return {'BCE': total_loss / n,
            'AUROC': auroc_m.compute().item(),
            'AP':    ap_m.compute().item()}


# ─────────────────────────────────────────────────────────────────────────────
def adapt(model, train_loader) -> torch.Tensor:
    """
    Gradient descent on alpha to minimise real-label BCE.
    Returns optimised alpha tensor.
    """
    alpha = torch.nn.Parameter(torch.ones(len(CHANNEL_NAMES), device=DEVICE))
    optimizer = torch.optim.Adam([alpha], lr=ADAPT_LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ADAPT_EPOCHS)

    for epoch in range(ADAPT_EPOCHS):
        epoch_loss, n = 0.0, 0
        for batch in train_loader:
            if n >= ADAPT_BATCHES_PER_EPOCH:
                break
            (X, pos), y, _, station_thresholds = batch
            X, pos = X.to(DEVICE), pos.to(DEVICE)
            y_bin  = (y >= station_thresholds).float().to(DEVICE)

            optimizer.zero_grad()
            logits = scaled_forward(model, X, pos, alpha)
            loss   = F.binary_cross_entropy_with_logits(logits, y_bin)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                alpha.clamp_(ALPHA_MIN, ALPHA_MAX)

            epoch_loss += loss.item()
            n += 1

        scheduler.step()
        a = alpha.detach().cpu().numpy().round(3)
        logging.info(f"Adapt epoch {epoch+1:3d}/{ADAPT_EPOCHS}: "
                     f"loss={epoch_loss/n:.4f}  "
                     f"alpha=[pr={a[0]}, tasmax={a[1]}, tasmin={a[2]}, elev={a[3]}]")

    return alpha.detach()


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

    logging.info("Loading data (supervised — real GSOD labels, vars: pr+tasmax+tasmin+elev)")
    train_loader, test_loader = build_loaders(cfg)

    logging.info(f"Loading synthetic CNN from {SYNTHETIC_CKPT}")
    model = load_model(SYNTHETIC_CKPT)

    alpha_init = torch.ones(len(CHANNEL_NAMES), device=DEVICE)

    # ── 1. Sensitivity analysis ───────────────────────────────────────────────
    logging.info("Computing sensitivity (finite differences, alpha=[1,1,1,1])...")
    S = sensitivity_analysis(model, train_loader, alpha_init)
    logging.info("Sensitivity |dp/d_alpha_i|:")
    for name, s in zip(CHANNEL_NAMES, S):
        logging.info(f"  {name:12s}: {s.item():.5f}")
    np.save('sensitivity_library.npy', S.cpu().numpy())

    # ── 2. Baseline: synthetic model on real labels ───────────────────────────
    logging.info("Evaluating synthetic CNN baseline (alpha=[1,1,1,1]) on real GSOD test...")
    m_base = evaluate(model, test_loader, alpha_init)
    logging.info(f"Baseline  AUROC={m_base['AUROC']:.4f}  AP={m_base['AP']:.4f}  BCE={m_base['BCE']:.4f}")

    # ── 3. Adaptation ─────────────────────────────────────────────────────────
    logging.info("Starting adaptation (learning alpha on real GSOD train 2000-2022)...")
    alpha_adapted = adapt(model, train_loader)
    np.save('adapted_alpha.npy', alpha_adapted.cpu().numpy())

    # ── 4. Evaluate adapted model ─────────────────────────────────────────────
    logging.info("Evaluating adapted CNN on real GSOD test 2023-2024...")
    m_adapt = evaluate(model, test_loader, alpha_adapted)
    logging.info(f"Adapted   AUROC={m_adapt['AUROC']:.4f}  AP={m_adapt['AP']:.4f}  BCE={m_adapt['BCE']:.4f}")

    a = alpha_adapted.cpu().numpy().round(3)
    logging.info(f"Final alpha: pr={a[0]}  tasmax={a[1]}  tasmin={a[2]}  elev={a[3]}")

    # ── 5. Load supervised upper bound ────────────────────────────────────────
    m_sup = None
    if SUPERVISED_CKPT and os.path.exists(SUPERVISED_CKPT):
        logging.info(f"Evaluating supervised upper bound from {SUPERVISED_CKPT}...")
        model_sup = load_model(SUPERVISED_CKPT)
        alpha_one = torch.ones(len(CHANNEL_NAMES), device=DEVICE)
        m_sup = evaluate(model_sup, test_loader, alpha_one)

    # ── 6. Paper summary table ────────────────────────────────────────────────
    logging.info("")
    logging.info("=" * 60)
    logging.info("PAPER TABLE — Test 2023-2024, real GSOD labels")
    logging.info("=" * 60)
    logging.info(f"{'Model':<35} {'AUROC':>7} {'AP':>7}")
    logging.info(f"{'CMIP6 threshold baseline':<35} {'—':>7} {'—':>7}  (compute separately)")
    logging.info(f"{'Synthetic CNN (no adaptation)':<35} {m_base['AUROC']:>7.3f} {m_base['AP']:>7.3f}")
    logging.info(f"{'Adapted synthetic CNN':<35} {m_adapt['AUROC']:>7.3f} {m_adapt['AP']:>7.3f}")
    if m_sup:
        logging.info(f"{'Supervised CNN (upper bound)':<35} {m_sup['AUROC']:>7.3f} {m_sup['AP']:>7.3f}")
    else:
        logging.info(f"{'Supervised CNN (upper bound)':<35} {'0.851':>7} {'0.757':>7}  (from prior run)")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
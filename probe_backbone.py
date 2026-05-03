"""
Probe: does the backbone actually matter?

Test 1 - X=zeros: pos removed from head, so output must be constant -> AUROC=0.50
  If AUROC > 0.55 -> pos leakage STILL present somewhere!
  If AUROC ~ 0.50 -> pos leakage confirmed FIXED.

Test 2 - X shuffled across ALL test samples: breaks station+time assignment.
  If AUROC drops a lot vs real -> model genuinely uses CMIP6 content per sample.
  If AUROC stays near real -> model does spatial/geographic lookup only (bad for paper).

Usage:
  CKPT_PATH="out/..." CUDA_VISIBLE_DEVICES=0 python probe_backbone.py \
    --config-name=cmip6_world [train=...]
"""
import sys, os
sys.path.append(os.getcwd())
import torch
_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'weights_only': False})

import numpy as np
from sklearn.metrics import roc_auc_score
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

ckpt_path = os.environ.get('CKPT_PATH')
assert ckpt_path, 'Set CKPT_PATH env var'
logging.info(f'Loading checkpoint: {ckpt_path}')

ckpt = torch.load(ckpt_path, map_location='cpu')
cfg = ckpt['hyper_parameters']['cfg']

from src.regression.datamodule import WindDataModule
from src.regression.models.pl_module import WindNetPL

dm = WindDataModule(cfg)
dm.setup('test')

model = WindNetPL.load_from_checkpoint(ckpt_path, map_location='cuda', strict=False)
model.eval()
logging.info('Model loaded. Running inference...')

all_X, scores_real, labels = [], [], []

with torch.no_grad():
    for batch in dm.test_dataloader():
        objs, target, _, thresholds = batch
        X = objs[0].cuda()
        pos = objs[1].cuda()

        s_real = torch.sigmoid(model.net([X, pos]).squeeze()).cpu()
        scores_real.append(s_real)
        labels.append((target >= thresholds).int())
        all_X.append(X.cpu())   # collect for global shuffle

y = torch.cat(labels).numpy()
s_real_np = torch.cat(scores_real).numpy()
auroc_real = roc_auc_score(y, s_real_np)
logging.info(f'AUROC (real X):        {auroc_real:.4f}  <- baseline')

# Test 1: X = zeros
logging.info('Running X=zeros test...')
scores_zero = []
with torch.no_grad():
    for batch in dm.test_dataloader():
        objs, _, _, _ = batch
        X = torch.zeros_like(objs[0]).cuda()
        pos = objs[1].cuda()
        scores_zero.append(torch.sigmoid(model.net([X, pos]).squeeze()).cpu())

auroc_zero = roc_auc_score(y, torch.cat(scores_zero).numpy())
logging.info(f'AUROC (X=zeros):       {auroc_zero:.4f}  <- expected ~0.50 if no leakage')

# Test 2: X globally shuffled (breaks station+time assignment)
logging.info('Running X=globally-shuffled test...')
all_X_cat = torch.cat(all_X, dim=0)  # (N, C, 27, 95, 95)
perm = torch.randperm(all_X_cat.shape[0])
all_X_shuffled = all_X_cat[perm]

scores_shuf = []
bs = cfg.train.batch_size
with torch.no_grad():
    for i in range(0, all_X_shuffled.shape[0], bs):
        X_shuf = all_X_shuffled[i:i+bs].cuda()
        # pos doesn't matter (not in head), use zeros as placeholder
        pos_dummy = torch.zeros(X_shuf.shape[0], model.net.time_window, 4, device='cuda')
        scores_shuf.append(torch.sigmoid(model.net([X_shuf, pos_dummy]).squeeze()).cpu())

auroc_shuf = roc_auc_score(y, torch.cat(scores_shuf).numpy())
logging.info(f'AUROC (X=shuffled):    {auroc_shuf:.4f}  <- if >> 0.5: backbone does geographic lookup')

logging.info('--- SUMMARY ---')
logging.info(f'Real:     {auroc_real:.4f}')
logging.info(f'X=zeros:  {auroc_zero:.4f}  (should be ~0.50)')
logging.info(f'Shuffled: {auroc_shuf:.4f}')
logging.info(f'Backbone contribution (real - shuffled): {auroc_real - auroc_shuf:.4f}')

if auroc_zero > 0.55:
    logging.info('PROBLEM: pos leakage still detected!')
elif auroc_shuf > auroc_real - 0.05:
    logging.info('FINDING: model does geographic/static lookup (backbone sees same X per station ~always)')
else:
    logging.info('FINDING: model uses CMIP6 content per sample (genuine temporal signal)')

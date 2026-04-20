---
name: Full model test results (2026-04-16)
description: Best test results for full model (sfcWindmax+pr+tasmax+tasmin+elevation), epoch 15
type: project
---
Checkpoint: out/2026-04-15/22-37-45/epoch=15-step=80000.ckpt
Config: undersampling 1:1, label_smoothing=0.05, weight_decay=0.001, gradient_clip=1.0, time_freq=14, 1 station/cell, CosineAnnealingLR, per-epoch resampling, psl variable added, 2xGPU

**Why:** These are the best results to date, to be used in Table 2 of the paper.
**How to apply:** Use as the "CNN (full)" row in Table 2. Still need BSS for noelev/nopr/notemp ablations.

## Global test metrics
- AUROC: 0.8474
- AP: 0.7499
- Precision: 0.6544
- Recall: 0.7637
- BSS: 0.3208 (clim_rate=0.357)

## Seasonal
- DJF: AUROC=0.872, AP=0.799 (best)
- MAM: AUROC=0.858, AP=0.786
- SON: AUROC=0.835, AP=0.712
- JJA: AUROC=0.818, AP=0.678 (worst)

## Regional highlights
- Best: russia_west_siberia AUROC=0.863
- Worst: russia_east_siberia AP=0.259 (pos_rate only 4.4%)
- africa_sahel_east weakest: AUROC=0.778

## Confusion matrix
TN=63622  FP=18374  FN=10770  TP=34786

## Previous best (epoch=12, 2026-04-13) for reference
AUROC=0.8470, AP=0.7479, BSS=0.3184

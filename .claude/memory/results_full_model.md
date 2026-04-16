---
name: Full model test results (2026-04-13)
description: Best test results for full model (sfcWindmax+pr+tasmax+tasmin+elevation), epoch 12
type: project
originSessionId: 9f89d0a6-a45d-44b1-b9b1-461b7890b7bf
---
Checkpoint: out/2026-04-12/21-12-58/epoch=12-step=65000.ckpt
Config: undersampling 1:1, label_smoothing=0.05, weight_decay=0.001, gradient_clip=1.0, time_freq=14, 1 station/cell

**Why:** These are the best results to date, to be used in Table 2 of the paper.
**How to apply:** Use as the "CNN (full)" row in Table 2. Still need BSS for noelev/nopr/notemp ablations.

## Global test metrics
- AUROC: 0.8470
- AP: 0.7479
- Precision: 0.6538
- Recall: 0.7604
- Brier Score: 0.1564
- BSS: 0.3184 (clim_rate=0.357)

## Seasonal
- DJF: AUROC=0.871, AP=0.793 (best)
- MAM: AUROC=0.859, AP=0.786
- SON: AUROC=0.836, AP=0.715
- JJA: AUROC=0.817, AP=0.672 (worst)

## Regional highlights
- Best: russia_west_siberia AUROC=0.865
- Worst: russia_east_siberia AP=0.342 (pos_rate only 4.6%)
- africa_sahel_east weakest: AUROC=0.744

## Confusion matrix
TN=63727  FP=18352  FN=10931  TP=34606

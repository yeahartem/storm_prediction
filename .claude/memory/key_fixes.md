---
name: Key fixes applied
description: Critical bugs found and fixed during CMIP6 pipeline setup
type: feedback
---

**1. BCELoss metric computation was wrong** (`float_to_score` on logits)
`float_to_score(logit, thresh=10) = logit/20` clipped [0,0.999] — destroys ranking for negative logits.
Fixed: added `_score()` and `_binary_pred()` helper methods in `WindNetPL` that use `torch.sigmoid()` for BCELoss.
**Why:** Logits from BCEWithLogitsLoss can be negative; clipping to 0 ruins AP/precision/recall metrics.

**2. Normalization data leakage**
Was computing mean/std over full dataset including test. Fixed to use only train portion via `sel(time=slice(None, split_date))`.
**Why:** Test data distribution leaked into normalization, inflating metrics.

**3. Windows cp1251 encoding crash**
Emoji characters (🔥📈🚨) in print statements caused `UnicodeEncodeError` when wandb/hydra captured stdout.
Also: polars DataFrame pretty-print uses Unicode box-drawing characters — also crashes.
Fixed: removed all emojis from `models.py` and `pl_module.py`; replaced `print(df)` with `logging.info(f'shape: {df.shape}')` in `assemble_target.py`.
**Why:** Windows with Russian locale uses cp1251 which doesn't support these Unicode characters.

**4. RMSE plot shape mismatch**
`thrs` (13 elements) vs `rmses` (variable, skips thresholds with no samples). Fixed by tracking `thrs_used` alongside `rmses`.
**Why:** Val/test subsets may not contain samples for all wind speed thresholds.

**5. torch.load weights_only=True (PyTorch 2.6+)**
PL 2.0.2 checkpoint loading fails because OmegaConf DictConfig is not a safe global.
Fixed: monkey-patched `torch.load` in `train.py` to default `weights_only=False`.
**Why:** PyTorch 2.6+ changed the default for security, breaking old PL checkpoint format.

**6. GhostNetV2 in_chans=3 default**
`timm.create_model('ghostnetv2_160')` defaults to 3 input channels (RGB) but we have 4 climate variables.
Fixed: added `in_chans=4` parameter.

**7. Double MXWDSP/WDSP conversion in combined parquet (ROOT CAUSE of train/test imbalance)**
`world_stations_2000_2025_25_days_6_months.parquet` was created by combining:
- old parquet (already in m/s from preprocess_csv) with new 2021-2025 data
- but ×0.51444 was applied again to old data during merge
Result: 2000-2020 rows had MXWDSP ≈ 3.7 m/s (half the real value), >10 m/s = 1.7% instead of 22%.
Fixed: divided pre-2021 MXWDSP and WDSP by 0.51444. target.parquet rebuilt.
**Why:** Led to 49.7% positive rate in test vs 19.8% in train — appeared as severe class imbalance bug.

**8. TEMP in Fahrenheit for 2021+ GSOD data**
New GSOD download (2021-2024) had TEMP not converted from Fahrenheit (mean 56°F).
Old data (2000-2020) was already in Celsius from preprocess_csv.
Fixed: applied (T-32)*5/9 to TEMP for DATE >= 2021-01-01 in the combined parquet.
**Why:** TEMP is stored in target cleaned parquet; inconsistent units are confusing even if not used as model feature.

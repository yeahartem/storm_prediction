---
name: Project state
description: Current state of storm prediction CMIP6 pipeline — what's done, what's pending
type: project
---

Data bug fixed and target.parquet rebuilt on 2026-04-03.

**Data:**
- CMIP6 MRI-ESM2-0, historical+ssp245, 4 variables: sfcWindmax, pr, tasmax, tasmin
- Preprocessed to `data/cmip6_world/` as float16 normalized npy files
- Station data: `world_stations_2000_2025_25_days_6_months.parquet`, ~78M rows, MXWDSP in m/s
- Train 2000-2020, Val 2021-2022, Test 2023-2024 (split at `start_of_test: 2023-01-01`)
- Normalization: train-only stats (no leakage)

**Data bug fixed (2026-04-03):**
- Pre-2021 MXWDSP/WDSP were double-converted (×0.51444 applied twice). Fixed: divided by 0.51444.
- 2021+ TEMP was in Fahrenheit. Fixed: converted to Celsius.
- Backup saved as `world_stations_2000_2025_25_days_6_months.parquet.bak`
- target.parquet rebuilt. Correct positive rates: train=24.5%, val=12.7%, test=12.1%

**Model:** GhostWindNet27, 11.3M params, BCELoss with pos_weight computed dynamically (~3.08 after fix)

**Positive rates after fix:**
- train: 24.5% (2000-2020 GSOD, denser coverage)
- val: 12.7% (2021-2022 new GSOD)
- test: 12.1% (2023-2024 new GSOD)
- Train/test difference is real (different station sets), not a bug

**Ready for full training:** `python train.py --config-name=cmip6_world` (max_epoch=100)

**Why:** Building a scientific paper on wind/storm prediction with a global ML model.
**How to apply:** Full 100-epoch training is the next step. Data pipeline is solid.

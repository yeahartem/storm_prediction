# Storm Prediction Project

## Goal
Global wind/storm prediction model for a scientific paper. Binary classification: will wind exceed threshold at a station in the next 25 days over a 6-month season?

## Environment
- Python: `C:\Users\Amd\miniconda3\envs\wind\python.exe` (Python 3.10, conda env `wind`)
- GPU: NVIDIA RTX 3070 8GB, CUDA 12.7
- **Run training:** `C:/Users/Amd/miniconda3/envs/wind/python.exe train.py --config-name=cmip6_world`
- conda is NOT on PATH in bash — always use the full python path above
- Miniconda installed at: `C:\Users\Amd\miniconda3`

## Package constraints (do not change)
- torch 2.11.0+cu126 — must be CUDA build, not CPU
- numpy < 2.0 (1.26.4) — numpy 2.x breaks wandb and scipy
- setuptools < 70 (69.5.1) — pkg_resources compatibility with PL 2.0.2
- pytorch-lightning 2.0.2

## Project structure
```
train.py                          # entry point (Hydra)
configs/cmip6_world.yaml          # main training config
src/
  regression/
    models/models.py              # GhostWindNet27 (11.3M params, GhostNetV2 backbone, in_chans=4)
    models/pl_module.py           # WindNetPL — Lightning module, BCEWithLogitsLoss
    datamodule.py                 # DataModule — loads CMIP6 + station targets
    data_load.py                  # dataset classes
    eval.py                       # evaluation / metrics
  data_assemble/
    prepare_cmip.py               # CMIP6 preprocessing → float16 npy
    assemble_target.py            # builds target.parquet from GSOD station data
  utils/
    metrics.py                    # AP, precision@k, RMSE threshold curves
    norm_values.py                # normalization stats (train-only, no leakage)
data/
  cmip6_world/                    # preprocessed CMIP6 float16 npy files
  weatherstation_data/
    world_stations_2000_2025_25_days_6_months.parquet   # ~78M rows, MXWDSP in m/s
    world_stations_2000_2025_25_days_6_months_fixed.parquet
```

## Data
- **CMIP6:** MRI-ESM2-0, historical+ssp245, 4 variables: sfcWindmax, pr, tasmax, tasmin
- **Stations:** GSOD, ~78M rows, MXWDSP in m/s
- **Split:** Train 2000-2020 | Val 2021-2022 | Test 2023-2024 (start_of_test: 2023-01-01)
- **Positive rates (correct after bug fix):** train 24.5%, val 12.7%, test 12.1%
- **pos_weight** for BCELoss: ~3.08, computed dynamically from train set

## Known bugs fixed — do not reintroduce
1. **BCELoss metric bug** — use `torch.sigmoid()` for scoring, never `float_to_score` on logits
2. **Normalization leakage** — norm stats computed on train portion only (`slice(None, split_date)`)
3. **Windows cp1251 crash** — NO emojis (🔥📈🚨) in any print/log statement; no polars DataFrame pretty-print; use `logging.info(f'shape: {df.shape}')` instead
4. **RMSE plot shape mismatch** — track `thrs_used` alongside `rmses`, not just `thrs`
5. **torch.load weights_only** — monkey-patched in train.py to `weights_only=False` (PL 2.0.2 + PyTorch 2.6+)
6. **GhostNetV2 in_chans** — must pass `in_chans=4` to `timm.create_model`

## Data bugs fixed (2026-04-03) — history only
- Pre-2021 MXWDSP/WDSP were double-converted (×0.51444 twice). Fixed by dividing by 0.51444.
- 2021+ TEMP was in Fahrenheit. Fixed: (T-32)*5/9 applied.
- target.parquet was rebuilt after these fixes.

## Windows-specific
- Non-admin account — cannot install system-wide packages
- Use `pip install --user` or install into conda env only
- Shell: bash via Git Bash / VSCode terminal
- Locale cp1251 — avoid all non-ASCII in stdout/stderr output from Python scripts

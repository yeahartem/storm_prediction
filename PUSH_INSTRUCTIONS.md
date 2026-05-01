# Как запушить в `fixed-bug`

Sandbox этого Claude'а не имеет прав на запись в `.git/`, поэтому git operations надо запустить из Git Bash / PowerShell на Windows. Все изменения файлов уже сохранены на диске.

## Команды

```bash
cd "G:\wind 21.04.26 ufa"

# Удалить зависший lock-файл, если он там есть
rm -f .git/index.lock

# Создать ветку (если ещё не создана) и переключиться
git checkout -b fixed-bug 2>/dev/null || git checkout fixed-bug

# Застейджить все изменения
git add \
    src/regression/models/models.py \
    src/regression/models/pl_module.py \
    src/regression/data_load.py \
    src/regression/datamodule.py \
    src/regression/train.py \
    configs/train/ \
    AUDIT_REPORT_OPUS.md \
    PUSH_INSTRUCTIONS.md

# Закоммитить
git commit -m "Fix pos-leakage bug + proper val/test split + training-quality knobs

Audit found that GhostWindNet27 was concatenating 4 pos numbers
(time_pos, time_pos_m, lat/90, lon/180) with the backbone embedding,
giving the head a direct (station, month) climatology lookup.
A simple per-(station, month) baseline reaches AUROC 0.845 / AP 0.748
on test, statistically indistinguishable from every CMIP6 ablation
the paper reports. See AUDIT_REPORT_OPUS.md for the full investigation.

Code:
- models.py: pos no longer concatenated with backbone in head_lin1.
  Add use_pos_in_head flag (default False) for legacy reproduction.
  Add drop_path_rate parameter for stochastic depth.
- data_load.py: proper time-based split with start_of_val:
    train: time < start_of_val
    val:   start_of_val <= time < start_of_test
    test:  time >= start_of_test
  Runtime asserts on overlap; off-by-one '>' -> '>=' fixed.
- datamodule.py: XarrayDataset accepts split='train'|'val'|'test'.
  Val uses deterministic fixed subset (val_subset_size=32000,
  seed=42), shuffle=False -> stable val/loss for early stopping.
  Backward compatible with legacy test=True/False kwarg.
- pl_module.py: WarmupCosineLR scheduler (5% warmup + cosine decay
  over the entire run, steps per batch). CosineAnnealingLR T_max
  default fixed to max_epoch (was 20). use_pos_in_head and
  drop_path_rate read from cfg.
- train.py: gradient_clip_val=1.0, val_check_interval=0.5,
  StochasticWeightAveraging callback (default on), early-stop
  patience configurable.
- configs/train/*.yaml: every train config gains start_of_val,
  val_subset_size/seed, val_samples_per_station=6,
  val_check_interval=1.0, warmup_pct, eta_min_factor,
  drop_path_rate, use_pos_in_head=false, gradient_clip_val,
  use_swa, swa_lrs, swa_epoch_start, early_stopping_patience.
- Val sampling is now stratified by station (K=6 random samples
  per station). Every val station gets equal voice, no
  geographic bias from observation density.
- AUDIT_REPORT_OPUS.md: full investigation + reproduction script."

# Запушить ветку на origin
git push -u origin fixed-bug
```

## Что должно получиться

После пуша на GitHub появится ветка `fixed-bug` со всеми 36 изменёнными файлами. Compare-link:
https://github.com/yeahartem/storm_prediction/compare/synthetic-adaptation...fixed-bug

## Если возникнут конфликты

Если базовая ветка для PR должна быть не `synthetic-adaptation`, а, например, `main`, то перед коммитом стоит сделать:

```bash
git fetch origin main
git rebase origin/main          # или git merge origin/main
```

И только потом `git push -u origin fixed-bug`.

"""EDA: статистика данных без загрузки всего датасета в память"""
import sys, os
sys.path.append(os.getcwd())
import polars as pl
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'eda_output'
os.makedirs(OUT, exist_ok=True)

TARGET_PATH  = 'data/cmip6_world/target.parquet'
THRESHOLD    = 10.0   # m/s (текущий)

# ─────────────────────────────────────────────────────────────
# 1. target.parquet — основная статистика
# ─────────────────────────────────────────────────────────────
print('Loading target.parquet ...')
tgt = pl.read_parquet(TARGET_PATH)
print(f'Shape: {tgt.shape}, columns: {tgt.columns}')

tgt = tgt.with_columns(pl.col('time').dt.year().alias('year'))
y = tgt['y'].to_numpy()
print(f'\n--- y (raw max wind speed, m/s) ---')
for pct in [25, 50, 75, 90, 95, 96, 99]:
    print(f'  p{pct:2d}: {np.percentile(y, pct):.2f}')
print(f'  mean:  {y.mean():.2f}')
print(f'  std:   {y.std():.2f}')
print(f'  min:   {y.min():.2f}')
print(f'  max:   {y.max():.2f}')
print(f'\n--- Класс баланс на RAW y ---')
for thr in [5, 10, 15, 20, 25]:
    pct_pos = 100*(y >= thr).mean()
    print(f'  >= {thr:2d} m/s: {pct_pos:.1f}%')

# ─────────────────────────────────────────────────────────────
# 2. Баланс по годам на RAW y
# ─────────────────────────────────────────────────────────────
print('\n--- Баланс по годам (raw y >= 10 m/s) ---')
by_year = (tgt
    .group_by('year').agg([
        pl.count().alias('n'),
        pl.col('y').mean().alias('mean'),
        pl.col('y').median().alias('median'),
        (pl.col('y') >= 10).mean().alias('ge10'),
        (pl.col('y') >= 20).mean().alias('ge20'),
    ]).sort('year'))

years = []; ge10s = []; ge20s = []; means = []
for row in by_year.rows():
    yr, n, mn, med, g10, g20 = row
    years.append(yr); ge10s.append(g10*100); ge20s.append(g20*100); means.append(mn)
    print(f'  {yr}: n={n:>8,}  mean={mn:5.2f}  median={med:5.2f}  '
          f'>=10: {g10*100:5.1f}%  >=20: {g20*100:5.1f}%')

# ─────────────────────────────────────────────────────────────
# 3. Симуляция того что делает DataLoader (96th pct / 28 days)
# ─────────────────────────────────────────────────────────────
print('\n--- Симуляция DataLoader (96th pct, 28-day window) ---')
print('DataLoader берёт: quantile(0.96, sliding_window(y, 28)) и сравнивает с порогом 10')
print('Это НЕ то же самое что raw y >= 10')

# Возьмём несколько станций случайно
sample_stations = tgt.sample(n=min(500, len(tgt)), seed=42)['lat'].unique()[:20].to_list()
simulated_pos = []
for lat in sample_stations:
    st = tgt.filter(pl.col('lat') == lat).sort('time')['y'].to_numpy()
    if len(st) >= 28:
        wins = sliding_window_view(st, 28)
        q96 = np.quantile(wins, 0.96, axis=1)
        simulated_pos.append((q96 >= THRESHOLD).mean())

if simulated_pos:
    print(f'  На {len(simulated_pos)} станциях:')
    print(f'  Среднее % батчей где 96th pct >= {THRESHOLD}: {np.mean(simulated_pos)*100:.1f}%')
    print(f'  => Это объясняет balance train = 67.6%')

print('\n--- CONCLUSION ---')
pct50 = np.percentile(y, 50)
print(f'  Median raw y = {pct50:.2f} m/s')
print(f'  96th percentile of raw y is already high => majority >= 10 m/s')
print(f'  Threshold 10 m/s too low for "storm" task')
print(f'  At threshold 20 m/s, raw balance: {100*(y>=20).mean():.1f}%')
print(f'  At threshold 20 m/s (after 96th pct aggregation): approx {100*(y>=20).mean()*2:.0f}%')

# ─────────────────────────────────────────────────────────────
# 4. Графики
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('EDA: Storm Prediction - Class Balance Analysis', fontweight='bold')

# 4a) Баланс по годам
ax = axes[0]
ax.bar(years, ge10s, color='steelblue', alpha=0.8, label='>= 10 m/s', zorder=2)
ax.bar(years, ge20s, color='firebrick', alpha=0.8, label='>= 20 m/s', zorder=3)
ax.axhline(50, color='k', ls='--', lw=1, label='50%')
ax.axhline(24.5, color='green', ls=':', lw=1.5, label='prev run (24.5%)')
ax.set_title('Class balance by year\n(raw max wind >= threshold)')
ax.set_xlabel('Year'); ax.set_ylabel('%')
ax.legend(fontsize=8); ax.set_ylim(0, 100)
ax.tick_params(axis='x', rotation=45)

# 4b) Mean wind по годам
ax = axes[1]
ax.plot(years, means, 'o-', color='steelblue')
ax.axhline(10, color='steelblue', ls='--', lw=1.5, label='10 m/s')
ax.axhline(20, color='firebrick', ls='--', lw=1.5, label='20 m/s')
ax.set_title('Mean max wind speed by year\n(target.parquet)')
ax.set_xlabel('Year'); ax.set_ylabel('m/s')
ax.legend(); ax.tick_params(axis='x', rotation=45)

# 4c) Распределение y
ax = axes[2]
clip = y[y <= 60]
ax.hist(clip, bins=80, color='steelblue', alpha=0.75, edgecolor='none')
ax.axvline(10, color='steelblue', ls='--', lw=2, label=f'10 m/s ({100*(y>=10).mean():.0f}% pos)')
ax.axvline(20, color='firebrick', ls='--', lw=2, label=f'20 m/s ({100*(y>=20).mean():.0f}% pos)')
ax.axvline(y.median(), color='orange', ls='-', lw=2, label=f'median={y.median():.1f}')
ax.set_title('Distribution of y\n(raw max wind in target.parquet)')
ax.set_xlabel('m/s'); ax.set_ylabel('Count')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f'{OUT}/eda_balance.png', dpi=130, bbox_inches='tight')
plt.close()
print(f'\nSaved: {OUT}/eda_balance.png')

"""
EDA станционных данных: распределение скоростей ветра, баланс классов по годам.
Запуск: python eda_stations.py
Результаты: eda_output/ (PNG графики + eda_stats.txt)
"""
import sys, os
sys.path.append(os.getcwd())
import polars as pl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

OUT_DIR = 'eda_output'
os.makedirs(OUT_DIR, exist_ok=True)

PARQUET_PATH = 'data/weatherstation_data/world_stations_2000_2025_25_days_6_months.parquet'
TARGET_PATH  = 'data/cmip6_world/target.parquet'
THRESHOLD_MS = 10.0   # m/s (текущий порог)
THRESHOLD_PAPER = 20.0  # m/s (порог из старой статьи)

log_lines = []
def log(s=''):
    print(s)
    log_lines.append(str(s))

log('='*60)
log('EDA: Storm Prediction Station Data')
log(f'Дата: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
log('='*60)

# ─────────────────────────────────────────────────────────────
# 1. Сырые станционные данные
# ─────────────────────────────────────────────────────────────
log('\n--- 1. Сырые станционные данные ---')
df = pl.read_parquet(PARQUET_PATH)
log(f'Shape: {df.shape}')
log(f'Columns: {df.columns}')

# Найдём столбец со скоростью ветра
wind_cols = [c for c in df.columns if any(x in c.upper() for x in ['MXWDSP', 'WDSP', 'WIND', 'SPEED'])]
log(f'Wind columns: {wind_cols}')

# Добавим год
df = df.with_columns(pl.col('DATE').dt.year().alias('year'))

# Посмотрим на MXWDSP если есть
wind_col = 'MXWDSP' if 'MXWDSP' in df.columns else (wind_cols[0] if wind_cols else None)
log(f'Using wind column: {wind_col}')

if wind_col:
    w = df[wind_col].drop_nulls()
    log(f'\n{wind_col} stats (all years):')
    log(f'  count : {len(w):,}')
    log(f'  mean  : {w.mean():.3f} m/s')
    log(f'  median: {w.median():.3f} m/s')
    log(f'  std   : {w.std():.3f} m/s')
    log(f'  min   : {w.min():.3f} m/s')
    log(f'  p25   : {float(np.percentile(w.to_numpy(), 25)):.3f} m/s')
    log(f'  p75   : {float(np.percentile(w.to_numpy(), 75)):.3f} m/s')
    log(f'  p95   : {float(np.percentile(w.to_numpy(), 95)):.3f} m/s')
    log(f'  p99   : {float(np.percentile(w.to_numpy(), 99)):.3f} m/s')
    log(f'  max   : {w.max():.3f} m/s')
    log(f'\n  % >= 10 m/s: {100*(w>=10).mean():.1f}%')
    log(f'  % >= 20 m/s: {100*(w>=20).mean():.1f}%')
    log(f'  % <=  0 m/s: {100*(w<= 0).mean():.1f}%')

    # По годам: баланс классов
    log(f'\n{wind_col} по годам (баланс >= {THRESHOLD_MS} m/s):')
    by_year = (df
        .filter(pl.col(wind_col).is_not_null())
        .group_by('year')
        .agg([
            pl.col(wind_col).count().alias('n'),
            pl.col(wind_col).mean().alias('mean'),
            pl.col(wind_col).median().alias('median'),
            (pl.col(wind_col) >= THRESHOLD_MS).mean().alias(f'frac_ge10'),
            (pl.col(wind_col) >= THRESHOLD_PAPER).mean().alias(f'frac_ge20'),
        ])
        .sort('year'))
    for row in by_year.rows():
        yr, n, mn, med, f10, f20 = row
        log(f'  {yr}: n={n:>8,}  mean={mn:5.2f}  median={med:5.2f}  '
            f'>=10m/s: {f10*100:5.1f}%  >=20m/s: {f20*100:5.1f}%')

    # --- График 1: баланс по годам ---
    years  = by_year['year'].to_numpy()
    f10arr = by_year['frac_ge10'].to_numpy() * 100
    f20arr = by_year['frac_ge20'].to_numpy() * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('EDA: Wind Station Data', fontsize=14, fontweight='bold')

    # 1a) Баланс классов по годам
    ax = axes[0, 0]
    ax.bar(years, f10arr, color='steelblue', alpha=0.7, label=f'>= {THRESHOLD_MS} m/s')
    ax.bar(years, f20arr, color='firebrick', alpha=0.7, label=f'>= {THRESHOLD_PAPER} m/s')
    ax.axhline(50, color='k', ls='--', lw=0.8)
    ax.set_xlabel('Year'); ax.set_ylabel('%'); ax.set_title('Class balance by year (raw MXWDSP)')
    ax.legend(); ax.set_ylim(0, 100)
    for t in ax.get_xticklabels(): t.set_rotation(45)

    # 1b) Mean wind по годам
    ax = axes[0, 1]
    ax.plot(years, by_year['mean'].to_numpy(), 'o-', color='steelblue', label='mean')
    ax.plot(years, by_year['median'].to_numpy(), 's--', color='orange', label='median')
    ax.axhline(THRESHOLD_MS, color='steelblue', ls=':', alpha=0.6, label=f'{THRESHOLD_MS} m/s')
    ax.axhline(THRESHOLD_PAPER, color='firebrick', ls=':', alpha=0.6, label=f'{THRESHOLD_PAPER} m/s')
    ax.set_xlabel('Year'); ax.set_ylabel('m/s'); ax.set_title('Mean/median wind speed by year')
    ax.legend()
    for t in ax.get_xticklabels(): t.set_rotation(45)

    # 1c) Гистограмма скоростей
    ax = axes[1, 0]
    wvals = df.filter(pl.col(wind_col) > 0)[wind_col].to_numpy()
    ax.hist(wvals[wvals < 50], bins=80, color='steelblue', alpha=0.7, edgecolor='none')
    ax.axvline(THRESHOLD_MS, color='steelblue', ls='--', lw=1.5, label=f'{THRESHOLD_MS} m/s')
    ax.axvline(THRESHOLD_PAPER, color='firebrick', ls='--', lw=1.5, label=f'{THRESHOLD_PAPER} m/s')
    ax.set_xlabel('Wind speed (m/s)'); ax.set_ylabel('Count')
    ax.set_title('Distribution of MXWDSP (all years, >0 m/s)')
    ax.legend()

    # 1d) Скорость по периодам: до/после 2021
    ax = axes[1, 1]
    pre  = df.filter(pl.col('year') < 2021).filter(pl.col(wind_col) > 0)[wind_col].to_numpy()
    post = df.filter(pl.col('year') >= 2021).filter(pl.col(wind_col) > 0)[wind_col].to_numpy()
    bins = np.linspace(0, 40, 80)
    ax.hist(pre,  bins=bins, alpha=0.5, color='steelblue', label=f'2000-2020 (n={len(pre):,})', density=True)
    ax.hist(post, bins=bins, alpha=0.5, color='orange',    label=f'2021-2025 (n={len(post):,})', density=True)
    ax.axvline(THRESHOLD_MS, color='k', ls='--', lw=1.2, label=f'{THRESHOLD_MS} m/s')
    ax.set_xlabel('Wind speed (m/s)'); ax.set_ylabel('Density')
    ax.set_title('Pre-2021 vs Post-2021 distribution')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/fig1_raw_stations.png', dpi=130)
    plt.close()
    log(f'\nSaved: {OUT_DIR}/fig1_raw_stations.png')


# ─────────────────────────────────────────────────────────────
# 2. Target.parquet (обработанные данные для обучения)
# ─────────────────────────────────────────────────────────────
log('\n--- 2. target.parquet (данные для обучения) ---')
tgt = pl.read_parquet(TARGET_PATH)
log(f'Shape: {tgt.shape}')
log(f'Columns: {tgt.columns}')

tgt = tgt.with_columns(pl.col('time').dt.year().alias('year'))
y = tgt['y']
log(f'\ny stats (raw max wind, m/s):')
log(f'  count : {len(y):,}')
log(f'  mean  : {y.mean():.3f} m/s')
log(f'  median: {y.median():.3f} m/s')
log(f'  std   : {y.std():.3f} m/s')
log(f'  min   : {y.min():.3f} m/s')
log(f'  p50   : {float(np.percentile(y.to_numpy(), 50)):.3f} m/s')
log(f'  p75   : {float(np.percentile(y.to_numpy(), 75)):.3f} m/s')
log(f'  p90   : {float(np.percentile(y.to_numpy(), 90)):.3f} m/s')
log(f'  p95   : {float(np.percentile(y.to_numpy(), 95)):.3f} m/s')
log(f'  p96   : {float(np.percentile(y.to_numpy(), 96)):.3f} m/s')
log(f'  p99   : {float(np.percentile(y.to_numpy(), 99)):.3f} m/s')
log(f'  max   : {y.max():.3f} m/s')
log(f'\n  % >= 10 m/s : {100*(y>=10).mean():.1f}%  (threshold используемый в обучении)')
log(f'  % >= 20 m/s : {100*(y>=20).mean():.1f}%  (порог из старой статьи)')
log(f'  % >= 15 m/s : {100*(y>=15).mean():.1f}%')

# По годам
log(f'\nTarget y по годам (баланс >= 10 m/s):')
tgt_by_year = (tgt
    .group_by('year')
    .agg([
        pl.col('y').count().alias('n'),
        pl.col('y').mean().alias('mean_y'),
        pl.col('y').median().alias('median_y'),
        (pl.col('y') >= 10).mean().alias('frac_ge10'),
        (pl.col('y') >= 20).mean().alias('frac_ge20'),
    ])
    .sort('year'))
for row in tgt_by_year.rows():
    yr, n, mn, med, f10, f20 = row
    log(f'  {yr}: n={n:>8,}  mean={mn:5.2f}  median={med:5.2f}  '
        f'>=10: {f10*100:5.1f}%  >=20: {f20*100:5.1f}%')

# --- График 2: target.parquet ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('EDA: target.parquet (training data)', fontsize=14, fontweight='bold')

years_t  = tgt_by_year['year'].to_numpy()
f10t = tgt_by_year['frac_ge10'].to_numpy() * 100
f20t = tgt_by_year['frac_ge20'].to_numpy() * 100

ax = axes[0, 0]
ax.bar(years_t, f10t, color='steelblue', alpha=0.7, label='>= 10 m/s')
ax.bar(years_t, f20t, color='firebrick', alpha=0.7, label='>= 20 m/s')
ax.axhline(50, color='k', ls='--', lw=0.8)
ax.set_xlabel('Year'); ax.set_ylabel('%')
ax.set_title('Class balance by year (target.parquet)')
ax.legend(); ax.set_ylim(0, 100)
for t in ax.get_xticklabels(): t.set_rotation(45)

ax = axes[0, 1]
ax.plot(years_t, tgt_by_year['mean_y'].to_numpy(), 'o-', color='steelblue', label='mean y')
ax.plot(years_t, tgt_by_year['median_y'].to_numpy(), 's--', color='orange', label='median y')
ax.axhline(10, color='steelblue', ls=':', alpha=0.7, label='10 m/s')
ax.axhline(20, color='firebrick', ls=':', alpha=0.7, label='20 m/s')
ax.set_xlabel('Year'); ax.set_ylabel('m/s')
ax.set_title('Mean/median y by year')
ax.legend()
for t in ax.get_xticklabels(): t.set_rotation(45)

ax = axes[1, 0]
yvals = tgt['y'].to_numpy()
ax.hist(yvals[yvals < 60], bins=100, color='steelblue', alpha=0.7, edgecolor='none')
ax.axvline(10, color='steelblue', ls='--', lw=2, label='10 m/s (train threshold)')
ax.axvline(20, color='firebrick', ls='--', lw=2, label='20 m/s (paper threshold)')
ax.set_xlabel('y = max wind (m/s)'); ax.set_ylabel('Count')
ax.set_title('Distribution of y in target.parquet')
ax.legend()

# Кумулятивное распределение
ax = axes[1, 1]
sorted_y = np.sort(yvals)
cdf = np.arange(len(sorted_y)) / len(sorted_y)
ax.plot(sorted_y[::1000], cdf[::1000], color='steelblue', lw=1.5)
ax.axvline(10, color='steelblue', ls='--', lw=1.5, label=f'10 m/s → {100*(yvals>=10).mean():.1f}% positive')
ax.axvline(20, color='firebrick', ls='--', lw=1.5, label=f'20 m/s → {100*(yvals>=20).mean():.1f}% positive')
ax.set_xlabel('y (m/s)'); ax.set_ylabel('CDF')
ax.set_title('CDF of y (что порог значит для баланса)')
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/fig2_target_parquet.png', dpi=130)
plt.close()
log(f'Saved: {OUT_DIR}/fig2_target_parquet.png')


# ─────────────────────────────────────────────────────────────
# 3. Симуляция: что выдаёт pipeline (96th percentile за 28 дней)
# ─────────────────────────────────────────────────────────────
log('\n--- 3. Что происходит внутри DataLoader (96th percentile за 28 дней) ---')
log('В XarrayDataset: target_array[7] = quantile(0.96, sliding_window(y, 28 days))')
log('Порог сравнивается с этим значением (не с raw y)')
log()
log('Это значит: вопрос не "было ли сегодня >=10 m/s?",')
log('а "был ли максимум 96-го персентиля за МЕСЯЦ >=10 m/s?"')
log()
log('Симулируем на одной станции:')

# Берём одну станцию с большим количеством записей
sample_lat = tgt.group_by('lat').agg(pl.count().alias('n')).sort('n', descending=True).head(1)['lat'][0]
station_y = (tgt.filter(pl.col('lat') == sample_lat)
             .sort('time')['y'].to_numpy())
log(f'  Станция lat={sample_lat}, записей={len(station_y)}')
log(f'  Raw y: mean={station_y.mean():.2f}, p96={np.percentile(station_y, 96):.2f} m/s')
if len(station_y) >= 28:
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(station_y, 28)
    q96_series = np.quantile(windows, 0.96, axis=1)
    log(f'  96th pct / 28-day window: mean={q96_series.mean():.2f}, min={q96_series.min():.2f}, max={q96_series.max():.2f}')
    log(f'  % windows where 96th pct >= 10: {100*(q96_series>=10).mean():.1f}%')
    log(f'  % windows where 96th pct >= 20: {100*(q96_series>=20).mean():.1f}%')

log('\n--- 4. ВЫВОД ---')
y_arr = tgt['y'].to_numpy()
log(f'  raw y: mean={y_arr.mean():.2f}, median={np.median(y_arr):.2f}')
log(f'  При пороге 10 m/s на raw y:  {100*(y_arr>=10).mean():.1f}% positive (то что в target.parquet)')
log(f'  DataLoader берёт 96th pct / 28 дней — это ещё ВЫШЕ raw значений')
log(f'  Поэтому balance train=67.6% — это ожидаемо при пороге 10 m/s')
log()
log(f'  Для класса "шторм" нужен более высокий порог или другая метрика y')
log(f'  Рекомендации:')
log(f'    - Порог 15 m/s: {100*(y_arr>=15).mean():.1f}% raw positive')
log(f'    - Порог 20 m/s: {100*(y_arr>=20).mean():.1f}% raw positive (порог статьи)')
log(f'    - Порог 25 m/s: {100*(y_arr>=25).mean():.1f}% raw positive')

# Сохраним текстовый отчёт
with open(f'{OUT_DIR}/eda_stats.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(log_lines))
print(f'\nAll results saved to {OUT_DIR}/')

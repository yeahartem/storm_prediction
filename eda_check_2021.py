"""
Диагностика разрыва 2020→2021 в станционных данных.
Использует lazy scan + sample для экономии памяти.
"""
import polars as pl
import numpy as np
import sys

PARQUET = 'data/weatherstation_data/world_stations_2000_2025_25_days_6_months.parquet'

print('Scanning parquet (lazy)...')
lf = pl.scan_parquet(PARQUET)
print('Columns:', lf.columns)

# Добавляем год
lf = lf.with_columns(pl.col('DATE').dt.year().alias('year'))

# --- Агрегат по годам для MXWDSP ---
print('\n--- MXWDSP по годам (raw station data) ---')
agg = (lf
    .filter(pl.col('MXWDSP').is_not_null() & (pl.col('MXWDSP') > 0))
    .group_by('year')
    .agg([
        pl.len().alias('n'),
        pl.col('MXWDSP').mean().alias('mean'),
        pl.col('MXWDSP').median().alias('median'),
        pl.col('MXWDSP').max().alias('max'),
        pl.col('MXWDSP').quantile(0.95).alias('p95'),
        (pl.col('MXWDSP') >= 10).mean().alias('ge10'),
        (pl.col('MXWDSP') >= 20).mean().alias('ge20'),
        pl.col('MXWDSP').n_unique().alias('n_unique'),
    ])
    .sort('year')
    .collect()
)

for row in agg.rows():
    yr, n, mn, med, mx, p95, g10, g20, uniq = row
    print(f'  {yr}: n={n:>8,}  mean={mn:5.2f}  median={med:4.1f}  p95={p95:5.1f}  '
          f'max={mx:5.1f}  >=10: {g10*100:5.1f}%  >=20: {g20*100:4.1f}%  uniq={uniq}')

# --- Проверим конкретные станции: есть ли данные до И после 2021 ---
print('\n--- Станции с данными в 2020 И 2021 (одна и та же станция) ---')
# Найдём станции которые есть в обоих годах
stations_2020 = (lf.filter(pl.col('year') == 2020)
    .select(['LATITUDE','LONGITUDE']).unique().collect())
stations_2021 = (lf.filter(pl.col('year') == 2021)
    .select(['LATITUDE','LONGITUDE']).unique().collect())

# join на одни и те же станции
common = stations_2020.join(stations_2021, on=['LATITUDE','LONGITUDE'], how='inner')
print(f'Станций в 2020: {len(stations_2020)}, в 2021: {len(stations_2021)}, общих: {len(common)}')

# Для первых 10 общих станций смотрим MXWDSP до и после
sample = common.head(10)
for row in sample.rows():
    lat, lon = row
    for yr in [2020, 2021]:
        d = (lf
            .filter((pl.col('LATITUDE')==lat) & (pl.col('LONGITUDE')==lon) & (pl.col('year')==yr))
            .select('MXWDSP')
            .collect()['MXWDSP']
            .drop_nulls())
        if len(d) > 0:
            sys.stdout.write(f'  ({lat:.2f},{lon:.2f}) {yr}: mean={d.mean():.2f} median={d.median():.2f} max={d.max():.2f} n={len(d)}\n')
    sys.stdout.write('\n')
sys.stdout.flush()

# --- Проверим уникальные значения (дискретность намекает на единицы) ---
print('--- Образец значений MXWDSP 2020 и 2021 (первые 20 уникальных) ---')
for yr in [2020, 2021]:
    vals = (lf
        .filter(pl.col('year') == yr)
        .select('MXWDSP')
        .filter(pl.col('MXWDSP') > 0)
        .collect()['MXWDSP']
        .drop_nulls()
        .unique()
        .sort()
        .to_numpy()[:30])
    print(f'  {yr}: {vals}')

print('\nDone.')

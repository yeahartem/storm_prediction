"""Минимальная диагностика 2021 разрыва — читаем только нужное"""
import polars as pl
import numpy as np

# target.parquet уже загружен без segfault в прошлый раз
TARGET = 'data/cmip6_world/target.parquet'
CLEANED = 'data/weatherstation_data/world_stations_2000_2025_25_days_6_months_cleaned.parquet'
WORLD   = 'data/weatherstation_data/world_stations_2000_2025_25_days_6_months.parquet'

import os
files = [('target', TARGET), ('cleaned', CLEANED), ('world', WORLD)]
for name, path in files:
    size = os.path.getsize(path) / 1e6 if os.path.exists(path) else -1
    print(f'{name}: {size:.0f} MB  exists={os.path.exists(path)}')

print()
# --- Читаем cleaned — меньше должен быть ---
if os.path.exists(CLEANED):
    print('Reading cleaned parquet (lazy)...')
    lf = pl.scan_parquet(CLEANED)
    schema = lf.collect_schema()
    print('Columns:', schema.names())

    # Агрегация только нужного
    result = (lf
        .with_columns(pl.col('DATE').dt.year().alias('year'))
        .filter(pl.col('MXWDSP').is_not_null() & (pl.col('MXWDSP') > 0))
        .group_by('year')
        .agg([
            pl.len().alias('n'),
            pl.col('MXWDSP').mean().alias('mean'),
            pl.col('MXWDSP').median().alias('median'),
            pl.col('MXWDSP').max().alias('max_val'),
            pl.col('MXWDSP').quantile(0.95).alias('p95'),
            (pl.col('MXWDSP') >= 10).mean().alias('ge10'),
            (pl.col('MXWDSP') >= 20).mean().alias('ge20'),
        ])
        .sort('year')
        .collect()
    )
    print('\nMXWDSP по годам (cleaned):')
    for row in result.rows():
        yr, n, mn, med, mx, p95, g10, g20 = row
        print(f'  {yr}: n={n:>7,}  mean={mn:5.2f}  median={med:4.1f}  '
              f'p95={p95:5.1f}  >=10: {g10*100:5.1f}%  >=20: {g20*100:4.1f}%')

    # Уникальные значения 2020 vs 2021
    print('\nУникальные значения MXWDSP (выборка):')
    for yr in [2019, 2020, 2021, 2022]:
        vals = (lf
            .with_columns(pl.col('DATE').dt.year().alias('year'))
            .filter((pl.col('year') == yr) & (pl.col('MXWDSP') > 0) & pl.col('MXWDSP').is_not_null())
            .select('MXWDSP')
            .limit(10000)
            .collect()['MXWDSP']
            .to_numpy())
        uniq = np.unique(np.round(vals, 3))[:25]
        print(f'  {yr}: min={vals.min():.3f} mean={vals.mean():.3f} p95={np.percentile(vals,95):.3f}')
        print(f'       uniq sample: {uniq}')

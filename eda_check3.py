import polars as pl
import numpy as np

CLEANED = 'data/weatherstation_data/world_stations_2000_2025_25_days_6_months_cleaned.parquet'

lf = pl.scan_parquet(CLEANED)
print('Columns:', lf.collect_schema().names())

result = (lf
    .with_columns(pl.col('time').dt.year().alias('year'))
    .filter(pl.col('max_speed').is_not_null() & (pl.col('max_speed') > 0))
    .group_by('year')
    .agg([
        pl.len().alias('n'),
        pl.col('max_speed').mean().alias('mean'),
        pl.col('max_speed').median().alias('median'),
        pl.col('max_speed').max().alias('max_val'),
        pl.col('max_speed').quantile(0.95).alias('p95'),
        (pl.col('max_speed') >= 10).mean().alias('ge10'),
        (pl.col('max_speed') >= 20).mean().alias('ge20'),
    ])
    .sort('year')
    .collect()
)

print('\nmax_speed по годам (cleaned stations):')
for row in result.rows():
    yr, n, mn, med, mx, p95, g10, g20 = row
    print(f'  {yr}: n={n:>7,}  mean={mn:5.2f}  median={med:4.1f}  p95={p95:5.1f}  max={mx:5.1f}  >=10: {g10*100:5.1f}%  >=20: {g20*100:4.1f}%')

# Уникальные значения для диагностики единиц
print('\nВыборка значений max_speed:')
for yr in [2019, 2020, 2021, 2022]:
    vals = (lf
        .with_columns(pl.col('time').dt.year().alias('year'))
        .filter((pl.col('year') == yr) & (pl.col('max_speed') > 0))
        .select('max_speed')
        .limit(5000)
        .collect()['max_speed']
        .to_numpy())
    uniq = np.unique(np.round(vals, 2))[:20]
    print(f'  {yr}: mean={vals.mean():.3f}  p95={np.percentile(vals,95):.3f}  min={vals.min():.3f}')
    print(f'       uniq[:20]={uniq}')

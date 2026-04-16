import polars as pl
import numpy as np

WORLD = 'data/weatherstation_data/world_stations_2000_2025_25_days_6_months.parquet'

lf = pl.scan_parquet(WORLD, low_memory=True)
print('Columns:', lf.collect_schema().names())

# Stats by year
result = (lf
    .with_columns(pl.col('DATE').dt.year().alias('year'))
    .filter(pl.col('MXWDSP').is_not_null() & (pl.col('MXWDSP') > 0))
    .group_by('year')
    .agg([
        pl.len().alias('n'),
        pl.col('MXWDSP').mean().alias('mean'),
        pl.col('MXWDSP').quantile(0.95).alias('p95'),
        (pl.col('MXWDSP') >= 10).mean().alias('ge10'),
    ])
    .sort('year')
    .collect()
)

print('\nMXWDSP by year (fixed stations):')
for row in result.rows():
    yr, n, mn, p95, g10 = row
    print(f'  {yr}: n={n:>7,}  mean={mn:5.2f}  p95={p95:5.1f}  >=10: {g10*100:5.1f}%')

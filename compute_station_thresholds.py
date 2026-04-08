"""
Compute per-station p95 climatology from 2000-2020 (train period).
Saves data/cmip6_world/station_thresholds.parquet with columns: lat, lon, p95
These are used as per-station thresholds for binary target:
  y = 1 if (max_speed > p95_station) AND (max_speed >= 15 m/s)
"""
import polars as pl
import numpy as np

TARGET = 'data/cmip6_world/target.parquet'
OUT    = 'data/cmip6_world/station_thresholds.parquet'

print('Computing per-station p95 on 2000-2020...')
thresholds = (pl.scan_parquet(TARGET)
    .filter(
        (pl.col('time').dt.year() >= 2000) &
        (pl.col('time').dt.year() <= 2020) &
        pl.col('y').is_not_null() &
        (pl.col('y') > 0)
    )
    .group_by(['lat', 'lon'])
    .agg([
        pl.col('y').quantile(0.95).alias('p95'),
        pl.col('y').count().alias('n'),
    ])
    .collect()
)

print(f'Stations with thresholds: {len(thresholds):,}')

# Summary stats
p95_vals = thresholds['p95'].to_numpy()
print(f'\np95 distribution across stations:')
print(f'  mean={p95_vals.mean():.2f}  median={np.median(p95_vals):.2f}')
print(f'  min={p95_vals.min():.2f}  max={p95_vals.max():.2f}')
print(f'  pct < 15 m/s: {100*(p95_vals < 15).mean():.1f}%  (these stations wont contribute positives)')
print(f'  pct >= 15 m/s: {100*(p95_vals >= 15).mean():.1f}%')

thresholds.write_parquet(OUT)
print(f'\nSaved to {OUT}')

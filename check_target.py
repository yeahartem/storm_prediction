import polars as pl
import numpy as np

TARGET = 'data/cmip6_world/target.parquet'
lf = pl.scan_parquet(TARGET)
print('Columns:', lf.collect_schema().names())

result = (lf
    .with_columns(pl.col('time').dt.year().alias('year'))
    .filter(pl.col('y').is_not_null() & (pl.col('y') > 0))
    .group_by('year')
    .agg([
        pl.len().alias('n'),
        pl.col('y').mean().alias('mean'),
        pl.col('y').quantile(0.95).alias('p95'),
        (pl.col('y') >= 10).mean().alias('ge10'),
        (pl.col('y') >= 1).mean().alias('ge1'),
    ])
    .sort('year')
    .collect()
)

print('y (target) in target.parquet by year:')
for row in result.rows():
    yr, n, mn, p95, g10, g1 = row
    print(f'  {yr}: n={n:>7,}  mean={mn:5.2f}  p95={p95:5.1f}  >=10: {g10*100:5.1f}%  >=1: {g1*100:5.1f}%')

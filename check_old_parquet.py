"""Проверяем старый и новый parquet: годы, единицы измерения"""
import polars as pl
import numpy as np

OLD   = 'data/weatherstation_data/world_stations_25_days_6_months.parquet'
COMB  = 'data/weatherstation_data/world_stations_2000_2025_25_days_6_months.parquet'
BAK   = 'data/weatherstation_data/world_stations_2000_2025_25_days_6_months.parquet.bak'

for name, path in [('OLD (original)', OLD), ('BAK (pre-fix)', BAK)]:
    print(f'\n=== {name} ===')
    try:
        lf = pl.scan_parquet(path, low_memory=True)
        cols = lf.collect_schema().names()
        print(f'Columns: {cols}')

        # Найдём колонку с датой
        date_col = next((c for c in cols if 'DATE' in c.upper() or 'date' in c or 'time' in c), None)
        wind_col = next((c for c in cols if 'MXWDSP' in c or 'max_speed' in c or 'MXSPD' in c), None)
        print(f'Date col: {date_col}, Wind col: {wind_col}')

        if date_col and wind_col:
            agg = (lf
                .with_columns(pl.col(date_col).cast(pl.Date).dt.year().alias('year'))
                .filter(pl.col(wind_col).is_not_null() & (pl.col(wind_col) > 0))
                .group_by('year')
                .agg([
                    pl.len().alias('n'),
                    pl.col(wind_col).mean().alias('mean'),
                    pl.col(wind_col).median().alias('median'),
                    pl.col(wind_col).quantile(0.95).alias('p95'),
                    pl.col(wind_col).max().alias('max_val'),
                    (pl.col(wind_col) >= 10).mean().alias('ge10'),
                ])
                .sort('year')
                .collect()
            )
            print(f'\n{wind_col} по годам:')
            for row in agg.rows():
                yr, n, mn, med, p95, mx, g10 = row
                print(f'  {yr}: n={n:>7,}  mean={mn:5.2f}  median={med:4.1f}  '
                      f'p95={p95:5.1f}  max={mx:5.0f}  >=10: {g10*100:5.1f}%')
    except Exception as e:
        print(f'Error: {e}')

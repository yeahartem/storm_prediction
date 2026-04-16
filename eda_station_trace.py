"""
Отслеживаем конкретные станции через 2020->2021 по cleaned parquet.
"""
import polars as pl
import numpy as np

CLEANED = 'data/weatherstation_data/world_stations_2000_2025_25_days_6_months_cleaned.parquet'

lf = pl.scan_parquet(CLEANED)

# Найдём общие станции 2019 и 2021 в narrow lat band
print('Finding stations present in both 2019 and 2021...')
s2019 = (lf
    .filter(pl.col('time').dt.year() == 2019)
    .select(['lat','lon']).unique()
    .collect()
)
s2021 = (lf
    .filter(pl.col('time').dt.year() == 2021)
    .select(['lat','lon']).unique()
    .collect()
)
common = s2019.join(s2021, on=['lat','lon'], how='inner')
print(f'Stations in 2019: {len(s2019)}, in 2021: {len(s2021)}, common: {len(common)}')

# Возьмём 5 случайных общих станций
sample_stations = common.sample(5, seed=42)
print('\nTracking 5 stations across 2018-2023:')
for row in sample_stations.rows():
    lat_s, lon_s = row
    print(f'\n  Station lat={lat_s:.4f}, lon={lon_s:.4f}:')
    for yr in [2018, 2019, 2020, 2021, 2022, 2023]:
        d = (lf
            .filter(
                (pl.col('lat') == lat_s) &
                (pl.col('lon') == lon_s) &
                (pl.col('time').dt.year() == yr)
            )
            .select('max_speed')
            .collect()['max_speed']
            .drop_nulls()
            .to_numpy()
        )
        if len(d) > 0:
            print(f'    {yr}: n={len(d):3d} mean={d.mean():6.3f} '
                  f'max={d.max():6.1f} p95={np.percentile(d,95):6.1f} '
                  f'uniq={sorted(set(d.round(1)))[:8]}')

# Глобальная статистика по общим станциям
print('\n--- Баланс только на ОБЩИХ станциях (2019 vs 2021) ---')
for yr in [2019, 2020, 2021, 2022]:
    d = (lf
        .join(
            common.lazy(),
            on=['lat', 'lon'],
            how='inner'
        )
        .filter(pl.col('time').dt.year() == yr)
        .select('max_speed')
        .collect()['max_speed']
        .drop_nulls()
        .to_numpy()
    )
    print(f'  {yr}: n={len(d):>8,}  mean={d.mean():.3f}  '
          f'p95={np.percentile(d,95):.1f}  >=10: {100*(d>=10).mean():.1f}%')

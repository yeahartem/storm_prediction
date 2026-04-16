"""
Сравнение одних и тех же станций в 2020 vs 2021.
Читаем маленький срез по конкретным координатам.
"""
import polars as pl
import numpy as np

WORLD = 'data/weatherstation_data/world_stations_2000_2025_25_days_6_months.parquet'

lf = pl.scan_parquet(WORLD, low_memory=True)
schema = lf.collect_schema()
print('Columns:', schema.names()[:15], '...')

# Ищем станции в узком диапазоне координат — чтобы не грузить всё
# Берём полосу latitude 50-51 (Европа/Сев.Аме рика — хороший охват)
print('\nСтанции latitude 50-51 в 2019 и 2021:')
for yr in [2019, 2020, 2021, 2022]:
    sample = (lf
        .filter(
            (pl.col('LATITUDE').is_between(50, 51)) &
            (pl.col('DATE').dt.year() == yr) &
            pl.col('MXWDSP').is_not_null() &
            (pl.col('MXWDSP') > 0)
        )
        .select(['DATE', 'LATITUDE', 'LONGITUDE', 'MXWDSP', 'WDSP'])
        .collect()
    )
    v = sample['MXWDSP'].to_numpy()
    uniq = np.sort(np.unique(np.round(v, 4)))[:20]
    print(f'\n  Year {yr}: n={len(sample)}, mean={v.mean():.3f}, median={np.median(v):.3f}, '
          f'p95={np.percentile(v,95):.3f}, max={v.max():.3f}')
    print(f'  Unique MXWDSP values (first 20): {uniq}')

    # Найдём конкретную станцию что есть в этом году
    if len(sample) > 0:
        # возьмём первую станцию с несколькими наблюдениями
        by_station = (sample
            .group_by(['LATITUDE','LONGITUDE'])
            .agg([pl.len().alias('n'), pl.col('MXWDSP').mean().alias('mean_wind')])
            .sort('n', descending=True)
            .head(3)
        )
        for row in by_station.rows():
            lat_s, lon_s, n_s, mean_s = row
            print(f'    Station ({lat_s:.3f},{lon_s:.3f}): n={n_s}, mean_MXWDSP={mean_s:.3f}')

print('\n--- Конкретная станция во всех годах ---')
# Найдём станцию с данными во ВСЕХ годах 2018-2023
# Берём lat=50.x, lon=8-9 (Frankfurt area)
for lat_range in [(50.0, 51.0), (48.0, 49.0), (51.0, 52.0)]:
    stations_all_years = None
    for yr in range(2018, 2024):
        s = (lf
            .filter(
                pl.col('LATITUDE').is_between(*lat_range) &
                (pl.col('DATE').dt.year() == yr)
            )
            .select(['LATITUDE', 'LONGITUDE'])
            .unique()
            .collect()
        )
        if stations_all_years is None:
            stations_all_years = s
        else:
            stations_all_years = stations_all_years.join(s, on=['LATITUDE','LONGITUDE'], how='inner')

    if stations_all_years is not None and len(stations_all_years) > 0:
        lat_s = stations_all_years['LATITUDE'][0]
        lon_s = stations_all_years['LONGITUDE'][0]
        print(f'\nStation ({lat_s:.3f},{lon_s:.3f}) - lat range {lat_range}:')
        for yr in range(2018, 2024):
            d = (lf
                .filter(
                    (pl.col('LATITUDE') == lat_s) &
                    (pl.col('LONGITUDE') == lon_s) &
                    (pl.col('DATE').dt.year() == yr)
                )
                .select(['MXWDSP', 'WDSP'])
                .collect()
            )
            if len(d) > 0:
                v = d['MXWDSP'].drop_nulls().to_numpy()
                if len(v) > 0:
                    print(f'  {yr}: n={len(v)}, mean={v.mean():.3f}, '
                          f'max={v.max():.3f}, uniq[:10]={np.sort(np.unique(np.round(v,3)))[:10]}')
        break

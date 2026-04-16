"""
Пересобираем world_stations parquet:
  - 2000-2020: берём OLD parquet (уже правильные m/s, проверенные станции)
  - 2021-2025: берём BAK parquet (2021+ данные в правильных m/s),
               фильтруем только по станциям которые есть в OLD

Результат: единый файл с одним и тем же набором станций,
           все значения в m/s, без разрыва 2020->2021.
"""
import polars as pl
import numpy as np
import time

OLD   = 'data/weatherstation_data/world_stations_25_days_6_months.parquet'
BAK   = 'data/weatherstation_data/world_stations_2000_2025_25_days_6_months.parquet.bak'
OUT   = 'data/weatherstation_data/world_stations_2000_2025_25_days_6_months.parquet'

t0 = time.time()

# ── 1. Загружаем OLD (2000-2020) ──────────────────────────────
print('Loading OLD parquet (2000-2020)...')
old = pl.scan_parquet(OLD, low_memory=True)
# Колонки OLD: DATE, STATION, NAME, MXWDSP, WDSP, TEMP, STP, SLP, PRCP, DEWP, LATITUDE, LONGITUDE, ELEVATION
# Оставляем только нужные (совпадают с BAK)
old_cols = ['DATE', 'STATION', 'NAME', 'MXWDSP', 'WDSP', 'TEMP', 'LATITUDE', 'LONGITUDE', 'ELEVATION']
old_filtered = (old
    .select(old_cols)
    .filter(pl.col('DATE').dt.year() <= 2020)
    .collect()
)
print(f'  OLD rows: {len(old_filtered):,}  ({time.time()-t0:.1f}s)')

# Уникальные станции в OLD (по lat/lon)
old_stations = (old_filtered
    .select(['LATITUDE', 'LONGITUDE'])
    .unique()
)
print(f'  Unique OLD stations: {len(old_stations):,}')

# ── 2. Загружаем BAK 2021+ и фильтруем по станциям OLD ────────
print('\nLoading BAK 2021+ and filtering to OLD stations...')
# В BAK нет STP/SLP/PRCP/DEWP, колонки: DATE, STATION, NAME, MXWDSP, WDSP, TEMP, LATITUDE, LONGITUDE, ELEVATION
bak_cols = ['DATE', 'STATION', 'NAME', 'MXWDSP', 'WDSP', 'TEMP', 'LATITUDE', 'LONGITUDE', 'ELEVATION']
new_data = (pl.scan_parquet(BAK, low_memory=True)
    .filter(pl.col('DATE').dt.year() >= 2021)
    .select(bak_cols)
    .collect()
)
print(f'  BAK 2021+ rows before filter: {len(new_data):,}')

# Фильтр: только станции из OLD
new_data_filtered = new_data.join(
    old_stations,
    on=['LATITUDE', 'LONGITUDE'],
    how='inner'
)
print(f'  After filtering to OLD stations: {len(new_data_filtered):,}')

n_stations_new = new_data_filtered.select(['LATITUDE','LONGITUDE']).unique().__len__()
print(f'  Unique stations in 2021+ after filter: {n_stations_new:,}')

# ── 3. Объединяем ─────────────────────────────────────────────
print('\nConcatenating...')
# Убеждаемся что OLD не содержит строк 2021+
old_clean = old_filtered.filter(pl.col('DATE').dt.year() <= 2020)
combined = pl.concat([old_clean, new_data_filtered], how='vertical_relaxed')
combined = combined.sort(['LATITUDE', 'LONGITUDE', 'DATE'])
print(f'  Combined rows: {len(combined):,}')

# ── 4. Диагностика по годам ───────────────────────────────────
print('\nMXWDSP stats by year (final combined):')
diag = (combined
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
)
for row in diag.rows():
    yr, n, mn, p95, g10 = row
    print(f'  {yr}: n={n:>7,}  mean={mn:5.2f}  p95={p95:5.1f}  >=10: {g10*100:5.1f}%')

# ── 5. Сохраняем ──────────────────────────────────────────────
print(f'\nSaving to {OUT} ...')
combined.write_parquet(OUT)
print(f'Done in {time.time()-t0:.1f}s')
print(f'File size: {__import__("os").path.getsize(OUT)/1e6:.0f} MB')

"""
EDA: per-station percentile thresholds for extreme wind definition.
target = 1 if (wind > p_N_station) AND (wind >= 15 m/s)
Tests p90, p92, p95, p96, p97, p98 to find good balance.
Uses only 2000-2020 for climatology (train period).
"""
import pyarrow.parquet as pq
import pyarrow.compute as pc
import numpy as np
import collections

WORLD = 'data/weatherstation_data/world_stations_2000_2025_25_days_6_months.parquet'
ABS_THRESH = 15.0  # m/s
PERCENTILES = [90, 92, 95, 96, 97, 98]
TRAIN_YEARS = set(range(2000, 2021))

print(f'Computing per-station percentiles on 2000-2020, abs threshold={ABS_THRESH} m/s\n')

# Pass 1: collect wind speeds per station for climatology (train years only)
print('Pass 1: collecting per-station wind speeds for climatology...')
station_winds = collections.defaultdict(list)
pf = pq.ParquetFile(WORLD)
for batch in pf.iter_batches(batch_size=1_000_000, columns=['DATE', 'LATITUDE', 'LONGITUDE', 'MXWDSP']):
    years  = pc.year(batch.column('DATE')).to_pylist()
    lats   = batch.column('LATITUDE').to_pylist()
    lons   = batch.column('LONGITUDE').to_pylist()
    winds  = batch.column('MXWDSP').to_pylist()
    for yr, la, lo, w in zip(years, lats, lons, winds):
        if yr in TRAIN_YEARS and w is not None and w > 0:
            station_winds[(la, lo)].append(w)

print(f'  Stations with train data: {len(station_winds):,}')

# Compute per-station percentiles
station_pct = {}  # (lat,lon) -> {pct: value}
for (la, lo), vals in station_winds.items():
    arr = np.array(vals)
    station_pct[(la, lo)] = {p: float(np.percentile(arr, p)) for p in PERCENTILES}

# Summary of percentile values across stations
print('\nDistribution of per-station percentile thresholds (m/s):')
for p in PERCENTILES:
    vals = [station_pct[k][p] for k in station_pct]
    arr = np.array(vals)
    print(f'  p{p}: median={np.median(arr):.1f}  mean={arr.mean():.1f}  '
          f'min={arr.min():.1f}  max={arr.max():.1f}  '
          f'pct_below15={100*(arr<15).mean():.1f}% of stations')

# Pass 2: count positives per year per threshold
print('\nPass 2: counting positives by year and threshold...')
# counts[year][pct] = (n_positive, n_total)
counts = collections.defaultdict(lambda: {p: [0, 0] for p in PERCENTILES})

pf2 = pq.ParquetFile(WORLD)
for batch in pf2.iter_batches(batch_size=1_000_000, columns=['DATE', 'LATITUDE', 'LONGITUDE', 'MXWDSP']):
    years = pc.year(batch.column('DATE')).to_pylist()
    lats  = batch.column('LATITUDE').to_pylist()
    lons  = batch.column('LONGITUDE').to_pylist()
    winds = batch.column('MXWDSP').to_pylist()
    for yr, la, lo, w in zip(years, lats, lons, winds):
        if w is None or w <= 0:
            continue
        key = (la, lo)
        if key not in station_pct:
            continue
        for p in PERCENTILES:
            thresh = station_pct[key][p]
            counts[yr][p][1] += 1
            if w >= ABS_THRESH and w > thresh:
                counts[yr][p][0] += 1

print(f'\nPositive rate (wind >= {ABS_THRESH} m/s AND > p_N_station) by year:')
header = '  year  ' + ''.join(f'  p{p:2d}   ' for p in PERCENTILES)
print(header)
print('  ' + '-'*len(header))
for yr in sorted(counts.keys()):
    row = f'  {yr}'
    for p in PERCENTILES:
        pos, tot = counts[yr][p]
        rate = 100*pos/tot if tot > 0 else 0
        row += f'  {rate:5.1f}%'
    print(row)

# Overall train / val / test split
print('\nOverall positive rate by split:')
for label, years in [('train 2000-2020', range(2000, 2021)),
                     ('val   2021-2022', range(2021, 2023)),
                     ('test  2023-2024', range(2023, 2025))]:
    for p in PERCENTILES:
        tot = sum(counts[y][p][1] for y in years if y in counts)
        pos = sum(counts[y][p][0] for y in years if y in counts)
        rate = 100*pos/tot if tot > 0 else 0
        print(f'  {label}  p{p}: {rate:.2f}%  ({pos:,}/{tot:,})')
    print()

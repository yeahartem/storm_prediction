"""
Step 2+3: Merge yearly parquets + apply continuity filter.
Uses two streaming passes to avoid OOM.

Pass 1: accumulate date counts per station -> find passing stations
Pass 2: stream yearly parquets, write only rows from passing stations
"""
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import numpy as np
import collections
import time
from pathlib import Path

TMP_DIR  = Path('data/weatherstation_data/gsod_yearly_tmp')
OUT_FILE = Path('data/weatherstation_data/world_stations_2000_2025_25_days_6_months.parquet')

DAYS_IN_PERIOD   = 25
PERIODS_IN_A_ROW = 6

t0 = time.time()
all_files = sorted(TMP_DIR.glob('*.parquet'))
print(f'Found {len(all_files)} yearly parquets')

# ── Pass 1: accumulate per-station BIN COUNTS (not raw dates) ────────────────
# Store only Counter of 28-day bin indices -> ~100x less memory than full dates
# Bin index = (date_ordinal - EPOCH) // 28, EPOCH = 2000-01-01
print('\nPass 1: collecting station bin counts...')
import datetime as dtmod
EPOCH = dtmod.date(2000, 1, 1).toordinal()

# station_bins[key] = Counter({bin_idx: n_days})
station_bins = collections.defaultdict(collections.Counter)

for fp in all_files:
    year = fp.stem
    t = pq.read_table(fp, columns=['DATE', 'LATITUDE', 'LONGITUDE'])
    dates = t.column('DATE').to_pylist()
    lats  = t.column('LATITUDE').to_pylist()
    lons  = t.column('LONGITUDE').to_pylist()
    for d, la, lo in zip(dates, lats, lons):
        if la is None or lo is None or d is None:
            continue
        key = (round(float(la), 3), round(float(lo), 3))
        if hasattr(d, 'toordinal'):
            ordinal = d.toordinal()
        else:
            ordinal = int(d) + EPOCH  # pyarrow date32 is days since epoch
        bin_idx = (ordinal - EPOCH) // 28
        station_bins[key][bin_idx] += 1
    print(f'  {year}: {len(t):,} rows, stations so far: {len(station_bins):,}  ({time.time()-t0:.0f}s)')

print(f'Total unique stations: {len(station_bins):,}')

# ── Continuity filter ────────────────────────────────────────────────────────
print('\nApplying continuity filter...')

def passes_continuity(bin_counts, days_in_period=25, periods_in_a_row=6):
    if not bin_counts:
        return False
    max_bin = max(bin_counts)
    run = 0
    for b in range(max_bin + 1):
        if bin_counts.get(b, 0) >= days_in_period:
            run += 1
            if run >= periods_in_a_row:
                return True
        else:
            run = 0
    return False

passing_stations = set()
for key, bin_counts in station_bins.items():
    if passes_continuity(bin_counts, DAYS_IN_PERIOD, PERIODS_IN_A_ROW):
        passing_stations.add(key)

print(f'Stations passing filter: {len(passing_stations):,} / {len(station_bins):,}')
del station_bins  # free memory

# ── Pass 2: stream write filtered rows ──────────────────────────────────────
print('\nPass 2: streaming filtered rows to output...')
schema = pq.read_schema(all_files[0])
writer = pq.ParquetWriter(OUT_FILE, schema, compression='snappy')
total_written = 0

year_stats = {}  # year -> list of mxwdsp values (sample)

for fp in all_files:
    year = int(fp.stem)
    t = pq.read_table(fp)
    lats = t.column('LATITUDE').to_pylist()
    lons = t.column('LONGITUDE').to_pylist()

    mask = pa.array([
        (round(float(la), 3), round(float(lo), 3)) in passing_stations
        if la is not None and lo is not None else False
        for la, lo in zip(lats, lons)
    ])
    filtered = t.filter(mask)
    if len(filtered) > 0:
        writer.write_table(filtered)
        total_written += len(filtered)

        # Collect wind stats
        winds = [w for w in filtered.column('MXWDSP').to_pylist() if w is not None and w > 0]
        if winds:
            year_stats[year] = winds

    print(f'  {year}: kept {len(filtered):,}/{len(t):,}  total={total_written:,}  ({time.time()-t0:.0f}s)')

writer.close()

# ── Stats by year ────────────────────────────────────────────────────────────
print('\nMXWDSP stats by year:')
for yr in sorted(year_stats):
    v = np.array(year_stats[yr])
    print(f'  {yr}: n={len(v):>7,}  mean={v.mean():5.2f}  p95={np.percentile(v,95):5.1f}'
          f'  >=10:{(v>=10).mean()*100:5.1f}%  >=15:{(v>=15).mean()*100:4.1f}%')

print(f'\nTotal rows: {total_written:,}')
print(f'File size: {OUT_FILE.stat().st_size/1e6:.0f} MB')
print(f'Done in {time.time()-t0:.1f}s')

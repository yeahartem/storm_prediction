"""
Build world_stations parquet from scratch from GSOD CSV files (2000-2025).

Pipeline:
1. Read all CSVs year by year using pyarrow -> convert to parquet per year
2. Merge all years into one parquet
3. Apply continuity filter: station must have >= days_in_period (25) days
   in each 4-week period, for >= periods_in_a_row (6) consecutive periods
4. Filter applies to the FULL 2000-2025 time series -> consistent station set

Units: MXWDSP and WDSP in m/s (converted from knots), TEMP in C.
"""
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import numpy as np
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

GSOD_DIR  = Path('gsod_data')
OUT_DIR   = Path('data/weatherstation_data')
OUT_FILE  = OUT_DIR / 'world_stations_2000_2025_25_days_6_months.parquet'
TMP_DIR   = Path('data/weatherstation_data/gsod_yearly_tmp')

DAYS_IN_PERIOD   = 25   # min days in a 4-week period
PERIODS_IN_A_ROW = 6    # min consecutive 4-week periods

KNOTS_TO_MS = 0.51444

# Missing value sentinels used in GSOD
MISSING_VALUES = {99.99, 999.9, 9999.9, 999.0, -999.9, -999.0}

t0 = time.time()

# ── Step 1: Read CSVs per year, save as yearly parquet ──────────────────────
TMP_DIR.mkdir(parents=True, exist_ok=True)

def process_csv_file(path: str) -> pa.Table | None:
    """Read one GSOD CSV and return a pyarrow Table with cleaned columns."""
    import csv
    rows = []
    try:
        with open(path, newline='', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                def g(col, default=None):
                    v = row.get(col, '').strip()
                    return v if v else default

                def gf(col):
                    v = row.get(col, '').strip()
                    if not v:
                        return None
                    try:
                        fv = float(v)
                        return None if fv in MISSING_VALUES else fv
                    except ValueError:
                        return None

                date_str = g('DATE')
                try:
                    date = datetime.strptime(date_str, '%Y-%m-%d').date()
                except Exception:
                    continue

                lat = gf('LATITUDE')
                lon = gf('LONGITUDE')
                if lat is None or lon is None:
                    continue

                # MXWDSP = max(GUST, MXSPD) in knots -> m/s
                gust  = gf('GUST')
                mxspd = gf('MXSPD')
                if gust is not None and mxspd is not None:
                    mxwdsp = max(gust, mxspd) * KNOTS_TO_MS
                elif gust is not None:
                    mxwdsp = gust * KNOTS_TO_MS
                elif mxspd is not None:
                    mxwdsp = mxspd * KNOTS_TO_MS
                else:
                    mxwdsp = None

                wdsp_raw = gf('WDSP')
                wdsp = wdsp_raw * KNOTS_TO_MS if wdsp_raw is not None else None

                temp_raw = gf('TEMP')
                temp = (temp_raw - 32) * 5 / 9 if temp_raw is not None else None

                elev = gf('ELEVATION')

                rows.append({
                    'DATE': date,
                    'STATION': g('STATION'),
                    'NAME': g('NAME'),
                    'MXWDSP': mxwdsp,
                    'WDSP': wdsp,
                    'TEMP': temp,
                    'LATITUDE': lat,
                    'LONGITUDE': lon,
                    'ELEVATION': elev,
                })
    except Exception:
        return None

    if not rows:
        return None

    return pa.table({
        'DATE':      pa.array([r['DATE'] for r in rows], type=pa.date32()),
        'STATION':   pa.array([r['STATION'] for r in rows], type=pa.string()),
        'NAME':      pa.array([r['NAME'] for r in rows], type=pa.string()),
        'MXWDSP':    pa.array([r['MXWDSP'] for r in rows], type=pa.float32()),
        'WDSP':      pa.array([r['WDSP'] for r in rows], type=pa.float32()),
        'TEMP':      pa.array([r['TEMP'] for r in rows], type=pa.float32()),
        'LATITUDE':  pa.array([r['LATITUDE'] for r in rows], type=pa.float32()),
        'LONGITUDE': pa.array([r['LONGITUDE'] for r in rows], type=pa.float32()),
        'ELEVATION': pa.array([r['ELEVATION'] for r in rows], type=pa.float32()),
    })


years = sorted([d.name for d in GSOD_DIR.iterdir() if d.is_dir() and d.name.isdigit()])
print(f'Processing {len(years)} years: {years[0]}-{years[-1]}')

schema = pa.schema([
    ('DATE',      pa.date32()),
    ('STATION',   pa.string()),
    ('NAME',      pa.string()),
    ('MXWDSP',   pa.float32()),
    ('WDSP',      pa.float32()),
    ('TEMP',      pa.float32()),
    ('LATITUDE',  pa.float32()),
    ('LONGITUDE', pa.float32()),
    ('ELEVATION', pa.float32()),
])

for year in years:
    tmp_file = TMP_DIR / f'{year}.parquet'
    if tmp_file.exists():
        print(f'  {year}: already done, skipping')
        continue

    year_dir = GSOD_DIR / year
    csv_files = sorted(year_dir.glob('*.csv'))
    print(f'  {year}: {len(csv_files)} CSVs...', end=' ', flush=True)

    tables = []
    for csv_path in csv_files:
        t = process_csv_file(str(csv_path))
        if t is not None and len(t) > 0:
            tables.append(t)

    if tables:
        year_table = pa.concat_tables(tables)
        pq.write_table(year_table, tmp_file, compression='snappy')
        print(f'{len(year_table):,} rows ({time.time()-t0:.0f}s)')
    else:
        print('empty!')

# ── Step 2: Merge all yearly parquets ────────────────────────────────────────
print(f'\nMerging all years...')
all_files = sorted(TMP_DIR.glob('*.parquet'))
full_table = pq.read_table(all_files)
print(f'  Total rows: {len(full_table):,}  ({time.time()-t0:.0f}s)')

# ── Step 3: Continuity filter ─────────────────────────────────────────────────
print(f'\nApplying continuity filter (>={DAYS_IN_PERIOD} days per 4-week, >={PERIODS_IN_A_ROW} in a row)...')

import collections

# Build {station_id: sorted list of dates} using STATION id (not lat/lon)
# to correctly handle station splits/mergers
df_dict = full_table.to_pydict()
n = len(df_dict['DATE'])

# Group by (LATITUDE, LONGITUDE) rounded to 4 decimals for consistency
station_dates = collections.defaultdict(list)
station_indices = collections.defaultdict(list)
for i in range(n):
    la = round(float(df_dict['LATITUDE'][i]), 4) if df_dict['LATITUDE'][i] is not None else None
    lo = round(float(df_dict['LONGITUDE'][i]), 4) if df_dict['LONGITUDE'][i] is not None else None
    if la is None or lo is None:
        continue
    key = (la, lo)
    station_dates[key].append(df_dict['DATE'][i])
    station_indices[key].append(i)

print(f'  Unique stations (lat/lon): {len(station_dates):,}')

def check_continuity(dates_sorted, days_in_period=25, periods_in_a_row=6):
    """Returns set of date indices (into dates_sorted) that pass the filter."""
    if not dates_sorted:
        return set()

    # Build 4-week bins from first date
    first = dates_sorted[0]
    # Bin each date into 4-week period index
    bins = [(d - first).days // 28 for d in dates_sorted]

    # Count days per bin
    bin_counts = collections.Counter(bins)
    max_bin = max(bin_counts.keys())

    # Find consecutive runs of bins with >= days_in_period
    valid_bins = set()
    run = 0
    run_start = None
    for b in range(max_bin + 1):
        if bin_counts.get(b, 0) >= days_in_period:
            if run == 0:
                run_start = b
            run += 1
            if run >= periods_in_a_row:
                # Mark all bins in this run as valid
                for bb in range(run_start, b + 1):
                    valid_bins.add(bb)
        else:
            run = 0

    if not valid_bins:
        return set()

    # Return indices of dates whose bin is valid
    return {i for i, b in enumerate(bins) if b in valid_bins}

keep_indices = set()
n_stations_kept = 0
for key, dates in station_dates.items():
    sorted_pairs = sorted(zip(dates, station_indices[key]))
    dates_s = [p[0] for p in sorted_pairs]
    idxs_s  = [p[1] for p in sorted_pairs]
    valid_pos = check_continuity(dates_s, DAYS_IN_PERIOD, PERIODS_IN_A_ROW)
    if valid_pos:
        n_stations_kept += 1
        for pos in valid_pos:
            keep_indices.add(idxs_s[pos])

print(f'  Stations passing filter: {n_stations_kept:,}')
print(f'  Rows after filter: {len(keep_indices):,}  ({time.time()-t0:.0f}s)')

# Apply filter
keep_idx_sorted = sorted(keep_indices)
filtered = full_table.take(keep_idx_sorted)

# Sort by lat, lon, date
filtered = filtered.sort_by([('LATITUDE', 'ascending'), ('LONGITUDE', 'ascending'), ('DATE', 'ascending')])

# ── Step 4: Quick stats by year ───────────────────────────────────────────────
print('\nMXWDSP stats by year:')
years_col = pc.year(filtered.column('DATE')).to_pylist()
wind_col  = filtered.column('MXWDSP').to_pylist()
year_winds = collections.defaultdict(list)
for yr, w in zip(years_col, wind_col):
    if w is not None and w > 0:
        year_winds[yr].append(w)
for yr in sorted(year_winds.keys()):
    v = np.array(year_winds[yr])
    print(f'  {yr}: n={len(v):>7,}  mean={v.mean():5.2f}  p95={np.percentile(v,95):5.1f}  >=10:{(v>=10).mean()*100:5.1f}%  >=15:{(v>=15).mean()*100:4.1f}%')

# ── Step 5: Save ──────────────────────────────────────────────────────────────
print(f'\nSaving to {OUT_FILE}...')
OUT_DIR.mkdir(parents=True, exist_ok=True)
pq.write_table(filtered, OUT_FILE, compression='snappy')
print(f'File size: {OUT_FILE.stat().st_size/1e6:.0f} MB')
print(f'Total rows: {len(filtered):,}')
print(f'Done in {time.time()-t0:.1f}s')

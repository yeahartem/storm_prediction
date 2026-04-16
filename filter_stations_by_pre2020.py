"""
Filter world_stations parquet using pyarrow batched reading.
  - Find stations present in 2000-2020
  - Keep 2021+ only for those stations
"""
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
import time
import os

WORLD = 'data/weatherstation_data/world_stations_2000_2025_25_days_6_months.parquet'
OUT   = 'data/weatherstation_data/world_stations_fixed_tmp.parquet'

t0 = time.time()

# ── Pass 1: find pre-2020 stations ───────────────────────────────────────────
print('Pass 1: finding pre-2020 stations...')
pf = pq.ParquetFile(WORLD)
pre2020_latlon = set()
for batch in pf.iter_batches(batch_size=1_000_000, columns=['DATE', 'LATITUDE', 'LONGITUDE']):
    years = pc.year(batch.column('DATE')).to_pylist()
    lats  = batch.column('LATITUDE').to_pylist()
    lons  = batch.column('LONGITUDE').to_pylist()
    for yr, la, lo in zip(years, lats, lons):
        if yr <= 2020 and la is not None and lo is not None:
            pre2020_latlon.add((la, lo))
print(f'  Unique stations in 2000-2020: {len(pre2020_latlon):,}  ({time.time()-t0:.1f}s)')

# ── Pass 2: filter and write ──────────────────────────────────────────────────
print('Pass 2: filtering and writing...')
schema = pf.schema_arrow
writer = pq.ParquetWriter(OUT, schema, compression='snappy')
total_written = 0

for batch in pf.iter_batches(batch_size=1_000_000):
    years = np.array(pc.year(batch.column('DATE')).to_pylist())
    lats  = batch.column('LATITUDE').to_pylist()
    lons  = batch.column('LONGITUDE').to_pylist()
    n = len(years)

    # Vectorized mask: keep if year<=2020 OR (lat,lon) in pre-2020 set
    mask_pre = years <= 2020
    mask_station = np.array([(la, lo) in pre2020_latlon for la, lo in zip(lats, lons)])
    mask = mask_pre | mask_station

    if mask.any():
        indices = np.where(mask)[0].tolist()
        filtered = batch.take(indices)
        writer.write_table(pa.Table.from_batches([filtered]))
        total_written += len(indices)

    elapsed = time.time() - t0
    print(f'  batch done: kept {mask.sum():,}/{n:,}, total={total_written:,}  ({elapsed:.1f}s)', flush=True)

writer.close()
print(f'\nTotal written: {total_written:,} rows ({time.time()-t0:.1f}s)')

# ── Replace original ──────────────────────────────────────────────────────────
print(f'Replacing {WORLD}...')
os.replace(OUT, WORLD)
print(f'File size: {os.path.getsize(WORLD)/1e6:.0f} MB')
print(f'Done in {time.time()-t0:.1f}s')

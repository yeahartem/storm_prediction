import shutil, os
TMP  = 'data/weatherstation_data/world_stations_fixed_tmp.parquet'
DEST = 'data/weatherstation_data/world_stations_2000_2025_25_days_6_months.parquet'
shutil.copy2(TMP, DEST)
os.remove(TMP)
print('Done. File size:', os.path.getsize(DEST)/1e6, 'MB')

import numpy as np

time = np.load('data/cmip6_world/time.npy', allow_pickle=True)
sfcW = np.load('data/cmip6_world/sfcWindmax_16.npy', mmap_mode='r')
mean_v = np.load('data/cmip6_world/mean_32.npy')[0]
std_v = np.load('data/cmip6_world/std_32.npy')[0]

# Convert back to m/s (denormalize)
import pandas as pd
years = np.array([pd.Timestamp(t).year for t in time])

print('sfcWindmax by year (denormalized m/s):')
for yr in range(2000, 2025, 1):
    mask = years == yr
    if mask.sum() == 0:
        continue
    data_yr = sfcW[mask].astype(np.float32) * std_v + mean_v
    print(f'  {yr}: n_days={mask.sum():3d}  mean={data_yr.mean():6.2f}  p95={np.percentile(data_yr, 95):6.2f}  max={data_yr.max():6.1f}')

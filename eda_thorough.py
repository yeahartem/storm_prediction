"""
Thorough EDA for storm prediction paper.
Covers: station data, target labels, thresholds, CMIP6 variables, class balance.
Output: eda_output/ figures + console statistics.
"""
import os
import sys
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger()

os.makedirs('eda_output', exist_ok=True)

TRAIN_END   = '2020-12-31'
VAL_START   = '2021-01-01'
VAL_END     = '2022-12-31'
TEST_START  = '2023-01-01'
TEST_END    = '2024-12-31'

STATION_PATH    = 'data/weatherstation_data/world_stations_2000_2025_25_days_6_months_cleaned.parquet'
TARGET_PATH     = 'data/cmip6_world/target.parquet'
THRESHOLD_PATH  = 'data/cmip6_world/station_thresholds.parquet'
CMIP_DIR        = 'data/cmip6_world/'

SEASON_MAP = {12: 'DJF', 1: 'DJF', 2: 'DJF',
              3: 'MAM', 4: 'MAM', 5: 'MAM',
              6: 'JJA', 7: 'JJA', 8: 'JJA',
              9: 'SON', 10: 'SON', 11: 'SON'}

def get_continent(lat, lon):
    if lat > 15 and lon > -170 and lon < -50:
        return 'North America'
    if lat <= 15 and lat > -60 and lon > -85 and lon < -30:
        return 'South America'
    if lat > 35 and lon > -15 and lon < 45:
        return 'Europe'
    if lat > -40 and lat <= 40 and lon > -20 and lon < 55:
        return 'Africa'
    if lat > 0 and lon >= 45 and lon < 180:
        return 'Asia'
    if lat <= 0 and lon > 100 and lon < 180:
        return 'Oceania'
    return 'Other'

# ============================================================
log.info('='*60)
log.info('SECTION 1: STATION DATA')
log.info('='*60)

log.info('Loading station data...')
st = pd.read_parquet(STATION_PATH)
log.info('Station parquet columns: %s', list(st.columns))
log.info('Station parquet shape: %s', st.shape)

# Normalize column names
if 'max_speed' in st.columns:
    st = st.rename(columns={'max_speed': 'wind_ms'})
elif 'MXWDSP' in st.columns:
    st = st.rename(columns={'MXWDSP': 'wind_ms'})

if 'time' not in st.columns and 'DATE' in st.columns:
    st = st.rename(columns={'DATE': 'time'})
if 'lat' not in st.columns and 'LATITUDE' in st.columns:
    st = st.rename(columns={'LATITUDE': 'lat', 'LONGITUDE': 'lon'})

st['time'] = pd.to_datetime(st['time'])
st['year'] = st['time'].dt.year
st['month'] = st['time'].dt.month
st['season'] = st['month'].map(SEASON_MAP)
st['split'] = 'train'
st.loc[st['time'] >= VAL_START, 'split'] = 'val'
st.loc[st['time'] >= TEST_START, 'split'] = 'test'
st['continent'] = [get_continent(r.lat, r.lon) for r in st[['lat','lon']].itertuples()]

# Remove missing wind
st_clean = st.dropna(subset=['wind_ms'])
st_clean = st_clean[st_clean['wind_ms'] >= 0]

log.info('Total rows: %d', len(st))
log.info('Rows with valid wind: %d (%.1f%%)', len(st_clean), 100*len(st_clean)/len(st))

# Unique stations
stations_all = st.drop_duplicates(subset=['lat','lon'])
log.info('Unique station locations: %d', len(stations_all))

for split, grp in st.groupby('split'):
    n_sta = grp.drop_duplicates(subset=['lat','lon']).shape[0]
    log.info('  %s: %d unique stations, %d observations', split, n_sta, len(grp))

log.info('Stations per continent:')
cont_sta = stations_all.groupby('continent').size().sort_values(ascending=False)
for c, n in cont_sta.items():
    log.info('  %s: %d', c, n)

# Wind speed stats
log.info('Wind speed stats (m/s) by split:')
for split, grp in st_clean.groupby('split'):
    w = grp['wind_ms']
    log.info('  %s: mean=%.2f std=%.2f median=%.2f p95=%.2f p99=%.2f max=%.2f',
             split, w.mean(), w.std(), w.median(),
             w.quantile(0.95), w.quantile(0.99), w.max())

log.info('Wind speed stats by season (full dataset):')
for season in ['DJF','MAM','JJA','SON']:
    w = st_clean[st_clean['season']==season]['wind_ms']
    log.info('  %s: mean=%.2f std=%.2f p95=%.2f n=%d', season, w.mean(), w.std(), w.quantile(0.95), len(w))

log.info('Wind speed stats by continent:')
for cont, grp in st_clean.groupby('continent'):
    w = grp['wind_ms']
    log.info('  %s: mean=%.2f std=%.2f p95=%.2f n=%d', cont, w.mean(), w.std(), w.quantile(0.95), len(w))

# Stations per year
log.info('Unique stations per year:')
sta_per_year = st.groupby('year').apply(lambda g: g.drop_duplicates(subset=['lat','lon']).shape[0])
for y, n in sta_per_year.items():
    log.info('  %d: %d stations', y, n)

# Observations per year
log.info('Observations per year:')
obs_per_year = st_clean.groupby('year').size()
for y, n in obs_per_year.items():
    log.info('  %d: %d', y, n)

# ============================================================
log.info('='*60)
log.info('SECTION 2: TARGET (LABELS)')
log.info('='*60)

log.info('Loading target.parquet...')
tgt = pd.read_parquet(TARGET_PATH)
log.info('Target columns: %s', list(tgt.columns))
log.info('Target shape: %s', tgt.shape)

tgt['time'] = pd.to_datetime(tgt['time'])
tgt['year'] = tgt['time'].dt.year
tgt['month'] = tgt['time'].dt.month
tgt['season'] = tgt['month'].map(SEASON_MAP)
tgt['split'] = 'train'
tgt.loc[tgt['time'] >= VAL_START, 'split'] = 'val'
tgt.loc[tgt['time'] >= TEST_START, 'split'] = 'test'
tgt['continent'] = [get_continent(r.lat, r.lon) for r in tgt[['lat','lon']].itertuples()]

# Check what threshold column we have
log.info('Target y stats: mean=%.3f std=%.3f min=%.3f max=%.3f',
         tgt['y'].mean(), tgt['y'].std(), tgt['y'].min(), tgt['y'].max())

# Positive rate at 15 m/s threshold (the one used in training)
ABS_THRESH = 15.0
tgt['pos'] = (tgt['y'] >= ABS_THRESH).astype(int)

log.info('Positive rate at %.0f m/s threshold:', ABS_THRESH)
for split, grp in tgt.groupby('split'):
    log.info('  %s: %.3f (%d pos / %d total)',
             split, grp['pos'].mean(), grp['pos'].sum(), len(grp))

log.info('Positive rate by year:')
for year, grp in tgt.groupby('year'):
    log.info('  %d: %.3f (n=%d)', year, grp['pos'].mean(), len(grp))

log.info('Positive rate by season:')
for season in ['DJF','MAM','JJA','SON']:
    grp = tgt[tgt['season']==season]
    log.info('  %s: %.3f (n=%d)', season, grp['pos'].mean(), len(grp))

log.info('Positive rate by continent:')
for cont, grp in tgt.groupby('continent'):
    log.info('  %s: %.3f (n=%d)', cont, grp['pos'].mean(), len(grp))

# Unique station locations in target
n_sta_target = tgt.drop_duplicates(subset=['lat','lon']).shape[0]
log.info('Unique station-grid cells in target: %d', n_sta_target)
log.info('Time range: %s to %s', tgt['time'].min(), tgt['time'].max())

# ============================================================
log.info('='*60)
log.info('SECTION 3: STATION THRESHOLDS')
log.info('='*60)

if os.path.exists(THRESHOLD_PATH):
    thr = pd.read_parquet(THRESHOLD_PATH)
    log.info('Threshold parquet columns: %s', list(thr.columns))
    log.info('Threshold parquet shape: %s', thr.shape)
    # find threshold column
    thr_col = [c for c in thr.columns if 'p95' in c.lower() or 'thresh' in c.lower() or 'threshold' in c.lower()]
    if not thr_col:
        thr_col = [c for c in thr.columns if c not in ['lat','lon','station_id','USAF','WBAN']]
    log.info('Threshold columns detected: %s', thr_col)
    for col in thr_col[:3]:
        vals = thr[col].dropna()
        log.info('  %s: mean=%.2f std=%.2f min=%.2f p25=%.2f median=%.2f p75=%.2f p95=%.2f max=%.2f',
                 col, vals.mean(), vals.std(), vals.min(),
                 vals.quantile(0.25), vals.median(), vals.quantile(0.75),
                 vals.quantile(0.95), vals.max())
else:
    log.info('station_thresholds.parquet not found, skipping')
    thr = None
    thr_col = []

# ============================================================
log.info('='*60)
log.info('SECTION 4: CMIP6 DATA')
log.info('='*60)

lat = np.load(os.path.join(CMIP_DIR, 'lat.npy'))
lon = np.load(os.path.join(CMIP_DIR, 'lon.npy'))
time_np = np.load(os.path.join(CMIP_DIR, 'time.npy'), allow_pickle=True)
log.info('CMIP6 grid: lat %s (n=%d), lon %s (n=%d)',
         (lat.min(), lat.max()), len(lat), (lon.min(), lon.max()), len(lon))
log.info('CMIP6 time: %s to %s (n=%d)', time_np[0], time_np[-1], len(time_np))

cmip_vars = ['sfcWindmax', 'pr', 'tasmax', 'tasmin', 'psl']
cmip_data = {}
for var in cmip_vars:
    fpath = os.path.join(CMIP_DIR, f'{var}_16.npy')
    if os.path.exists(fpath):
        arr = np.load(fpath, mmap_mode='r')
        log.info('CMIP6 %s: shape=%s dtype=%s', var, arr.shape, arr.dtype)
        # sample 10000 values for stats
        flat = arr.ravel()
        idx = np.random.choice(len(flat), min(500000, len(flat)), replace=False)
        sample = flat[idx].astype(np.float32)
        sample = sample[np.isfinite(sample)]
        log.info('  mean=%.4f std=%.4f min=%.4f max=%.4f',
                 sample.mean(), sample.std(), sample.min(), sample.max())
        cmip_data[var] = arr
    else:
        log.info('CMIP6 %s: not found', var)

# ============================================================
log.info('='*60)
log.info('SECTION 5: FIGURES')
log.info('='*60)

# ------ Fig 1: Stations per year ------
log.info('Plotting stations per year...')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

years = sorted(sta_per_year.index)
colors = ['steelblue' if y <= 2020 else ('darkorange' if y <= 2022 else 'crimson') for y in years]
axes[0].bar(years, [sta_per_year[y] for y in years], color=colors, edgecolor='white', linewidth=0.4)
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Unique station locations')
axes[0].set_title('Station coverage by year')
axes[0].axvline(2020.5, color='k', linestyle='--', linewidth=1, label='train/val split')
axes[0].axvline(2022.5, color='gray', linestyle='--', linewidth=1, label='val/test split')
axes[0].legend(fontsize=8)
axes[0].tick_params(axis='x', rotation=45)

obs = [obs_per_year.get(y, 0) for y in years]
axes[1].bar(years, obs, color=colors, edgecolor='white', linewidth=0.4)
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Valid observations')
axes[1].set_title('Observations per year')
axes[1].axvline(2020.5, color='k', linestyle='--', linewidth=1)
axes[1].axvline(2022.5, color='gray', linestyle='--', linewidth=1)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('eda_output/01_station_coverage_by_year.png', dpi=150)
plt.close()
log.info('Saved 01_station_coverage_by_year.png')

# ------ Fig 2: Stations per continent ------
log.info('Plotting stations per continent...')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cont_order = cont_sta.index.tolist()
cont_colors = plt.cm.Set2(np.linspace(0, 1, len(cont_order)))

axes[0].barh(cont_order, cont_sta.values, color=cont_colors)
axes[0].set_xlabel('Number of station locations')
axes[0].set_title('Station distribution by continent')
axes[0].invert_yaxis()

# Observations per continent
obs_cont = st_clean.groupby('continent').size().reindex(cont_order).fillna(0)
axes[1].barh(cont_order, obs_cont.values / 1e6, color=cont_colors)
axes[1].set_xlabel('Observations (millions)')
axes[1].set_title('Observations by continent')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('eda_output/02_stations_by_continent.png', dpi=150)
plt.close()
log.info('Saved 02_stations_by_continent.png')

# ------ Fig 3: Wind speed distribution by split ------
log.info('Plotting wind speed distributions...')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

splits = ['train', 'val', 'test']
split_colors = {'train': 'steelblue', 'val': 'darkorange', 'test': 'crimson'}
bins = np.linspace(0, 30, 80)

for i, split in enumerate(splits):
    w = st_clean[st_clean['split']==split]['wind_ms']
    axes[i].hist(w, bins=bins, color=split_colors[split], alpha=0.8, density=True, edgecolor='none')
    axes[i].axvline(w.mean(), color='k', linestyle='-', linewidth=1.5, label=f'mean={w.mean():.1f}')
    axes[i].axvline(w.median(), color='k', linestyle='--', linewidth=1.5, label=f'median={w.median():.1f}')
    axes[i].axvline(15.0, color='red', linestyle=':', linewidth=2, label='threshold=15 m/s')
    axes[i].set_xlabel('Wind speed (m/s)')
    axes[i].set_ylabel('Density' if i==0 else '')
    axes[i].set_title(f'{split.capitalize()} ({len(w)/1e6:.1f}M obs)')
    axes[i].legend(fontsize=8)
    axes[i].set_xlim(0, 30)

plt.suptitle('Wind speed distribution by split', fontsize=13)
plt.tight_layout()
plt.savefig('eda_output/03_wind_distribution_by_split.png', dpi=150)
plt.close()
log.info('Saved 03_wind_distribution_by_split.png')

# ------ Fig 4: Wind speed distribution by season ------
log.info('Plotting wind speed by season...')
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
season_colors = {'DJF': '#2166ac', 'MAM': '#92c5de', 'JJA': '#f4a582', 'SON': '#d6604d'}

for i, season in enumerate(['DJF','MAM','JJA','SON']):
    w = st_clean[st_clean['season']==season]['wind_ms']
    axes[i].hist(w, bins=bins, color=season_colors[season], alpha=0.85, density=True, edgecolor='none')
    axes[i].axvline(w.mean(), color='k', linestyle='-', linewidth=1.5, label=f'mean={w.mean():.1f}')
    axes[i].axvline(15.0, color='red', linestyle=':', linewidth=2, label='15 m/s')
    axes[i].set_xlabel('Wind speed (m/s)')
    axes[i].set_ylabel('Density' if i==0 else '')
    axes[i].set_title(f'{season} (n={len(w)/1e6:.1f}M)')
    axes[i].legend(fontsize=8)
    axes[i].set_xlim(0, 30)

plt.suptitle('Wind speed distribution by season (all years)', fontsize=13)
plt.tight_layout()
plt.savefig('eda_output/04_wind_distribution_by_season.png', dpi=150)
plt.close()
log.info('Saved 04_wind_distribution_by_season.png')

# ------ Fig 5: Positive rate by year ------
log.info('Plotting positive rate by year...')
pos_by_year = tgt.groupby('year')['pos'].mean()

fig, ax = plt.subplots(figsize=(12, 5))
colors_yr = ['steelblue' if y <= 2020 else ('darkorange' if y <= 2022 else 'crimson')
             for y in pos_by_year.index]
ax.bar(pos_by_year.index, pos_by_year.values, color=colors_yr, edgecolor='white', linewidth=0.4)
ax.axvline(2020.5, color='k', linestyle='--', linewidth=1.5, label='train/val split')
ax.axvline(2022.5, color='gray', linestyle='--', linewidth=1.5, label='val/test split')
ax.set_xlabel('Year')
ax.set_ylabel('Positive rate')
ax.set_title('Fraction of samples exceeding 15 m/s threshold by year')
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
from matplotlib.patches import Patch
legend_els = [Patch(facecolor='steelblue', label='Train (2000-2020)'),
              Patch(facecolor='darkorange', label='Val (2021-2022)'),
              Patch(facecolor='crimson', label='Test (2023-2024)')]
ax.legend(handles=legend_els)
plt.tight_layout()
plt.savefig('eda_output/05_positive_rate_by_year.png', dpi=150)
plt.close()
log.info('Saved 05_positive_rate_by_year.png')

# ------ Fig 6: Positive rate by season x split ------
log.info('Plotting positive rate by season and split...')
fig, ax = plt.subplots(figsize=(10, 5))
seasons = ['DJF','MAM','JJA','SON']
x = np.arange(len(seasons))
width = 0.25

for i, split in enumerate(splits):
    rates = [tgt[(tgt['split']==split) & (tgt['season']==s)]['pos'].mean() for s in seasons]
    bars = ax.bar(x + i*width, rates, width, label=split.capitalize(),
                  color=list(split_colors.values())[i], alpha=0.85)

ax.set_xlabel('Season')
ax.set_ylabel('Positive rate')
ax.set_title('Positive rate by season and split')
ax.set_xticks(x + width)
ax.set_xticklabels(seasons)
ax.legend()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
plt.tight_layout()
plt.savefig('eda_output/06_positive_rate_by_season_split.png', dpi=150)
plt.close()
log.info('Saved 06_positive_rate_by_season_split.png')

# ------ Fig 7: Positive rate by continent ------
log.info('Plotting positive rate by continent...')
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# By continent
cont_pos = tgt.groupby('continent')['pos'].agg(['mean','count']).sort_values('mean', ascending=False)
axes[0].barh(cont_pos.index, cont_pos['mean'], color=plt.cm.RdYlBu_r(cont_pos['mean']/cont_pos['mean'].max()))
axes[0].set_xlabel('Positive rate')
axes[0].set_title('Positive rate by continent')
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
for i, (idx, row) in enumerate(cont_pos.iterrows()):
    axes[0].text(row['mean']+0.002, i, f"n={row['count']//1000}k", va='center', fontsize=8)

# By season x continent heatmap
pivot = tgt.groupby(['continent','season'])['pos'].mean().unstack()[seasons]
im = axes[1].imshow(pivot.values, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=0.5)
axes[1].set_xticks(range(len(seasons)))
axes[1].set_xticklabels(seasons)
axes[1].set_yticks(range(len(pivot.index)))
axes[1].set_yticklabels(pivot.index)
axes[1].set_title('Positive rate: continent x season')
plt.colorbar(im, ax=axes[1], label='Positive rate')
for i in range(len(pivot.index)):
    for j in range(len(seasons)):
        axes[1].text(j, i, f'{pivot.values[i,j]:.2f}', ha='center', va='center', fontsize=8, color='black')

plt.tight_layout()
plt.savefig('eda_output/07_positive_rate_by_continent.png', dpi=150)
plt.close()
log.info('Saved 07_positive_rate_by_continent.png')

# ------ Fig 8: Threshold distribution ------
if thr is not None and thr_col:
    log.info('Plotting threshold distribution...')
    col = thr_col[0]
    vals = thr[col].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(vals, bins=60, color='steelblue', alpha=0.85, edgecolor='none')
    axes[0].axvline(15.0, color='red', linestyle='--', linewidth=2, label='abs threshold 15 m/s')
    axes[0].axvline(vals.median(), color='k', linestyle='-', linewidth=1.5, label=f'median={vals.median():.1f}')
    axes[0].set_xlabel('p95 wind threshold (m/s)')
    axes[0].set_ylabel('Count of stations')
    axes[0].set_title(f'Per-station p95 threshold distribution (n={len(vals)})')
    axes[0].legend()

    # CDF
    sorted_vals = np.sort(vals)
    cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
    axes[1].plot(sorted_vals, cdf, color='steelblue', linewidth=2)
    axes[1].axvline(15.0, color='red', linestyle='--', linewidth=2, label='15 m/s')
    frac_above = (vals >= 15).mean()
    axes[1].set_xlabel('p95 threshold (m/s)')
    axes[1].set_ylabel('CDF')
    axes[1].set_title(f'CDF of station thresholds ({frac_above:.1%} stations have p95 >= 15 m/s)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('eda_output/08_threshold_distribution.png', dpi=150)
    plt.close()
    log.info('Saved 08_threshold_distribution.png')

# ------ Fig 9: CMIP6 wind vs station wind distribution ------
log.info('Plotting CMIP6 wind vs station wind...')
if 'sfcWindmax' in cmip_data:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    arr = cmip_data['sfcWindmax']
    idx = np.random.choice(arr.size, min(500000, arr.size), replace=False)
    cmip_wind = arr.ravel()[idx].astype(np.float32)
    cmip_wind = cmip_wind[np.isfinite(cmip_wind) & (cmip_wind > 0)]

    w_all = st_clean['wind_ms'].values
    idx2 = np.random.choice(len(w_all), min(500000, len(w_all)), replace=False)
    sta_wind = w_all[idx2]

    bins2 = np.linspace(0, 25, 80)
    axes[0].hist(cmip_wind, bins=bins2, density=True, alpha=0.7, color='steelblue', label='CMIP6 sfcWindmax', edgecolor='none')
    axes[0].hist(sta_wind, bins=bins2, density=True, alpha=0.7, color='darkorange', label='GSOD stations', edgecolor='none')
    axes[0].set_xlabel('Wind speed (m/s)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('CMIP6 vs station wind distribution')
    axes[0].legend()

    # QQ-style percentiles
    pcts = np.arange(1, 100)
    q_cmip = np.percentile(cmip_wind, pcts)
    q_sta = np.percentile(sta_wind, pcts)
    axes[1].scatter(q_cmip, q_sta, c=pcts, cmap='viridis', s=20, alpha=0.8)
    lim = max(q_cmip.max(), q_sta.max())
    axes[1].plot([0, lim], [0, lim], 'r--', linewidth=1.5, label='y=x')
    axes[1].set_xlabel('CMIP6 percentiles (m/s)')
    axes[1].set_ylabel('Station percentiles (m/s)')
    axes[1].set_title('Q-Q plot: CMIP6 vs stations')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('eda_output/09_cmip6_vs_station_wind.png', dpi=150)
    plt.close()
    log.info('Saved 09_cmip6_vs_station_wind.png')

# ------ Fig 10: CMIP6 variable distributions ------
log.info('Plotting CMIP6 variable distributions...')
var_labels = {
    'sfcWindmax': 'Max surface wind (normalized)',
    'pr': 'Precipitation (normalized)',
    'tasmax': 'Max temperature (normalized)',
    'tasmin': 'Min temperature (normalized)',
    'psl': 'Sea level pressure (normalized)',
}

fig, axes = plt.subplots(1, len(cmip_data), figsize=(4*len(cmip_data), 4))
if len(cmip_data) == 1:
    axes = [axes]

for ax, (var, arr) in zip(axes, cmip_data.items()):
    idx = np.random.choice(arr.size, min(300000, arr.size), replace=False)
    sample = arr.ravel()[idx].astype(np.float32)
    sample = sample[np.isfinite(sample)]
    ax.hist(sample, bins=60, color='steelblue', alpha=0.85, density=True, edgecolor='none')
    ax.set_title(var_labels.get(var, var), fontsize=9)
    ax.set_xlabel('Normalized value')
    ax.set_ylabel('Density' if ax == axes[0] else '')
    ax.text(0.98, 0.95, f'mean={sample.mean():.2f}\nstd={sample.std():.2f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8)

plt.suptitle('CMIP6 variable distributions (normalized float16)', fontsize=12)
plt.tight_layout()
plt.savefig('eda_output/10_cmip6_variable_distributions.png', dpi=150)
plt.close()
log.info('Saved 10_cmip6_variable_distributions.png')

# ------ Fig 11: Sample count per (season x year) heatmap ------
log.info('Plotting sample count heatmap...')
pivot_n = tgt.groupby(['year','season']).size().unstack()[seasons]
pivot_pos = tgt.groupby(['year','season'])['pos'].mean().unstack()[seasons]

fig, axes = plt.subplots(1, 2, figsize=(14, 8))
im1 = axes[0].imshow(pivot_n.values / 1000, aspect='auto', cmap='Blues')
axes[0].set_xticks(range(4))
axes[0].set_xticklabels(seasons)
axes[0].set_yticks(range(len(pivot_n.index)))
axes[0].set_yticklabels(pivot_n.index)
axes[0].set_title('Sample count (thousands) per year x season')
plt.colorbar(im1, ax=axes[0])
for i in range(len(pivot_n.index)):
    for j in range(4):
        v = pivot_n.values[i,j]
        if not np.isnan(v):
            axes[0].text(j, i, f'{v/1000:.0f}k', ha='center', va='center', fontsize=6)

im2 = axes[1].imshow(pivot_pos.values, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=0.5)
axes[1].set_xticks(range(4))
axes[1].set_xticklabels(seasons)
axes[1].set_yticks(range(len(pivot_pos.index)))
axes[1].set_yticklabels(pivot_pos.index)
axes[1].set_title('Positive rate per year x season')
plt.colorbar(im2, ax=axes[1])
for i in range(len(pivot_pos.index)):
    for j in range(4):
        v = pivot_pos.values[i,j]
        if not np.isnan(v):
            axes[1].text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6)

plt.tight_layout()
plt.savefig('eda_output/11_heatmap_year_season.png', dpi=150)
plt.close()
log.info('Saved 11_heatmap_year_season.png')

# ------ Summary stats for paper ------
log.info('='*60)
log.info('SUMMARY FOR PAPER')
log.info('='*60)
log.info('Total station-grid-cell observations in target: %d', len(tgt))
log.info('Unique station locations: %d', len(stations_all))
log.info('Date range: 2000-01-01 to 2024-12-31')
log.info('Train positive rate: %.1f%%', tgt[tgt['split']=='train']['pos'].mean()*100)
log.info('Val positive rate: %.1f%%', tgt[tgt['split']=='val']['pos'].mean()*100)
log.info('Test positive rate: %.1f%%', tgt[tgt['split']=='test']['pos'].mean()*100)
log.info('CMIP6 grid: %dx%d (%.2f deg resolution)', len(lat), len(lon), lat[1]-lat[0])

log.info('All done. Figures saved to eda_output/')

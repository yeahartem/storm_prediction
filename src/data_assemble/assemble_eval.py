import numpy as np
from omegaconf import DictConfig, OmegaConf
import os 
import pandas as pd

def load_dataset(cfg: DictConfig, time_slices=None):
    """Load climate data from given folder"""

    cfg.precision = 32 #only 32 bit is supported
    
    time_coords = np.load(os.path.join(cfg.data_dir, 'time.npy')).astype('datetime64[D]')
    lat_coords = np.load(os.path.join(cfg.data_dir, 'lat.npy'))
    lon_coords = np.load(os.path.join(cfg.data_dir, 'lon.npy'))
    var_data = np.empty((len(cfg.variables), len(time_coords), len(lat_coords), len(lon_coords)), dtype=np.float32)

    for i, var in enumerate(cfg.variables):

        var_data[i] = np.load(os.path.join(cfg.data_dir, var + f'_{cfg.precision}.npy'))

    # time_range = [np.datetime64(pd.to_datetime(t)) for t in cfg.get("time_limits")]
    # time_idxs = np.where((time_coords >= time_range[0]) & (time_coords <= time_range[1]))[0]
    # var_data = var_data[:, time_idxs]
    if time_slices:
        var_data = var_data[:, :time_slices]
    return var_data, time_coords, lat_coords, lon_coords

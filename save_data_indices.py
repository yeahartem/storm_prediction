"""
Run DataPreLoader with the elevation config and save train/test indices
to numpy files so baselines.py can use identical samples.

Usage:
    python save_data_indices.py
Output:
    data/cmip6_world/train_data_idxs.npy   shape (15, N_train)
    data/cmip6_world/test_data_idxs.npy    shape (15, N_test)

Row index meaning (from data_load.py pixel_aggregation):
    0  lat_idx (with shift)
    1  lon_idx (with shift)
    2  time_idx
    3  time_position (fractional year)
    4  time_position_m (month/12)
    5  lat_real
    6  lon_real
    7  y  (max wind over time_agg_window, m/s)
    ...
   14  station_threshold  (max(p95_station, 15.0))
"""
import sys, os
sys.path.append(os.getcwd())

import logging
import numpy as np
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CONFIG_DIR = os.path.abspath("configs")
CONFIG_NAME = "cmip6_world_elevation_server"
OUT_DIR = "data/cmip6_world"

log.info(f"Loading config: {CONFIG_NAME}")
with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
    cfg = compose(config_name=CONFIG_NAME)

log.info("Instantiating DataPreLoader (takes ~10 min) ...")
from src.regression.data_load import DataPreLoader
dpl = DataPreLoader(cfg)

log.info(f"train_data_idxs shape: {dpl.train_data_idxs.shape}")
log.info(f"test_data_idxs  shape: {dpl.test_data_idxs.shape}")

train_pos = (dpl.train_data_idxs[7, :] >= dpl.train_data_idxs[8, :]).mean()
test_pos  = (dpl.test_data_idxs[7, :]  >= dpl.test_data_idxs[8, :]).mean()
log.info(f"Train positive rate (per-station thresh): {train_pos:.3f}")
log.info(f"Test  positive rate (per-station thresh): {test_pos:.3f}")

np.save(os.path.join(OUT_DIR, "train_data_idxs.npy"), dpl.train_data_idxs)
np.save(os.path.join(OUT_DIR, "test_data_idxs.npy"),  dpl.test_data_idxs)
log.info("Saved train_data_idxs.npy and test_data_idxs.npy")

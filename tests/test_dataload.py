import numpy as np 
import pandas as pd

import os, shutil
import sys
sys.path.append(os.path.join(os.getcwd()))

import hydra
from omegaconf import DictConfig

from src.regression import data_load as dl
from dummy_data import dummy_climate

def test_pre_prepare_target():
    pass
@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs/unit_test_configs"), config_name="dummy_data_leap.yaml")
def test_climatedata_to_patches(cfg: DictConfig):

    #create dummy data
    data, time_coords, lat_coords, lon_coords, climate_vars, df_data, cfg = dummy_climate()

    #testing DataPreLoader init
    preloader = dl.DataPreLoader(cfg)
    blocks = preloader.dataset_as_blocks
    #making identical precision
    data = data.astype(np.float16 if cfg.precision == 16 else np.float32)
    # check on shapes:
    ## size of block
    assert (blocks.shape[-1] == blocks.shape[-2]) and (blocks.shape[-1] == (cfg.half_side_size * 2 + 1)), "spatial block size is not correct"
    ## number of blocks
    assert (blocks.shape[0] * blocks.shape[1] == (len(lat_coords) - 2 * cfg.half_side_size) * (len(lon_coords) - 2 * cfg.half_side_size)), "number of spatial blocks is not correct"
    ## len of time axis
    assert (blocks.shape[2] == len(time_coords) - cfg.time_window // 2 * 2), "temporal number of blocks is not correct"
    ## number of channels
    assert (blocks.shape[-3] == len(climate_vars)), "number of channels is not correct"
    ## each block vs each block made by hands 
    for i_idx, i in enumerate(range(cfg.half_side_size, len(lat_coords) - cfg.half_side_size)):
        for j_idx, j in enumerate(range(cfg.half_side_size, len(lon_coords) - cfg.half_side_size)):
            for t_idx, t in enumerate(range(cfg.time_window//2, len(time_coords) - cfg.time_window//2)):
                handmade_block = data[t-cfg.time_window//2:t+cfg.time_window//2 + 1, :, i-cfg.half_side_size:i+cfg.half_side_size + 1, j-cfg.half_side_size:j+cfg.half_side_size + 1]
                sliding_block  = blocks[i_idx,j_idx,t_idx]
                assert (handmade_block.shape == sliding_block.shape), f"at {(i, j, t)} block shape does not coincide with handmade"
                assert (np.allclose(handmade_block, sliding_block)), f"at {(i, j, t)} block values do not coincide with handmade"
    # deleting test files
    shutil.rmtree(cfg.data_dir)
    

if __name__ == "__main__":
    test_climatedata_to_patches()
import numpy as np 
import pandas as pd
import os, shutil
import sys
sys.path.append(os.path.join(os.getcwd()))
from src.regression import data_load as dl
import datetime
from omegaconf import OmegaConf

def str_to_date(string):
    return datetime.datetime.strptime(string, '%Y-%m-%d').date()

def test_pre_prepare_target():
    pass

def test_climatedata_to_patches():

    # create dummy data
    climate_vars = ['var_1', 'var_2', 'var_3', 'var_4', 'var_5', 'var_6' ]
    var_1 = np.arange(0, 100, 1).reshape(10, 10) + 0.1
    var_2 = np.arange(0, 100, 1).reshape(10, 10) + 0.2
    var_3 = np.arange(0, 100, 1).reshape(10, 10) + 0.3
    var_4 = np.arange(0, 100, 1).reshape(10, 10) + 0.4
    var_5 = np.arange(0, 100, 1).reshape(10, 10) + 0.5
    var_6 = np.arange(0, 100, 1).reshape(10, 10) + 0.6

    time_slice = np.stack([var_1, var_2, var_3, var_4, var_5, var_6], axis=0)
    data = np.stack([time_slice + 0.001*(i+1) for i in range(60)], axis=0)

    start_date = str_to_date('2020-12-02')
    time_coords = np.array([start_date + datetime.timedelta(days=x) for x in range(60)]).astype('datetime64')
    lat_coords = np.arange(0, 12, 1.25)
    lon_coords = np.arange(0, 11, 1.15)

    assert data.shape[0] == len(time_coords)
    assert data.shape[1] == len(climate_vars)
    assert data.shape[2] == len(lat_coords)
    assert data.shape[3] == len(lon_coords)

    print(f"climate data shape: {data.shape}")
    print(f"time coords shape: {len(time_coords)}")
    print(f"lat coords: {lat_coords}")
    print(f"lon coords: {lon_coords}")
    #dummy config
    cfg = OmegaConf.create({"data_dir": "data/dummy_test", "path_to_prepared_target_data": "data/dummy_test/target.parquet",
    "precision": 16, "normalize": False, "time_window": 3, "half_side_size": 3, "start_of_test": str(time_coords[4]), "variables": climate_vars})
    if not os.path.exists(cfg.data_dir):
        os.makedirs(cfg.data_dir)
    
    # saving test files
    with open(os.path.join(cfg.data_dir, 'lat.npy'), 'wb') as f:
        np.save(f, lat_coords)
    with open(os.path.join(cfg.data_dir, 'lon.npy'), 'wb') as f:
        np.save(f, lon_coords)
    with open(os.path.join(cfg.data_dir, 'time.npy'), 'wb') as f:
        np.save(f, time_coords)
        pass
    dummy_mean = np.zeros(len(climate_vars))
    with open(os.path.join(cfg.data_dir, 'mean_16.npy'), 'wb') as f:
        np.save(f, dummy_mean)
    with open(os.path.join(cfg.data_dir, 'mean_32.npy'), 'wb') as f:
        np.save(f, dummy_mean)
    dummy_std = np.ones(len(climate_vars))
    with open(os.path.join(cfg.data_dir, 'std_16.npy'), 'wb') as f:
        np.save(f, dummy_std)
    with open(os.path.join(cfg.data_dir, 'std_32.npy'), 'wb') as f:
        np.save(f, dummy_std)
    for i, name in enumerate(climate_vars):
        with open(os.path.join(cfg.data_dir, name + f'_{cfg.precision}' + '.npy'), 'wb') as f:
            np.save(f, data[:, i, :, :])

    df_data = {"time": np.concatenate((time_coords[0:6],time_coords[0:6])),
               "y": [3, 7] * 3 + [3, 7] * 3,
               "lat": [3, 7] * 3 + [3, 7] * 3,
               "lon": [3, 7] * 3 + [3, 7] * 3,
            #    "station_name": ["A"] * 6 + ["B"] * 6
              }
    df_data = pd.DataFrame(df_data)
    df_data.to_parquet(os.path.join(cfg.data_dir, 'target.parquet'))

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
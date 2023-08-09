import sys,os
sys.path.append(os.getcwd())
import numpy as np
import xarray as xr
import pandas as pd
import logging
import dask
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from omegaconf.errors import ConfigAttributeError
from src.data_assemble.assemble_target import clean_weather_data_RU, clean_weather_data_WORLD, make_target
import time
from datetime import datetime
from omegaconf.omegaconf import open_dict


class CMIP5File():
    """Parse the filename of a CMIP5 file to get the model name and experiment name.
        e.g. filename = 'pr_day_MRI-CGCM3_rcp45_r1i1p1_20560101-20651231.nc' """

    def __init__(self, path=None):
        if path:
            self.filename = os.path.basename(path)
            self.path = path
            self.variable_name = self.filename.split('_')[0]
            self.variable_table = self.filename.split('_')[1]
            if self.variable_table != 'day':
                raise NotImplementedError(f'Only daily data is supported. This file is {self.variable_table}')
            self.model_name = self.filename.split('_')[2]
            self.experiment_name = self.filename.split('_')[3]
            self.ensemble_member = self.filename.split('_')[4]
            self.temporal_subset = self.filename.split('_')[-1]
            self.start_date = self.temporal_subset.split('-')[0]
            self.end_date = self.temporal_subset.split('-')[1].split('.')[0]
            self.start_year = int(self.start_date[:4])
            self.end_year = int(self.end_date[:4])
        # logging.debug(f'CMIP5File: {self}')

    def __str__(self):
        return self.filename()

    def __repr__(self):
        return self.filename()
    
    def filename(self):
        return f"{self.variable_name}_{self.variable_table}_{self.model_name}_{self.experiment_name}_{self.ensemble_member}_{self.temporal_subset}.nc"
    

def get_cmip5_files(folder: str, variables) -> list:
    """Get all the CMIP5 files in the directories in folders list"""
    if isinstance(variables, str):
        variables = [variables]
    if isinstance(folder, (list, ListConfig)):
        folder = folder[0]
    files = []
    print(folder)
    for root, dirs, filenames in os.walk(folder):
        for filename in filenames:
            if filename.endswith('.nc'):                
                file = CMIP5File(os.path.join(root, filename))
                if file.variable_name in variables:
                    files.append(file)
    return files


def process_coords(ds, concat_dim='time', drop=True):    
    coord_vars = ['height']
    if drop:
        return ds.drop_vars(coord_vars, errors="ignore")
    else:
        return ds.set_coords(coord_vars)    

def erase_leap_years(data_arr):
    return data_arr.sel(time=~((data_arr.time.dt.month == 2) & (data_arr.time.dt.day == 29)))

def climate_to_npy(files: list, var: str, cfg, save: bool = True):
    """Convert climate data to nc files."""

    assert (cfg.precision == 16 and cfg.saved_normalized) or (cfg.precision == 32 and not cfg.saved_normalized), \
    ''' 16 bit precision works only normalized,
        32 bit precision should be used with saved_normalized=False.'''

    experiment_name = cfg.experiment_name
    train_coords = cfg.train_coords
    rect_coords = list(train_coords.values()) #rect_coords = [min_lat, max_lat, min_lon, max_lon]    
    time_range = [np.datetime64(pd.to_datetime(t)) for t in cfg.get("time_limits")]
    file_paths = [file.path for file in files]

    if not experiment_name:
        experiment_name = files[0].experiment_name
    try:
        data_arr = xr.open_mfdataset(file_paths, preprocess=process_coords, parallel=True, engine='scipy', drop_variables=['height'])
    except TypeError:
        data_arr = xr.open_mfdataset(file_paths, preprocess=process_coords, parallel=True, drop_variables=['height'])
    
    if time_range:
        data_arr = data_arr.sel(time=slice(time_range[0], time_range[1]))
    data_arr.coords['lon'] = (data_arr.coords['lon'] + 180) % 360 - 180
    data_arr = data_arr.sortby(data_arr.lon)
    if cfg.spatial_crop:
        data_arr = data_arr.sel(lat=slice(rect_coords[0], rect_coords[1]), lon=slice(rect_coords[2], rect_coords[3]))
    # Remove leap days
    data_arr = erase_leap_years(data_arr)

    if cfg.precision == 16:
        dtype = np.float16
    elif cfg.precision == 32:
        dtype = np.float32
    else:
        raise NotImplementedError
    
    #Calculate mean and std on train data
    if cfg.start_of_test:
        split_date = datetime.strptime(cfg.start_of_test, '%Y-%m-%d').date()
    else:
        split_date = time_range[1].astype(datetime).date()

    if cfg.load_normalization:
        stds = np.load(os.path.join(cfg.path_to_folder_with_norm_for_infer, "std" + f"_{32}.npy"))
        std = stds[cfg.variables.index(var)]
        means = np.load(os.path.join(cfg.path_to_folder_with_norm_for_infer, "mean" + f"_{32}.npy"))
        mean = means[cfg.variables.index(var)]
    else:
        std = data_arr[var].sel({'time': slice(None, split_date)}).std().compute()
        mean = data_arr[var].sel({'time': slice(None, split_date)}).mean().compute()

    if save:
        logging.info(f"Saving: {var}")
        if cfg.saved_normalized:
            data = data_arr[var].data
            data = np.divide((data - data.mean()), data.std())
            np.save(os.path.join(cfg.data_dir, var + f"_{cfg.precision}.npy"), data.astype(dtype))
        else:
            np.save(os.path.join(cfg.data_dir, var + f"_{cfg.precision}.npy"), data_arr[var].data.astype(dtype))
        time = data_arr[var]["time"].to_numpy()       
        lat = data_arr[var]["lat"].to_numpy()
        lon = data_arr[var]["lon"].to_numpy()
        np.save(os.path.join(cfg.data_dir, "time.npy"), time)
        np.save(os.path.join(cfg.data_dir, "lat.npy"), lat)
        np.save(os.path.join(cfg.data_dir, "lon.npy"), lon)
        logging.info(f"Coords saved: time {time.min()}-{time.max()}, lat {lat.min()}-{lat.max()} step {lat[1]-lat[0]}, lon {lon.min()}-{lon.max()}  step {lon[1]-lon[0]} ")
    return mean, std

def elevation_to_npy(file: str, cfg, save: bool = True):
    """Convert elevation data to nc files."""
    var = 'topo'
    assert (cfg.precision == 16 and cfg.saved_normalized) or (cfg.precision == 32 and not cfg.saved_normalized), \
    ''' 16 bit precision works only normalized,
        32 bit precision should be used with saved_normalized=False.'''
    experiment_name = cfg.experiment_name
    train_coords = cfg.train_coords
    rect_coords = list(train_coords.values()) #rect_coords = [min_lat, max_lat, min_lon, max_lon]    
    try:
        data_arr = xr.open_mfdataset(file, preprocess=process_coords, parallel=True, engine='scipy')
    except TypeError:
        data_arr = xr.open_mfdataset(file, preprocess=process_coords, parallel=True)
    data_arr = data_arr.rename({"X": 'lon', "Y": 'lat'})
    data_arr = data_arr.fillna(0)
    if cfg.spatial_crop:
        data_arr = data_arr.sel(lat=slice(rect_coords[0], rect_coords[1]), lon=slice(rect_coords[2], rect_coords[3]))
    if cfg.precision == 16:
        dtype = np.float16
    elif cfg.precision == 32:
        dtype = np.float32
    else:
        raise NotImplementedError
    #Calculate mean and std
    std = data_arr[var].std().compute()
    mean = data_arr[var].mean().compute()

    if save:
        logging.info(f"Saving: {var}")
        if cfg.saved_normalized:
            data = data_arr[var].data
            data = np.divide((data - data.mean()), data.std())
            np.save(os.path.join(cfg.data_dir, var + f"_{cfg.precision}.npy"), data.astype(dtype))
        else:
            np.save(os.path.join(cfg.data_dir, var + f"_{cfg.precision}.npy"), data_arr[var].data.astype(dtype))
        lat = data_arr[var]["lat"].to_numpy()
        lon = data_arr[var]["lon"].to_numpy()
        np.save(os.path.join(cfg.data_dir, "topo_lat.npy"), lat)
        np.save(os.path.join(cfg.data_dir, "topo_lon.npy"), lon)
        logging.info(f"Coords saved: lat {lat.min()}-{lat.max()} step {lat[1]-lat[0]}, lon {lon.min()}-{lon.max()}  step {lon[1]-lon[0]} ")
    return mean, std

def save_normalization_values(mean_channels: np.array, std_channels: np.array, cfg: DictConfig, prefix: str = ''):
    """Save normalization values for the climate data in given folder"""
    for i in zip(mean_channels, std_channels):
        print(f" mean: {i[0]}, std:  {i[1]}")
    np.save(os.path.join(cfg.data_dir, prefix + "mean_32.npy"), mean_channels.astype(np.float32))
    np.save(os.path.join(cfg.data_dir, prefix + "std_32.npy"), std_channels.astype(np.float32))
    np.save(os.path.join(cfg.data_dir, prefix + "mean_16.npy"), mean_channels.astype(np.float16))
    np.save(os.path.join(cfg.data_dir, prefix + "std_16.npy"), std_channels.astype(np.float16))
    

def load_dataset(cfg: DictConfig):
    """Load climate data from folder in cfg.paths_to_climate_files_folders"""
    files = get_cmip5_files(cfg.paths_to_climate_files_folders, cfg.variables)
    file_paths = [file.path for file in files]
    logging.info(f'loading {file_paths}')
    try:
        data_arr = xr.open_mfdataset(file_paths, combine="by_coords", parallel=True, engine='scipy', preprocess=process_coords, drop_variables=['height']) 
    except TypeError:
        data_arr = xr.open_mfdataset(file_paths, combine="by_coords", parallel=True, preprocess=process_coords, drop_variables=['height']) 
    return data_arr


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs/dataset_configs"), config_name="cmip6_dataset_elevation.yaml")
def prepare_cmip(cfg: DictConfig):    
    logging.info(OmegaConf.to_yaml(cfg))
    logging.info(f"Starting climate data processing")    
    os.makedirs(cfg.data_dir, exist_ok=True)

    with open_dict(cfg):
        cfg.path_to_prepared_target_data = os.path.join(cfg.data_dir, "target.parquet")
        cfg.path_to_prepared_stations = os.path.join(cfg.data_dir, "stations.parquet")
        cfg.path_to_weather_stations_data = os.path.join(cfg.path_to_weather_stations, "data_meteo_full.parquet")
        cfg.path_to_weather_station_list = os.path.join(cfg.path_to_weather_stations, "weatherstation_list.json")
        cfg.path_to_world_weather_stations_data = os.path.join(cfg.path_to_weather_stations, "world_stations_25_days_6_months.parquet")

    #save netcdf files to npy and get normalization values
    mean_channels, std_channels = [], []
    if cfg.make_climate_data:
        for folder in cfg.paths_to_climate_files_folders:
            for var in cfg.variables:
                logging.info(f"{var} in work")
                files = get_cmip5_files(folder, var)
                mean, std = climate_to_npy(files, var, cfg)
                mean_channels.append(mean)
                std_channels.append(std)
                logging.info(f"{var} data saved to {cfg.data_dir}")
    #elevation
    if cfg.make_elevation_data:
        mean_elev, std_elev = elevation_to_npy(cfg.path_to_elevation, cfg)
        save_normalization_values(np.array([mean_elev]), np.array([std_elev]), cfg, prefix='elev_')
        logging.info(f"elevation data saved to {cfg.data_dir}")

    #save normalization values
    if cfg.make_normalization and cfg.make_climate_data:
        save_normalization_values(np.array(mean_channels), np.array(std_channels), cfg)
        logging.info(f"Normalization values saved to {cfg.data_dir}")
    elif cfg.make_normalization:
        for folder in cfg.paths_to_climate_files_folders:
            for var in cfg.variables:
                logging.info(f"{var} in work")
                files = get_cmip5_files(folder, var)
                mean, std = climate_to_npy(files, var, cfg, False)
                mean_channels.append(mean)
                std_channels.append(std)

        save_normalization_values(np.array(mean_channels), np.array(std_channels), cfg)
        logging.info(f"Normalization values saved to {cfg.data_dir}")
    
    #save cleaned target data
    if cfg.make_cleaned_weather_data:
        start_time = time.process_time()
        clean_weather_data_RU(cfg.path_to_weather_stations_data)
        logging.info(f"Ru data clean took {time.process_time() - start_time} seconds")
        start_time = time.process_time()
        clean_weather_data_WORLD(cfg.path_to_world_weather_stations_data)
        logging.info(f"World data clean took {time.process_time() - start_time} seconds")

    #save target data   
    if cfg.make_target:
        make_target(cfg, load_dataset(cfg))        
        logging.info(f"Target data saved to {cfg.data_dir} as {cfg.prepared_target_data_name}")

    with open(os.path.join(cfg.data_dir, 'dataset_config.yaml'), 'w') as file:
        OmegaConf.save(cfg, file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')
    prepare_cmip()
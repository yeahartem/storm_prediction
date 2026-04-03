import sys,os
sys.path.append(os.getcwd())
import numpy as np
import xarray as xr
import pandas as pd
import logging
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from src.data_assemble.assemble_target import clean_weather_data_RU, clean_weather_data_WORLD, make_target
import time as t
from datetime import datetime
from src.utils.norm_values import mean_channels_cmip6, std_channels_cmip6, mean_channels_cmip5, std_channels_cmip5
import warnings
warnings.filterwarnings("ignore")

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
    """Convert climate CMIP data to nc files."""

    assert (cfg.process.precision == 16 and cfg.process.saved_normalized) or (cfg.process.precision == 32 and not cfg.process.saved_normalized), \
    ''' 16 bit precision works only normalized,
        32 bit precision should be used with saved_normalized=False.'''

    experiment_name = cfg.experiment_name
    train_coords = cfg.process.coords
    rect_coords = list(train_coords.values()) #rect_coords = [min_lat, max_lat, min_lon, max_lon]    
    time_range = [np.datetime64(pd.to_datetime(t)) for t in cfg.process.get("time_limits")]
    file_paths = [file.path for file in files]

    if not experiment_name:
        experiment_name = files[0].experiment_name
    try:
        data_arr = xr.open_mfdataset(file_paths, preprocess=process_coords, parallel=True, engine='scipy', drop_variables=['height'])
    except TypeError:
        data_arr = xr.open_mfdataset(file_paths, preprocess=process_coords, parallel=True, drop_variables=['height'])
    
    if time_range:
        candidate = data_arr.sel(time=slice(time_range[0], time_range[1]))
        try:
            has_time = ('time' in candidate.coords) and (candidate.sizes.get('time', 0) > 0)
        except Exception:
            has_time = False
        if has_time:
            data_arr = candidate
        else:
            logging.warning(f"Requested time_limits {cfg.process.get('time_limits')} yield empty selection; using full available time range.")
    data_arr.coords['lon'] = (data_arr.coords['lon'] + 180) % 360 - 180
    data_arr = data_arr.sortby(data_arr.lon)
    if cfg.process.spatial_crop:
        data_arr = data_arr.sel(lat=slice(rect_coords[0], rect_coords[1]), lon=slice(rect_coords[2], rect_coords[3]))
    # Remove leap days
    data_arr = erase_leap_years(data_arr)

    if cfg.process.precision == 16:
        dtype = np.float16
    elif cfg.process.precision == 32:
        dtype = np.float32
    else:
        raise NotImplementedError
    
    #Calculate mean and std on train data
    if cfg.process.start_of_test:
        split_date = datetime.strptime(cfg.process.start_of_test, '%Y-%m-%d').date()
    else:
        split_date = time_range[1].astype(datetime).date()

    if cfg.process.load_normalization:
        if cfg.cmip_type == 'CMIP6':
            stds = std_channels_cmip6
            means = mean_channels_cmip6
        elif cfg.cmip_type == 'CMIP5':
            stds = std_channels_cmip5      # Это заранее посчитанные значения. Выглядит так: 
            means = mean_channels_cmip5    # mean_channels_cmip5 = [9.64286994934082,  # sfcWindmax  7.422664165496826,  # sfcWind...]
        else:
            raise NotImplementedError
        std = stds[cfg.process.variables.index(var)]
        mean = means[cfg.process.variables.index(var)]
    else:
        std = float(data_arr[var].sel({'time': slice(None, split_date)}).std().compute())      # На трейне считают статистику
        mean = float(data_arr[var].sel({'time': slice(None, split_date)}).mean().compute())    # var is one from [sfcWindmax pr tasmax tasmin]


# STOPPED HERE, разбираюсь с нормализацией. Оставлять старую для высоких метрик или переходить на новую.


    if save:
        logging.info(f"Saving: {var}")
        if cfg.process.saved_normalized:
            data = data_arr[var].data
            data = np.divide((data - mean), std)
            np.save(os.path.join(cfg.process.data_dir, var + f"_{cfg.process.precision}.npy"), data.astype(dtype)) # Сохраняется ведь трейн+тест слито, даже без разбиения, а когда он бьется то тогда??? Где отдельный трейн???
        else:
            np.save(os.path.join(cfg.process.data_dir, var + f"_{cfg.process.precision}.npy"), data_arr[var].data.astype(dtype))
        # Retrieve coordinates robustly (dataset-level first, then variable-level), and log safely
        def _get_coord_values(ds, var_da, name_options):
            for name in name_options:
                if name in ds.coords:
                    try:
                        values = ds.coords[name].to_numpy()
                    except Exception:
                        values = np.asarray(ds.coords[name].values)
                    return values
            for name in name_options:
                if name in var_da.coords:
                    try:
                        values = var_da[name].to_numpy()
                    except Exception:
                        values = np.asarray(var_da[name].values)
                    return values
            return np.array([])

        time = _get_coord_values(data_arr, data_arr[var], ["time"])      
        lat = _get_coord_values(data_arr, data_arr[var], ["lat", "latitude", "y"]) 
        lon = _get_coord_values(data_arr, data_arr[var], ["lon", "longitude", "x"]) 

        np.save(os.path.join(cfg.process.data_dir, "time.npy"), time)
        np.save(os.path.join(cfg.process.data_dir, "lat.npy"), lat)
        np.save(os.path.join(cfg.process.data_dir, "lon.npy"), lon)

        # Safe logging without reductions on empty arrays
        time_str = "EMPTY" if time.size == 0 else f"{time.min()}-{time.max()}"
        lat_str = "EMPTY" if lat.size == 0 else f"{lat.min()}-{lat.max()}"
        lon_str = "EMPTY" if lon.size == 0 else f"{lon.min()}-{lon.max()}"
        lat_step = "N/A" if lat.size < 2 else f"{lat[1] - lat[0]}"
        lon_step = "N/A" if lon.size < 2 else f"{lon[1] - lon[0]}"
        logging.info(
            f"Coords saved: time {time_str}, lat {lat_str} step {lat_step}, lon {lon_str}  step {lon_step} "
        )
    # print(f'For {var} mean, std =', mean, std)
    return mean, std

def elevation_to_npy(file: str, cfg, save: bool = True):
    """Convert elevation data to nc files."""
    var = 'topo'
    assert (cfg.process.precision == 16 and cfg.process.saved_normalized) or (cfg.process.precision == 32 and not cfg.process.saved_normalized), \
    ''' 16 bit precision works only normalized,
        32 bit precision should be used with saved_normalized=False.'''
    train_coords = cfg.process.coords
    rect_coords = list(train_coords.values()) #rect_coords = [min_lat, max_lat, min_lon, max_lon]    
    try:
        data_arr = xr.open_mfdataset(file, preprocess=process_coords, parallel=True, engine='scipy')
    except TypeError:
        data_arr = xr.open_mfdataset(file, preprocess=process_coords, parallel=True)
    data_arr = data_arr.rename({"X": 'lon', "Y": 'lat'})
    data_arr = data_arr.fillna(0)
    if cfg.process.spatial_crop:
        data_arr = data_arr.sel(lat=slice(rect_coords[0], rect_coords[1]), lon=slice(rect_coords[2], rect_coords[3]))
    if cfg.process.precision == 16:
        dtype = np.float16
    elif cfg.process.precision == 32:
        dtype = np.float32
    else:
        raise NotImplementedError
    #Calculate mean and std
    std = data_arr[var].std().compute()
    mean = data_arr[var].mean().compute()

    if save:
        logging.info(f"Saving: {var}")
        if cfg.process.saved_normalized:
            data = data_arr[var].data
            data = np.divide((data - data.mean()), data.std())
            np.save(os.path.join(cfg.process.data_dir, f"elev_{cfg.process.precision}.npy"), data.astype(dtype))
        else:
            np.save(os.path.join(cfg.process.data_dir, f"elev_{cfg.process.precision}.npy"), data_arr[var].data.astype(dtype))
        # Retrieve elevation coords robustly and log safely
        try:
            lat = data_arr[var]["lat"].to_numpy()
        except Exception:
            lat = np.asarray(data_arr.coords.get("lat", np.array([])))
        try:
            lon = data_arr[var]["lon"].to_numpy()
        except Exception:
            lon = np.asarray(data_arr.coords.get("lon", np.array([])))
        np.save(os.path.join(cfg.process.data_dir, "elev_lat.npy"), lat)
        np.save(os.path.join(cfg.process.data_dir, "elev_lon.npy"), lon)
        lat_str = "EMPTY" if lat.size == 0 else f"{lat.min()}-{lat.max()}"
        lon_str = "EMPTY" if lon.size == 0 else f"{lon.min()}-{lon.max()}"
        lat_step = "N/A" if lat.size < 2 else f"{lat[1] - lat[0]}"
        lon_step = "N/A" if lon.size < 2 else f"{lon[1] - lon[0]}"
        logging.info(
            f"Coords saved: lat {lat_str} step {lat_step}, lon {lon_str}  step {lon_step} "
        )
    return mean, std


def save_normalization_values(mean_channels: np.array, std_channels: np.array, cfg: DictConfig, prefix: str = ''):
    """Save normalization values for the climate data in given folder"""
    for i in zip(mean_channels, std_channels):
        print(f" mean: {i[0]}, std:  {i[1]}")
    np.save(os.path.join(cfg.process.data_dir, prefix + "mean_32.npy"), mean_channels.astype(np.float32)) # Посмотреть как и где это будет вызываться, проверить тип восстановленных данных, чтобы не было “тихого object dtype”. И порядок проверить, чтобы не было мусора
    np.save(os.path.join(cfg.process.data_dir, prefix + "std_32.npy"), std_channels.astype(np.float32))
    

def load_dataset(cfg: DictConfig):
    """Load climate CMIP data from folder in cfg.raw.paths_to_climate_files_folders""" # ...in './data/cmip5_world_orig' - CMIP
    files = get_cmip5_files(cfg.raw.paths_to_climate_files_folders, cfg.process.variables)
    file_paths = [file.path for file in files]
    logging.info(f'loading {file_paths}')
    try:
        data_arr = xr.open_mfdataset(file_paths, combine="by_coords", parallel=True, engine='scipy', preprocess=process_coords, drop_variables=['height']) 
    except TypeError:
        data_arr = xr.open_mfdataset(file_paths, combine="by_coords", parallel=True, preprocess=process_coords, drop_variables=['height']) 
    return data_arr


# @hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="cmip6_GhostWindNet27")
def prepare_cmip(cfg: DictConfig):     # cfg == 'cmip5_TestNet.yaml'
    print(os.path.join(cfg.process.data_dir, cfg.process.prepared_target_data_name + '.pp1'))
    logging.info(OmegaConf.to_yaml(cfg))
    logging.info(f"Starting climate data processing")    
    os.makedirs(cfg.process.data_dir, exist_ok=True)

    #save netcdf files to npy and get normalization values
    mean_channels, std_channels = [], []
    if cfg.process.make_climate_data:
        for folder in cfg.raw.paths_to_climate_files_folders:   # './data/cmip5_world_orig'
            for var in cfg.process.variables:                   # sfcWindmax pr tasmax tasmin  
                logging.info(f"{var} in work")
                files = get_cmip5_files(folder, var)
                mean, std = climate_to_npy(files, var, cfg)
                mean_channels.append(mean)
                std_channels.append(std)
                logging.info(f"{var} data saved to {cfg.process.data_dir}")
    #elevation
    if cfg.process.make_elevation_data:
        mean_elev, std_elev = elevation_to_npy(cfg.raw.path_to_elevation, cfg)
        save_normalization_values(np.array([mean_elev]), np.array([std_elev]), cfg, prefix='elev_')
        logging.info(f"elevation data saved to {cfg.process.data_dir}")

    #save normalization values
    if cfg.process.make_normalization and cfg.process.make_climate_data:  # Почему передаются общим array и сохраняются с префиксом 32?
        save_normalization_values(np.array(mean_channels), np.array(std_channels), cfg)
        logging.info(f"Normalization values saved to {cfg.process.data_dir}")
    elif cfg.process.make_normalization: # Можно просто сделать нормализацию без подготовки данных (make_climate_data = False)
        for folder in cfg.raw.paths_to_climate_files_folders:
            for var in cfg.process.variables:
                logging.info(f"{var} in work")
                files = get_cmip5_files(folder, var)
                mean, std = climate_to_npy(files, var, cfg, False) #  Почему False ? - потому что это выполняется при make_climate_data = false
                mean_channels.append(mean)
                std_channels.append(std)

        save_normalization_values(np.array(mean_channels), np.array(std_channels), cfg)
        logging.info(f"Normalization values saved to {cfg.process.data_dir}")

    #save cleaned target data
    if cfg.process.make_cleaned_weather_data:
        # start_time = time.process_time()
        # clean_weather_data_RU(cfg.raw.path_to_weather_stations_data)
        # logging.info(f"Ru data clean took {time.process_time() - start_time} seconds")
        start_time = t.process_time()
        clean_weather_data_WORLD(cfg.raw.path_to_world_weather_stations_data) # It saves to "data/weatherstation_data/world_stations_25_days_6_months_cleaned.parquet". (all columns preserved)
        logging.info(f"World data clean took {t.process_time() - start_time} seconds")

    #save target data   
    if cfg.process.make_target:
        make_target(cfg, load_dataset(cfg))   # load_dataset(cfg) is not used here. It loads original CMIP data, but target is based on Stations data     
        logging.info(f"Target data saved to {cfg.process.data_dir} as {cfg.process.prepared_target_data_name}")

    with open(os.path.join(cfg.process.data_dir, 'dataset_config.yaml'), 'w') as file:
        OmegaConf.save(cfg, file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')
    prepare_cmip()
import sys,os
sys.path.append(os.getcwd())
import numpy as np
import xarray as xr
import pandas as pd
import logging
import dask
import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
from src.data_assemble.assemble_target import make_target



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
            self.temporal_subset = self.filename.split('_')[5]
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



def climate_to_netcdf(files: list, var: str, cfg):
    """Convert climate data to nc files."""
    experiment_name = cfg.experiment_name
    train_coords = cfg.train_coords
    rect_coords = list(train_coords.values()) #rect_coords = [min_lat, max_lat, min_lon, max_lon]    
    time_range = [np.datetime64(pd.to_datetime(t)) for t in cfg.get("time_limits")]
    file_paths = [file.path for file in files]

    if not experiment_name:
        experiment_name = files[0].experiment_name
    data_arr = xr.open_mfdataset(file_paths, preprocess=process_coords, parallel=True)
    if time_range:
        data_arr = data_arr.sel(time=slice(time_range[0], time_range[1]))
    if rect_coords:
        data_arr = data_arr.sel(lat=slice(rect_coords[0], rect_coords[1]), lon=slice(rect_coords[2], rect_coords[3]))
    data_arr = data_arr.sel(time=~((data_arr.time.dt.month == 2) & (data_arr.time.dt.day == 29)))

    print(f"Saving: {var}")
    new_file = CMIP5File(None)
    new_file.variable_name = var
    new_file.variable_table = 'day'
    new_file.model_name = 'cmip5'
    new_file.experiment_name = experiment_name
    new_file.ensemble_member = files[0].ensemble_member
    new_file.temporal_subset = f"{data_arr.time.values[0].astype('datetime64[D]')}-{data_arr.time.values[-1].astype('datetime64[D]')}"
    filename = new_file.filename()    
    i = data_arr[var].isnull().sum().compute().data
    print(f"Number of NaNs: {i}")
    data_arr[var].encoding.clear()
    data_arr[var].to_netcdf(os.path.join(cfg.path_to_prepared_data_dir, filename), engine='scipy')  # TODO use faster engine


def make_normalization_values(cfg: DictConfig):
    """Get normalization values for the climate data in given folder"""

    files = get_cmip5_files(cfg.path_to_prepared_data_dir, cfg.variables)
    file_paths = [file.path for file in files]
    data_arr = xr.open_mfdataset(file_paths, combine="by_coords", parallel=True, engine='scipy', preprocess=process_coords)

    norm_time_slice = slice(max(pd.to_datetime(cfg.time_limits[0]), pd.to_datetime(cfg.start_of_test)- pd.DateOffset(years=16)),
                                 pd.to_datetime(cfg.start_of_test))

    mean_channels = data_arr.sel(time=norm_time_slice).mean(dim=['lat', 'lon', 'time']).to_array().compute().data
    std_channels = data_arr.sel(time=norm_time_slice).std(dim=['lat', 'lon', 'time']).to_array().compute().data
    for i in zip(mean_channels, std_channels):
        print(f" mean: {i[0]}, std:  {i[1]}")

    np.save(os.path.join(cfg.path_to_prepared_data_dir, cfg.normalization_values_name + "_mean.npy"), mean_channels)
    np.save(os.path.join(cfg.path_to_prepared_data_dir, cfg.normalization_values_name + "_std.npy"), std_channels)
    


def load_dataset(cfg: DictConfig):

    """Load climate data from folder in cfg.paths_to_climate_files_folders"""
    files = get_cmip5_files(cfg.paths_to_climate_files_folders, cfg.variables)
    file_paths = [file.path for file in files]
    logging.info(f'loading {file_paths}')
    data_arr = xr.open_mfdataset(file_paths, combine="by_coords", parallel=True, engine='scipy', preprocess=process_coords) 
    return data_arr



def test_data_load(cfg: DictConfig):
    """Test if data was loaded correctly."""
    data_arr = load_dataset(cfg)
    i = data_arr.isnull().sum().compute()
    print(f"Number of NaNs total: {i}")

    for var in cfg.variables:
        data_var = data_arr[var]
        i = data_var.isnull().sum().compute().data
        print(f"Opening: {var}")
        print(f"Number of NaNs: {i}")
        print(data_var.shape)
        print(f"""min: {dask.array.min(data_var).compute()}, max: {dask.array.max(data_var).compute()}, std: {dask.array.std(data_var).compute()}""")     
    logging.info(f'OK')


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs/dataset_configs"), config_name="cmip5_dataset_world_local")
def main(cfg: DictConfig):    
    logging.info(OmegaConf.to_yaml(cfg))
    logging.info(f"Starting climate data processing")    
    os.makedirs(cfg.path_to_prepared_data_dir, exist_ok=True)
    

    if cfg.make_climate_data:
        for folder in cfg.paths_to_climate_files_folders:
            for var in cfg.variables:
                logging.info(f"{var} in work")
                files = get_cmip5_files(folder, var)
                climate_to_netcdf(files, var, cfg)
                logging.info(f"{var} data saved to {cfg.path_to_prepared_data_dir}")

    if cfg.make_normalization:
        make_normalization_values(cfg)
        logging.info(f"Normalization values saved to {cfg.path_to_prepared_data_dir} as {cfg.normalization_values_name}")
        
    if cfg.make_target:
        make_target(cfg, load_dataset(cfg))        
        logging.info(f"Target data saved to {cfg.path_to_prepared_data_dir} as {cfg.prepared_target_data_name}")


    test_data_load(cfg)

    with open(os.path.join(cfg.path_to_prepared_data_dir, 'dataset_config.yaml'), 'w') as file:
        OmegaConf.save(cfg, file)


if __name__ == "__main__":

    logging.basicConfig(filename='outputs/dataset.log',
                        filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)

    main()
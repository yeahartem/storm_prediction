import sys,os
sys.path.append(os.getcwd())
import numpy as np
import xarray as xr
import pandas as pd
import logging
import dask
import time
from src.utils.data_utils import interp_timewise_xarray
from src.utils.conf_utils import Config, Dict, timeit
from src.data_assemble.assemble_target import get_stations, get_y, load_weatherstations_RU


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
                raise NotImplementedError('Only daily data is supported.')
            self.model_name = self.filename.split('_')[2]
            self.experiment_name = self.filename.split('_')[3]
            self.ensemble_member = self.filename.split('_')[4]
            self.temporal_subset = self.filename.split('_')[5]

    def __str__(self):
        return self.filename

    def __repr__(self):
        return self.filename
    
    def filename(self):

        return f"{self.variable_name}_{self.variable_table}_{self.model_name}_{self.experiment_name}_{self.ensemble_member}_{self.temporal_subset}.nc"


def get_cmip5_files(folder: str, variables) -> list:
    """Get all the CMIP5 files in the directories in folders list"""
    if not isinstance(variables, list):
        variables = [variables]
    files = []
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
    

def climate_to_npz(files: list, var: str, save_dir: str, time_range: list, rect_coords: list, experiment_name: str):
    """Convert climate data to nc files."""

    file_paths = [file.path for file in files]
    if not experiment_name:
        experiment_name = files[0].experiment_name

    data_arr = xr.open_mfdataset(file_paths, preprocess=process_coords, parallel=True)

    # TODO add normalization data save

    print(f'before time cut {len(data_arr.time)}')
    if time_range:
        data_arr = data_arr.sel(time=slice(time_range[0], time_range[1]))
    if rect_coords:
        data_arr = data_arr.sel(lat=slice(rect_coords[0], rect_coords[1]), lon=slice(rect_coords[2], rect_coords[3]))

    print(f'before leap days excl {len(data_arr.time)}')
    data_arr = data_arr.sel(time=~((data_arr.time.dt.month == 2) & (data_arr.time.dt.day == 29)))
    print(f'after leap days excl {len(data_arr.time)}')
    
    print(f"Saving: {var}")
    new_file = CMIP5File(None)
    new_file.variable_name = var
    new_file.variable_table = 'day'
    new_file.model_name = 'cmip5'
    new_file.experiment_name = experiment_name
    new_file.ensemble_member = files[0].ensemble_member
    new_file.temporal_subset = f"{data_arr.time.values[0].astype('datetime64[D]')}-{data_arr.time.values[-1].astype('datetime64[D]')}"
    filename = new_file.filename()

    data_arr[var].to_netcdf(os.path.join(save_dir, filename), engine='scipy')  # TODO use faster engine




def test_data_load(save_dir: str, variables: list):
    """Test if data was loaded correctly."""

    files = get_cmip5_files(save_dir, variables)
    file_paths = [file.path for file in files]

    logging.info(f'loading {file_paths}')
    data_arr = xr.open_mfdataset(file_paths, combine="by_coords", parallel=True, engine='scipy')
    for var in variables:
        data_var = data_arr[var]
        print(f"Opening: {var}")
        print(data_var.shape)
        print(f"""min: {dask.array.min(data_var).compute()}, max: {dask.array.max(data_var).compute()}, std: {dask.array.std(data_var).compute()}""")     
    logging.info(f'OK')


@timeit
def make_target_data(cfg: Dict):
    
    rectangle_coords = cfg.get("rectangle_coords")
    rectangle_coords = list(rectangle_coords.values())    
    stations_df = get_stations(all_stations_data=cfg.path_to_weather_station_list,
                                    stations_allowed_path=cfg.path_to_allowed_stations,                                    
                                    min_lat=rectangle_coords[0],
                                    max_lat=rectangle_coords[1],                                    
                                    min_lon=rectangle_coords[2],
                                    max_lon=rectangle_coords[3],
                                    max_height= cfg.max_height,
                                    min_height=cfg.min_height)
    
    stations_list = list(stations_df['station_name'].str.casefold())
    if not stations_list:
        raise ValueError('No stations found in the given area')
    
    df = load_weatherstations_RU(cfg.path_to_weather_stations_data, stations_list)
    target_df = get_y(weather_stations_data=df,
                    start=cfg.time_limits[0],
                    end=cfg.time_limits[1],
                    speed_th=cfg.speed_th)
    
    
    # file_paths = [os.path.join(cfg.path_to_prepared_data_dir, var + '.nc') for var in cfg.variables]
    # dataset_as_xarray = xr.open_mfdataset(file_paths,  combine="by_coords", parallel=True, engine='scipy')
    # dataset_as_blocks = make_blocks_numpy(dataset_as_xarray, 3)   
    
    target_df.to_parquet(cfg.path_to_prepared_target_data)
    stations_df.to_parquet(cfg.path_to_prepared_stations)


def main(cfg: Dict):

    logging.info(f"Starting climate data processing")
    
    os.makedirs(cfg.path_to_prepared_data_dir, exist_ok=True)

    rectangle_coords = cfg.get("rectangle_coords")
    rectangle_coords = list(rectangle_coords.values()) #rect_coords = [min_lat, max_lat, min_lon, max_lon]

    time_limits = [np.datetime64(pd.to_datetime(t)) for t in cfg.get("time_limits")]

    if cfg.make_climate_data:

        for var in cfg.variables:
            files = get_cmip5_files(cfg.path_historical_files_folder, var)
            climate_to_npz(files, var, cfg.path_to_prepared_data_dir, time_limits, rectangle_coords, cfg.experiment_name)
            logging.info(f"{var} data saved to {cfg.path_to_prepared_data_dir}")

    if cfg.test_load:
        test_data_load(cfg.path_to_prepared_data_dir, cfg.variables)

    if cfg.make_target:
        make_target_data(cfg)        
        logging.info(f"Target data saved to {cfg.path_to_prepared_target_data}")


    # save lat and lon data
    # ps = glob.glob(os.path.join(root_dir, variables[0], f"*{train_years[0]}*.nc"))
    # x = xr.open_mfdataset(ps[0], parallel=True)
    # lat = x["lat"].to_numpy()
    # lon = x["lon"].to_numpy()
    # np.save(os.path.join(save_dir, "lat.npy"), lat)
    # np.save(os.path.join(save_dir, "lon.npy"), lon)


if __name__ == "__main__":

    logging.basicConfig(filename='logs/dataset.log',
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

    Configuration = Config()
    cfg = Configuration.load_json('./configs/dataset_configs/cmip5_dataset_basic_local.json')

    main(cfg)

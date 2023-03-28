import glob
import os
import numpy as np
import xarray as xr
from tqdm import tqdm
import pandas as pd
from data_utils import interp_timewise_xarray

class CMIP5File():
    """Parse the filename of a CMIP5 file to get the model name and experiment name.
        e.g. filename = 'pr_day_MRI-CGCM3_rcp45_r1i1p1_20560101-20651231.nc' """
    
    def __init__(self, path):
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

def get_cmip5_files(root_dir, variables):
    """Get all the CMIP5 files in the root directory."""
    files = []
    for root, dirs, filenames in os.walk(root_dir):
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
    
def climate_to_npz(files, variables, save_dir, time_range=None, rect_coords=None, interp_res=0.25):
    """Convert climate data to nc files."""

    file_paths = [file.path for file in files]
    # data_arr = xr.open_mfdataset(file_paths, combine='nested', concat_dim='bnds', parallel=True, coords='minimal')  
    data_arr = xr.open_mfdataset(file_paths, preprocess=process_coords, parallel=True)  

    print(f'before time cut {len(data_arr.time)}')
    if time_range:
        data_arr = data_arr.sel(time=slice(time_range[0], time_range[1]))
    if rect_coords:
        data_arr = data_arr.sel(lat=slice(rect_coords[0], rect_coords[1]), lon=slice(rect_coords[2], rect_coords[3]))   
    print(f'before leap days excl {len(data_arr.time)}')    
    data_arr = data_arr.sel(time=~((data_arr.time.dt.month == 2) & (data_arr.time.dt.day == 29)))
    print(f'after leap days excl {len(data_arr.time)}')

    # np.save(os.path.join(save_dir, f"{var}.npy"), data_arr)
    for var in variables:
        print(var)
        data_var = data_arr[var]                
        # data_var = interp_timewise_xarray(data_var,
        #                     res=interp_res,
        #                     interp_method='linear')
        
        data_var.to_netcdf(os.path.join(save_dir, f"{var}.nc"), engine='scipy') # TODO use faster engine


def test_data_load(save_dir, variables):

    file_paths = [os.path.join(save_dir, var + '.nc') for var in variables]
    print(f'loading {file_paths}')
    data_arr = xr.open_mfdataset(file_paths,  combine="by_coords", parallel=True, engine='scipy')
    print(data_arr)

def main(
    root_dir = '/home/teshbek/Datasets/cmip5_orig/rcp45',
    save_dir = './data_mounted/new_cmip5_npz',
    variables = [
        "sfcWindmax",
        "sfcWind",
        "pr",
        "psl",
        "tasmax",
        "tasmin"
    ]
):

    files = get_cmip5_files(root_dir, variables)
    os.makedirs(save_dir, exist_ok=True)

    rectangle_coords =  {
        "lat_min": 36.38,
        "lat_max": 80.52,
        "lon_min": 32.12,
        "lon_max": 181.45
    }

    time_limits = ["2006-01-01", "2015-11-14"]    

    time_limits = [np.datetime64(pd.to_datetime(t)) for t in time_limits]
    rectangle_coords = list(rectangle_coords.values())

    climate_to_npz(files, variables, save_dir, time_limits, rectangle_coords)

    test_data_load(save_dir, variables)
    # save lat and lon data
    # ps = glob.glob(os.path.join(root_dir, variables[0], f"*{train_years[0]}*.nc"))
    # x = xr.open_mfdataset(ps[0], parallel=True)
    # lat = x["lat"].to_numpy()
    # lon = x["lon"].to_numpy()
    # np.save(os.path.join(save_dir, "lat.npy"), lat)
    # np.save(os.path.join(save_dir, "lon.npy"), lon)

    

if __name__ == "__main__":
    main()  
    
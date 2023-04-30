import time
import numpy as np
from tqdm import tqdm
import pandas as pd
import xarray
from datetime import timedelta
import logging
import warnings
warnings.filterwarnings("ignore")


def make_blocks_no_target(
        dataset_as_xarray: dict,
        half_side_size: int = 4,
        time_stack_size=1,  # only 1 for dataset with target
) -> xarray.DataArray:

    start_time = time.process_time()
    bands_list = []
    n_channels = len(dataset_as_xarray.keys())
    example_key = list(dataset_as_xarray.data_vars)[0]

    channels_stack = np.zeros(((n_channels,) + dataset_as_xarray[example_key].shape))

    for i, band in enumerate(dataset_as_xarray.keys()):            
        channels_stack[i] = dataset_as_xarray[band]
        bands_list.append(band)

    channels_stack = np.moveaxis(channels_stack, 0, 1)
    windows = np.lib.stride_tricks.sliding_window_view(channels_stack,
                                                       (time_stack_size, n_channels, 2 * half_side_size + 1,
                                                        2 * half_side_size + 1))
    windows = np.squeeze(windows)
    windows = windows[::time_stack_size]
    windows = np.moveaxis(windows, 0, 2)
    time_coords = dataset_as_xarray[example_key].time.data[:-time_stack_size:time_stack_size]
    lat_coords = dataset_as_xarray[example_key].lat.data[half_side_size:-half_side_size]
    lon_coords = dataset_as_xarray[example_key].lon.data[half_side_size:-half_side_size]
    
    X = xarray.DataArray(
        windows,
        dims=["lat", "lon", "time", "time_stack", "channels", "window_lat", "window_lon"],
        coords={"lat": lat_coords,
                "lon": lon_coords,
                "time": time_coords,
                "time_stack": list(range(time_stack_size)),
                "channels": bands_list,
                "window_lat": list(range(2 * half_side_size + 1)),
                "window_lon": list(range(2 * half_side_size + 1))})
    logging.info(f"Numpy block preparation took {time.process_time() - start_time} seconds")

    return X


def make_filed_values(channel_stack, field, dataset_as_xarray, band):
    field = np.repeat(dataset_as_xarray[band].to_numpy()[np.newaxis, :, :], channel_stack.shape[1], axis=0)

    
def make_blocks(
        dataset_as_xarray: xarray.DataArray,
        half_side_size: int) -> xarray.DataArray:     
     
    start_time = time.process_time()
    n_channels = len(dataset_as_xarray.keys())
    example_key = list(dataset_as_xarray.data_vars)[0]

    channels_stack = np.zeros(
        ((n_channels,) + dataset_as_xarray[example_key].shape))  
    
    for i, band in enumerate(dataset_as_xarray.keys()):
        if 'elev' in band:
            channels_stack[i] = np.repeat(dataset_as_xarray[band].to_numpy()[np.newaxis, :, :],
                                          channels_stack.shape[1], axis=0)
        else: channels_stack[i] = dataset_as_xarray[band]

    channels_stack = np.moveaxis(channels_stack, 0, 1)
    windows = np.lib.stride_tricks.sliding_window_view(channels_stack,
                                                       (1, n_channels, 2 * half_side_size + 1,
                                                        2 * half_side_size + 1))
    windows = np.squeeze(windows)
    windows = np.moveaxis(windows, 0, 2)

    time_coords = dataset_as_xarray[example_key].time.data

    
    lat_coords = dataset_as_xarray[example_key].lat.data[half_side_size:-half_side_size]
    lon_coords = dataset_as_xarray[example_key].lon.data[half_side_size:-half_side_size]

    X = xarray.DataArray(
        windows,
        dims=["lat", "lon", "time", "channels", "window_lat", "window_lon"],
        coords={"lat": lat_coords,
                "lon": lon_coords,
                "time": time_coords.astype('datetime64[D]'),
                "channels": list(range(n_channels)),
                "window_lat": list(range(2 * half_side_size + 1)),
                "window_lon": list(range(2 * half_side_size + 1))})
    
    print(f"Numpy block preparation took {time.process_time() - start_time} seconds")
    
    return X.astype(np.float32)

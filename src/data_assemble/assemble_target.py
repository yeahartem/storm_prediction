import sys,os
sys.path.append(os.getcwd())
import pandas as pd
from functools import partial
import numpy as np
from geopy.distance import great_circle
import logging
import xarray as xr
import time
from omegaconf import DictConfig, OmegaConf, ListConfig
import dask.dataframe as ddt
from src.data_assemble.prepare_target import get_stations_RU, clean_weather_data_RU, clean_weather_data_WORLD
import gc
import swifter


def closest_pixel_for_station(station_row, grid_lat, grid_lon, precise=False):

    """Finds the closest pixel in the dataset for the station
    Returns:
    tuple: x, y lat and lon of the closest pixel on data grid
    """
    coord = [station_row["lat"], station_row["lon"]]        
    theta_offset = 5
    fi_offset = 5
    nearest_lat_idx = (np.abs(np.asarray(grid_lat) - coord[0])).argmin() # find nearest
    nearest_lon_idx = (np.abs(np.asarray(grid_lat) - coord[1])).argmin()
    closest = great_circle((grid_lat[nearest_lat_idx], grid_lon[nearest_lon_idx]), coord).kilometers
    if precise:
        for i, theta in enumerate(grid_lon[nearest_lon_idx - 5:nearest_lon_idx + 5]): # find the closest pixel in 10x10 grid by circle distance            
            for j, fi in enumerate(grid_lat[nearest_lat_idx - 5:nearest_lat_idx + 5]):
                r = great_circle((fi, theta), coord).kilometers
                if r < closest:
                    closest = r
                    theta_offset = i
                    fi_offset = j

    closest_x_idx = theta_offset - 5 + nearest_lon_idx
    closest_y_idx = fi_offset - 5 + nearest_lat_idx

    return grid_lon[closest_x_idx], grid_lat[closest_y_idx]    



def stations_to_data_grid(
        dataset_xarray: xr.DataArray,
        stations_df: pd.DataFrame) -> pd.DataFrame:
    
    """
    maps stations to the data grid pixels
    """
    grid_lon = np.array(dataset_xarray.lon.data) 
    grid_lat = np.array(dataset_xarray.lat[::-1].data)
    closest_partial = partial(closest_pixel_for_station, grid_lat=grid_lat, grid_lon=grid_lon)
    
    start_time = time.process_time()
    stations_df['lon'], stations_df['lat'] = zip(*stations_df.swifter.apply(closest_partial, axis=1))
    logging.info(f"Closest pixel search took {time.process_time() - start_time} seconds")

    return stations_df   



def pre_prepare_target_RU(cfg: DictConfig, dataset_xarray: xr.DataArray):
        
    start_time = time.process_time()
    df = pd.read_parquet(cfg.path_to_weather_stations_data.replace(".parquet", "_cleaned.parquet"))
    logging.info(f"Time to open ru parquet {time.process_time() - start_time} seconds")

    start = cfg.time_limits[0]
    end = cfg.time_limits[1]
    df = df.loc[(df["time"] >= pd.to_datetime(start)) & (df["time"] <= pd.to_datetime(end))]
    if len(cfg.target_column)>0:
        target_cols = [df[col] for col in cfg.target_column]
        df['y'] = list(zip(*target_cols)) 
    elif len(cfg.target_column)==1:
        df.rename(columns={cfg.target_column: 'y'})    
    else:
        raise ValueError

    df = df[["time", "station_name", "y" ]]
    gc.collect()

    stations_df_ru = get_stations_RU(cfg)       
    stations_df_ru = stations_to_data_grid(dataset_xarray=dataset_xarray,
                                          stations_df=stations_df_ru)    
    df = df.merge(stations_df_ru, on='station_name', how='left')
    df['time'] = df['time'].astype('datetime64[D]')

    df.to_parquet(os.path.join(cfg.path_to_prepared_data_dir, cfg.prepared_target_data_name + '.pp1'))

    

def pre_prepare_target_WORLD(cfg: DictConfig, dataset_xarray: xr.DataArray):

    start_time = time.process_time()
    df = pd.read_parquet(cfg.path_to_world_weather_stations_data.replace(".parquet", "_cleaned.parquet"))
    logging.info(f"Time to open world parquet {time.process_time() - start_time} seconds")

    start = cfg.time_limits[0]
    end = cfg.time_limits[1]
    df = df.loc[(df["time"] >= pd.to_datetime(start)) & (df["time"] <= pd.to_datetime(end))]
    if len(cfg.target_column)>0:
        target_cols = [df[col] for col in cfg.target_column]
        df['y'] = list(zip(*target_cols)) 
    elif len(cfg.target_column)==1:
        df.rename(columns={cfg.target_column: 'y'})    
    else:
        raise ValueError
    df = df[["time", "station_name", "y", "lat", "lon", "height" ]]
    gc.collect()         

    df = stations_to_data_grid(dataset_xarray=dataset_xarray,
                                           stations_df=df)
    df['time'] = df['time'].astype('datetime64[D]')
    df.to_parquet(os.path.join(cfg.path_to_prepared_data_dir, cfg.prepared_target_data_name + '.pp2'))


def make_target(cfg: DictConfig, dataset_xarray: xr.DataArray):    

    if cfg.make_cleaned_weather_data:
        start_time = time.process_time()
        clean_weather_data_RU(cfg.path_to_weather_stations_data)
        logging.info(f"Ru data clean took {time.process_time() - start_time} seconds")

        start_time = time.process_time()
        clean_weather_data_WORLD(cfg.path_to_world_weather_stations_data)
        logging.info(f"World data clean took {time.process_time() - start_time} seconds")

    pre_prepare_target_RU(cfg, dataset_xarray)
    pre_prepare_target_WORLD(cfg, dataset_xarray)

    df_ru = pd.read_parquet(os.path.join(cfg.path_to_prepared_data_dir, cfg.prepared_target_data_name + '.pp1'))
    logging.info(f'RU len: {len(df_ru)}')
    df_world = pd.read_parquet(os.path.join(cfg.path_to_prepared_data_dir, cfg.prepared_target_data_name + '.pp2'))
    logging.info(f'WORLD len: {len(df_world)}')

    df_world.drop(columns=['height'], inplace=True)
    start_time = time.process_time()
    target_df = pd.concat([df_ru.dropna(), df_world.dropna()], ignore_index=True)
    logging.info(f"Concat took {time.process_time() - start_time} seconds")
    logging.info(f'TOTAL len: {len(target_df)}')
    target_df.to_parquet(os.path.join(cfg.path_to_prepared_data_dir, cfg.prepared_target_data_name))
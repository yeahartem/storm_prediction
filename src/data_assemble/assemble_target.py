import sys,os
sys.path.append(os.getcwd())
import pandas as pd
from functools import partial
import numpy as np
from geopy.distance import great_circle
import logging
import xarray as xr
import time
import copy

def get_stations_RU(cfg) -> pd.DataFrame:

    all_stations_data = cfg.path_to_weather_station_list

    stations_allowed_path=cfg.path_to_allowed_stations 
    rectangle_coords = list(cfg.train_coords.values())                                   
    min_lat=rectangle_coords[0]
    max_lat=rectangle_coords[1]                                    
    min_lon=rectangle_coords[2]
    max_lon=rectangle_coords[3]
    max_height = cfg.max_height
    min_height = cfg.min_height

    all_stations = pd.read_json(all_stations_data)
    with open(stations_allowed_path) as f:
        stations_allowed = f.read().split('\n')

    all_stations = all_stations.rename(columns={
                            "Широта": "lat",
                            "Долгота": "lon",
                            "Наименование станции": "station_name",
                            "Высота метеопл.": "height"
                            })    
    
    all_stations = all_stations[["station_name", "lat", "lon", "height"]]
    result_stations = all_stations[all_stations['station_name'].isin(stations_allowed)]
    if max_lat:
        result_stations = result_stations[result_stations['lat'] < max_lat]
    if min_lat:
        result_stations = result_stations[result_stations['lat'] > min_lat]
    if max_lon:
        result_stations = result_stations[result_stations['lon'] < max_lon]
    if min_lon:
        result_stations = result_stations[result_stations['lon'] > min_lon]
    if max_height:
        result_stations = result_stations[result_stations['height'] < max_height]
    if min_height:
        result_stations = result_stations[result_stations['height'] > min_height]
    result_stations['station_name'] = result_stations['station_name'].str.casefold()
    
    return result_stations



def load_weatherstations_RU(path_to_weather_stations: str) -> pd.DataFrame:
    """ To load weather stations data from Russia 
        Features: ['Максимальная скорость ветра', 'Средняя скорость ветра', 'Направление ветра', 
                   'Температура воздуха по сухому терм-ру', 'Атмосферное давление на уровне станции', 
                   'Атмосферное давление на уровне моря', 'Сумма осадков', 'Температура поверхности почвы',
                   'Парциальное давление водяного пара', 'Относительная влажность воздуха', 'Температура точки росы'] 
    """  

    def cleanup_ms_name(name: str):
        name = name.replace('"', '')
        name = name.replace(',', '')
        return name

    columns = ["Название метеостанции", "Максимальная скорость", "Средняя скорость ветра", "Дата"] # Add more columns if needed
    df = pd.read_parquet(path_to_weather_stations, columns=columns)
    df["Название метеостанции"] = df["Название метеостанции"].apply(cleanup_ms_name)
    df['Название метеостанции'] = df['Название метеостанции'].str.casefold()
    df["Название метеостанции"] = df["Название метеостанции"].astype("category")
    df['Дата'] = pd.DatetimeIndex(df['Дата'])
    df["Максимальная скорость"] = df[["Максимальная скорость"]].apply(pd.to_numeric, downcast="float")
    df["Дата"] = pd.to_datetime((df["Дата"]), format="%Y/%m/%d")    
    # df = df.sort_values(by=['Название метеостанции', 'Дата'])
    df = df.rename(columns={"Дата": "time",
                            "Название метеостанции": "station_name",
                            "Максимальная скорость": "max_speed",
                            "Средняя скорость ветра": "mean_speed",
                            })    
    return df


def load_weatherstations_WORLD(path_to_weather_stations: str) -> pd.DataFrame: # TODO add target colums argument
    """ To load weather stations data from all world 
        Features: ['DATE', 'STATION', 'NAME', 'MXWDSP', 'WDSP', 'TEMP', 'STP', 'SLP',
       'PRCP', 'DEWP', 'LATITUDE', 'LONGITUDE', 'ELEVATION'] 
    """  

    columns = ["STATION", "MXWDSP", "LATITUDE",  "LONGITUDE", "ELEVATION", "DATE"] 
    df = pd.read_parquet(path_to_weather_stations, columns=columns)
    df['STATION'] = df['STATION'].str.casefold()
    df["STATION"] = df["STATION"].astype("category")

    df['DATE'] = pd.DatetimeIndex(df['DATE'])
    df["DATE"] = pd.to_datetime((df["DATE"]), format="%Y-%m-%d")    

    df["MXWDSP"] = df["MXWDSP"].apply(pd.to_numeric, downcast="float")
    df = df.rename(columns={"DATE": "time",
                            "STATION": "station_name",
                            "MXWDSP": "max_speed",
                            "LATITUDE": "lat",
                            "LONGITUDE": "lon",
                            "ELEVATION": "height"
                            })    
    return df


def get_y(
        weather_stations_data: pd.DataFrame,
        start: str,
        end: str,
        target_column: str = "max_speed") -> pd.DataFrame:
    
    """Returns target variable for objects
    Args:
        start (str): start date. YYYY-MM-DD
        end (str): end date. YYYY-MM-DD
        speed_th (float, optional): threshold value for binary classification. Defaults to 20
        pd.DataFrame: {station_name: indicators of exceeding threshold}
    """      
    df = weather_stations_data
    df = df.loc[(df["time"] >= pd.to_datetime(start)) & (df["time"] <= pd.to_datetime(end))]
    df = df.rename(columns={target_column: "y"})    

    if "lat" in weather_stations_data.columns:    
        df = df[["time", "station_name", "y", "lat", "lon", "height" ]]
    else:
        df = df[["time", "station_name", "y" ]]

    return df


  
def closest_pixel_for_station(station_row, grid_lat, grid_lon):

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
    
    """maps stations to the data grid pixels
    """
    grid_lon = np.array(dataset_xarray.lon.data) 
    grid_lat = np.array(dataset_xarray.lat[::-1].data)
    closest_partial = partial(closest_pixel_for_station, grid_lat=grid_lat, grid_lon=grid_lon)
    
    start_time = time.process_time()
    stations_df['lon'], stations_df['lat'] = zip(*stations_df.apply(closest_partial, axis=1))
    logging.info(f"Closest pixel search took {time.process_time() - start_time} seconds")

    return stations_df    

def target_to_data_grid(
        dataset_xarray: xr.DataArray,
        target_df: pd.DataFrame) -> pd.DataFrame:
    
    """maps stations to the data grid pixels
    """
    grid_lon = np.array(dataset_xarray.lon.data) 
    grid_lat = np.array(dataset_xarray.lat[::-1].data)
    closest_partial = partial(closest_pixel_for_station, grid_lat=grid_lat, grid_lon=grid_lon)
    
    start_time = time.process_time()
    target_df['lon'], target_df['lat'] = zip(*target_df.apply(closest_partial, axis=1))
    logging.info(f"Closest pixel search took {time.process_time() - start_time} seconds")

    return target_df   
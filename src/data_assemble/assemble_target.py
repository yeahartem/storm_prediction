import sys,os
sys.path.append(os.getcwd())
import pandas as pd
import json
import numpy as np
from geopy.distance import great_circle
import logging
import xarray as xr
import time

def get_stations(all_stations_data='data_mounted/weather_stations/weatherstation_list.json',
                 stations_allowed_path="conf/splits/time_split_stations.txt",
                 max_lat=None, min_lat=None, max_lon=None, min_lon=None, max_height=None, min_height=None) -> pd.DataFrame:
    
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



def load_weatherstations_RU(path_to_weather_stations: str, stations_allowed: list) -> pd.DataFrame:
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
    df[["Максимальная скорость", "Средняя скорость ветра"]].apply(pd.to_numeric, downcast="float")
    df["Дата"] = pd.to_datetime((df["Дата"]), format="%Y/%m/%d")    
    # df = df.sort_values(by=['Название метеостанции', 'Дата'])
    df = df.rename(columns={"Дата": "time",
                            "Название метеостанции": "station_name",
                            "Максимальная скорость": "max_speed",
                            "Средняя скорость ветра": "mean_speed",
                            })
    
    df = df.loc[df["station_name"].isin(stations_allowed)]

    return df

def get_y(
        weather_stations_data: pd.DataFrame,
        start: str,
        end: str,
        target_column: str = "max_speed",
        speed_th: float = 20.0) -> pd.DataFrame:
    
    """Returns target variable for objects
    Args:
        start (str): start date. YYYY-MM-DD
        end (str): end date. YYYY-MM-DD
        speed_th (float, optional): threshold value for binary classification. Defaults to 20
        pd.DataFrame: {station_name: indicators of exceeding threshold}
    """    
    df = weather_stations_data
    df = df.loc[(df["time"] >= pd.to_datetime(start)) & (df["time"] <= pd.to_datetime(end))]
    # df = df.set_index("time")    
    # df['time'] = df["time"].astype('datetime64[D]')

    df["y"] = (df[target_column] > speed_th)       
    df = df[["time", "station_name", "y" ]]

    return df
  

def stations_to_data_grid(
        dataset_xarray: xr.DataArray,
        stations_df: pd.DataFrame) -> pd.DataFrame:
    
    """maps stations to the pixels
    Args:
        station_names (list): list of station names
        station_list (pd.DataFrame): pandas table with information about stations
    Returns:
        dict: {station_name: (pixel coords)}
    """
    grid_lon = np.array(dataset_xarray.lon.data) # varibles for local function scope
    grid_lat = np.array(dataset_xarray.lat[::-1].data)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    def closest_pixel_for_station(station_row):

        """Finds the closest pixel in the dataset for the station
        Returns:
        tuple: x, y lat and lon of the closest pixel on data grid
        """
        coord = [station_row["lat"], station_row["lon"]]        
        theta_offset = 5
        fi_offset = 5
        nearest_lat_idx = find_nearest(grid_lat, coord[0])
        nearest_lon_idx = find_nearest(grid_lon, coord[1])
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
    
    start_time = time.process_time()
    stations_df['lon'], stations_df['lat'] = zip(*stations_df.apply(closest_pixel_for_station, axis=1))
    logging.info(f"Closest pixel search took {time.process_time() - start_time} seconds")

    return stations_df    

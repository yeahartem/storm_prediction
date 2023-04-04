import json
import time
import logging

import numpy as np
from tqdm import tqdm
import pandas as pd
import xarray
from datetime import timedelta
import warnings
from geopy.distance import great_circle

warnings.filterwarnings("ignore")


def load_weatherstation_df(path_to_weather_stations):

    def cleanup_ms_name(name: str):
        name = name.replace('"', '')
        name = name.replace(',', '')
        return name

    columns = ["Название метеостанции", "Максимальная скорость", "Средняя скорость ветра", "Дата"]
    df = pd.read_parquet(path_to_weather_stations, columns=columns)
    df["Название метеостанции"] = df["Название метеостанции"].apply(cleanup_ms_name)
    df["Название метеостанции"] = df["Название метеостанции"].astype("category")
    df[["Максимальная скорость", "Средняя скорость ветра"]].apply(pd.to_numeric, downcast="float")
    df["Дата"] = pd.to_datetime((df["Дата"]), format="%Y/%m/%d")
    return df

def assemble_numpy_ds(
        blocks: xarray.DataArray, target: dict, stations_pixs: dict) -> tuple:
    """Assembles numpy dataset
        stacks all available data into nd tensor, tiles elevation
        matches target and objects on pixels of stations
    Args:
        blocks (xarray.DataArray): block data from climate model
        target (dict): target from stations. see `get_y` function
        stations_pixs (dict): pixels of stations
    Returns:
        tuple: X, y
    """
    X = {}
    y = {}
    X_s = {}
    y_s = {}

    for k in tqdm(target.keys()):
        station_pix = stations_pixs[k.casefold()]
        y_i = target[k].dropna()
        lat = find_nearest(blocks.lat, station_pix[0])
        lon = find_nearest(blocks.lon, station_pix[1])
        X_i = blocks[lat, lon]
        a = X_i.time.data
        b = y_i.index.values
        inters = np.intersect1d(a, b)
        X[k] = X_i.sel(time=inters)
        y[k] = y_i.loc[inters]

        target_df = pd.DataFrame({"values": y[k].values, "date": y[k].index})
        # now max over week
        y[k] = target_df.groupby([pd.Grouper(key='date', freq='4W')]).max()
        # stacking over weeks
        # rename dimensions, since stacked dim is added
        rename_dict = {d_name: d_name for d_name in X_i.dims}
        rename_dict['time'] = 'stacked'
        X_s_i = []
        for week_idx in range(len(y[k].index) - 1):
            week_slice = X_i.sel(time=slice(y[k].index[week_idx], y[k].index[week_idx + 1] - timedelta(days=1)))
            week_slice = week_slice.rename(rename_dict)
            if len(week_slice) > 0:
                week_slice = week_slice.assign_coords({"time": (week_slice['stacked'].data[0])})
                week_slice = week_slice.assign_coords({'stacked': range(len(week_slice.stacked))})
                if len(week_slice.stacked) == 7 * 4:  # attention week only!!!
                    X_s_i.append(week_slice)

        X_s[k] = xarray.concat(X_s_i, dim='time')
        correct_time_data = X_s[k].time.data  # nans free, nans avoided by if in line 89
        y_s[k] = y[k].loc[correct_time_data]
        assert len(X_s[k]) == len(y_s[k])

    return X_s, y_s


def get_y(
        weather_stations_data: pd.DataFrame,
        start: str,
        end: str,
        station_name_list: list,
        speed_th: float = 20.0,
        station_name: str = None,
) -> dict:
    """Returns target variable for objects
    Args:
        start (str): start date. YYYY-MM-DD
        end (str): end date. YYYY-MM-DD
        speed_th (float, optional): threshold value for binary classification. Defaults to 20..
        station_name (str, optional): name of the station in russian, not sensitive to case. Defaults to None.
    Returns:
        dict: {station_name: indicators of exceeding threshold}
    """

    df = weather_stations_data
    df_start_end = df.loc[
        (df["Дата"] >= pd.to_datetime(start)) & (df["Дата"] <= pd.to_datetime(end))
        ]

    df_start_end.set_index('Дата', inplace=True)
    df_start_end = df_start_end.loc[df_start_end["Название метеостанции"].isin(station_name_list)]

    df_start_end = df_start_end[
        ["Название метеостанции", "Максимальная скорость", "Средняя скорость ветра"]
    ]
    df_start_end["y"] = (
            np.maximum(
                df_start_end["Максимальная скорость"],
                df_start_end["Средняя скорость ветра"],
            )
            > speed_th
    )
    df_start_end.drop(
        columns=["Максимальная скорость", "Средняя скорость ветра"], inplace=True
    )
    if station_name is not None:
        grpb = df_start_end.groupby(df_start_end["Название метеостанции"], observed=True)
        assert (
                station_name in grpb.groups.keys()
        ), "No such station found. Available: " + "; ".join(list(grpb.groups.keys()))

        y = {station_name: grpb.get_group(station_name).y}  # Should be Applied groupby as below
    else:
        grpb = df_start_end.groupby(df_start_end["Название метеостанции"], observed=True)
        ks = grpb.groups.keys()
        y = {
            k: df_start_end.groupby(df_start_end["Название метеостанции"], observed=True)
            .get_group(k)
            .y.groupby(df_start_end.groupby(df_start_end["Название метеостанции"], observed=True).get_group(k).y.index,
                       observed=True).max()  # Group 8 days in 1
            for k in ks
        }
    return y


def get_pixel_stations(
        dataset,
        station_names: list,
        station_list: pd.DataFrame) -> dict:
    """maps stations to the pixels
    Args:
        station_names (list): list of station names
        station_list (pd.DataFrame): pandas table with information about stations
    Returns:
        dict: {station_name: (pixel coords)}
    """

    stations_pixs = {}
    for station_name in station_names:
        pix = closest_pixel_for_station(
            station_name=station_name, dataset=dataset[list(dataset.keys())[0]], station_list=station_list
        )
        stations_pixs[station_name.casefold()] = pix

    return stations_pixs

def closest_pixel_for_station(station_name, dataset, station_list):
    """Finds the closest pixel in the dataset for the station
    FIXED: NOT INDICES, BUT ABS VALUES
    Args:
      station_name (str): name of the station
      dataset (xarray.DataArray): dataset with pixels
      station_list (pd.DataFrame): table with stations' coordinates
    Returns:
      tuple: x, y indices among the dataset
    """
    station = station_list[station_list["Наименование станции"] == station_name]
    try:
        coord = [station["Широта"].values[0], station["Долгота"].values[0]]
    except IndexError:
        logging.debug(f'Coords not read {station_name}')
        return None

    lon = np.array(dataset.lon.data)
    lat = np.array(dataset.lat[::-1].data)
    theta_bias = 5
    fi_bias = 5
    nearest_lat_idx = find_nearest(lat, coord[0])
    nearest_lon_idx = find_nearest(lon, coord[1])
    closest = great_circle((lat[nearest_lat_idx], lon[nearest_lon_idx]), coord).kilometers

    for i, theta in enumerate(lon[nearest_lon_idx - 5:nearest_lon_idx + 5]):
        for j, fi in enumerate(lat[nearest_lat_idx - 5:nearest_lat_idx + 5]):
            r = great_circle((fi, theta), coord).kilometers
            if r < closest:
                closest = r
                theta_bias = i  # + 1
                fi_bias = j  # + 1
    closest_x_idx = theta_bias - 5 + nearest_lon_idx
    closest_y_idx = fi_bias - 5 + nearest_lat_idx

    logging.debug(f'{station_name} pixel found')
    return lon[closest_x_idx], lat[closest_y_idx]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def make_blocks_numpy_no_target(
        dataset_as_xarray: dict,
        half_side_size: int = 4,
        time_stack_size=1,  # only 1 for dataset with target
) -> xarray.DataArray:

    start_time = time.process_time()
    bands_list = []
    channels_stack = np.zeros(((8,) + dataset_as_xarray['Wind_'].shape))  # TODO channel number and base band from cfg
    for i, band in enumerate(dataset_as_xarray.keys()):
        if 'elev' in band:
            channels_stack[i] = np.repeat(dataset_as_xarray[band].to_numpy()[np.newaxis, :, :],
                                          channels_stack.shape[1], axis=0)
        else: channels_stack[i] = dataset_as_xarray[band]
        bands_list.append(band)
    channels_stack = np.moveaxis(channels_stack, 0, 1)
    windows = np.lib.stride_tricks.sliding_window_view(channels_stack,
                                                       (time_stack_size, 8, 2 * half_side_size + 1,
                                                        2 * half_side_size + 1))
    windows = np.squeeze(windows)
    windows = windows[::time_stack_size]
    windows = np.moveaxis(windows, 0, 2)
    time_coords = dataset_as_xarray['Wind_'].time.data[:-time_stack_size:time_stack_size]
    lat_coords = dataset_as_xarray['Wind_'].lat.data[half_side_size:-half_side_size]
    lon_coords = dataset_as_xarray['Wind_'].lon.data[half_side_size:-half_side_size]
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
    print(f"Numpy block preparation took {time.process_time() - start_time} seconds")

    return X.astype(np.float32)


def make_blocks_numpy(
        dataset_as_xarray: dict,
        half_side_size: int) -> xarray.DataArray:

    start_time = time.process_time()
    channels_stack = np.zeros(
        ((8,) + dataset_as_xarray['Wind_'].shape))  # TODO channel number and base band from cfg
    for i, band in enumerate(dataset_as_xarray.keys()):
        if 'elev' in band:
            channels_stack[i] = np.repeat(dataset_as_xarray[band].to_numpy()[np.newaxis, :, :],
                                          channels_stack.shape[1], axis=0)
        else: channels_stack[i] = dataset_as_xarray[band]

    channels_stack = np.moveaxis(channels_stack, 0, 1)
    windows = np.lib.stride_tricks.sliding_window_view(channels_stack,
                                                       (1, 8, 2 * half_side_size + 1,
                                                        2 * half_side_size + 1))
    windows = np.squeeze(windows)

    windows = np.moveaxis(windows, 0, 2)
    time_coords = dataset_as_xarray['Wind_'].time.data
    lat_coords = dataset_as_xarray['Wind_'].lat.data[half_side_size:-half_side_size]
    lon_coords = dataset_as_xarray['Wind_'].lon.data[half_side_size:-half_side_size]

    X = xarray.DataArray(
        windows,
        dims=["lat", "lon", "time", "channels", "window_lat", "window_lon"],
        coords={"lat": lat_coords,
                "lon": lon_coords,
                "time": time_coords,
                "channels": list(range(8)),
                "window_lat": list(range(2 * half_side_size + 1)),
                "window_lon": list(range(2 * half_side_size + 1))})
    print(f"Numpy block preparation took {time.process_time() - start_time} seconds")

    del dataset_as_xarray

    return X.astype(np.float32)

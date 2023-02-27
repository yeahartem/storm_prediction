import logging
import time

import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import os
from src.data_utils import data_processing as dp
import pandas as pd
import xarray
from datetime import timedelta
import warnings
from src.data_utils.utils import cleanup_ms_name

warnings.filterwarnings("ignore")


def assemble_numpy_ds(
        blocks: OrderedDict, target: dict, stations_pixs: dict, include_target: bool = True
) -> tuple:
    """Assembles numpy dataset
        stacks all available data into nd tensor, tiles elevation
        matches target and objects on pixels of stations
    Args:
        blocks (OrderedDict): block data from climate model, keys must be ordered
        target (dict): target from stations. see `get_y` function
        stations_pixs (dict): pixels of stations
        include_target: (bool): if to include target into dataset
    Returns:
        tuple: X, y
    """
    X = {}
    y = {}
    X_s = {}
    y_s = {}

    def assemble_channels(pix):
        channels_stack = np.zeros(((8,) + blocks['Wind_'][pix].shape))  # TODO make blocks from dict to tensor
        for i, band in enumerate(blocks.keys()):
            if 'elev' in band:
                channels_stack[i] = np.repeat(blocks[band][pix].to_numpy()[np.newaxis, :, :],
                                              channels_stack.shape[1],
                                              axis=0)
            else:
                channels_stack[i] = blocks[band][pix]

        return channels_stack

    if include_target:
        for k in tqdm(target.keys()):
            station_pix = stations_pixs[k.casefold()]
            X_i = xarray.DataArray(
                assemble_channels(station_pix),
                dims=["channels", "time", "lat", "lon"],
                coords={"channels": list(range(8)),
                        "time": blocks['Wind_'][station_pix].time.data,
                        "lat": blocks['Wind_'][station_pix].lat.data,
                        "lon": blocks['Wind_'][station_pix].lon.data,
                        }
            )

            if len(X_i) > 0:
                y_i = target[k]
                X_i = X_i.transpose('time', 'channels', 'lat', 'lon')
                a = X_i.time.data
                b = y_i.index.values
                inters = np.intersect1d(a, b)
                X_i = X_i.loc[inters]
                X[k] = X_i
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
    else:
        some_key = list(blocks.keys())[0]
        for curr_pix in tqdm(blocks[some_key].keys()):
            X_i = []
            for fn in blocks.keys():
                X_i.append(blocks[fn][curr_pix])

            if len(X_i) > 0:
                X_i.sort(key=lambda x: x.shape[0])
                for X_i_idx in range(len(X_i)):
                    if len(X_i[X_i_idx].shape) == 2:
                        X_i[X_i_idx] = xarray.concat([X_i[X_i_idx] for _ in range(X_i[-1].shape[0])],
                                                     'time')  # repeating elevation
                        X_i[X_i_idx] = X_i[X_i_idx].assign_coords({'time': X_i[-1].time})
                X_i = xarray.concat(X_i, "channels").transpose('time', 'channels', 'lat', 'lon')

                dates = X_i.time.data
                target_df = pd.DataFrame({"values": np.zeros(len(dates)), "date": dates})
                # dummy for getting week indices
                y_dummy = target_df.groupby([pd.Grouper(key='date', freq='4W')]).max()

                # stacking over weeks
                # rename dimensions, since stacked dim is added
                rename_dict = {d_name: d_name for d_name in X_i.dims}
                rename_dict['time'] = 'stacked'
                X_s_i = []
                for week_idx in range(len(y_dummy.index) - 1):
                    week_slice = X_i.sel(
                        time=slice(y_dummy.index[week_idx], y_dummy.index[week_idx + 1] - timedelta(days=1)))
                    week_slice = week_slice.rename(rename_dict)
                    if len(week_slice) > 0:
                        week_slice = week_slice.assign_coords({"time": (week_slice['stacked'].data[0])})
                        week_slice = week_slice.assign_coords({'stacked': range(len(week_slice.stacked))})
                        if len(week_slice.stacked) == 7 * 4:  # attention week only!!!
                            X_s_i.append(week_slice)

                X_s[curr_pix] = xarray.concat(X_s_i, dim='time')

    if include_target:
        return X_s, y_s
    else:
        return X_s,


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
        path_to_weather_stations (str): path to weather stations
        start (str): start date. YYYY-MM-DD
        end (str): end date. YYYY-MM-DD
        speed_th (float, optional): threshold value for binary classification. Defaults to 20..
        station_name (str, optional): name of the station in russian, not sensitive to case. Defaults to None.
    Returns:
        dict: {station_name: indicators of exceeding threshold}
    """

    # df["Дата"] = pd.to_datetime((df["Дата"]), format="%Y/%m/%d")
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
        pix = dp.closest_pixel_for_station(
            station_name=station_name, dataset=dataset['Wind_'], station_list=station_list
        )
        stations_pixs[station_name.casefold()] = pix

    return stations_pixs


def make_blocks(
        dataset_as_xarray: dict,
        half_side_size: int = 4,
) -> OrderedDict:
    """slices blocks from data
    Args:
        filter_dict - things which should be included into the titles of files:
         {"years": ['2006', '2026'], "bands": ['max', 'Wind_']}
        rectangle_coords - border coordinates of the chosen territory:
         {'lat_min': 41.12, 'lat_max': 81.49,'lon_min': 19.38, 'lon_max': 169.40},
        target_res - {'lon_res': 0.25, 'lat_res': 0.25},
        half_side_size (int, optional) - square block half size. Defaults to 4.
    Returns:
        dict: {`block center pixel`: surrounding 3d tensor}
    """
    start_time = time.process_time()

    slices_dict = {k: {} for k in dataset_as_xarray.keys()}  # key = center of block
    for k in tqdm(dataset_as_xarray.keys()):
        np_ = dataset_as_xarray[k]
        for i in range(half_side_size, len(np_.lat.data) - half_side_size):
            for j in range(half_side_size, len(np_.lon.data) - half_side_size):
                slices_dict[k][(i, j)] = np_.sel(
                    lat=slice(np_.lat.data[i - half_side_size], np_.lat.data[i + half_side_size]),
                    lon=slice(np_.lon.data[j - half_side_size], np_.lon.data[j + half_side_size])).astype(np.float32)
    slices_dict = OrderedDict(sorted(slices_dict.items()))

    print(f"Block preparation took {time.process_time() - start_time} seconds")

    start_time = time.process_time()
    channels_stack = np.zeros(((8,) + dataset_as_xarray['Wind_'].shape))  # TODO channel number and base band from cfg
    for i, band in enumerate(dataset_as_xarray.keys()):
        if 'elev' in band:
            channels_stack[i] = np.repeat(dataset_as_xarray[band].to_numpy()[np.newaxis, :, :],
                                          channels_stack.shape[1],
                                          axis=0)
        else:
            channels_stack[i] = dataset_as_xarray[band]

    windows = np.lib.stride_tricks.sliding_window_view(channels_stack,
                                                       (8, 28, 2 * half_side_size + 1, 2 * half_side_size + 1))

    time_coords = dataset_as_xarray['Wind_'].time.data[28:-28:28]
    lat_coords = dataset_as_xarray['Wind_'].lat.data[2 * half_side_size:-2 * half_side_size]
    lon_coords = dataset_as_xarray['Wind_'].lon.data[2 * half_side_size:-2 * half_side_size]

    X_i = xarray.DataArray(
        windows,
        dims=["channels", "time", "lat", "lon"],
        coords={"channels": list(range(8)),
                "time": time_coords,
                "lat": lat_coords,
                "lon": lon_coords,
                }
    )
    print(f"Numpy block preparation took {time.process_time() - start_time} seconds")

    return slices_dict


def make_blocks_no_target(
        dataset_as_xarray: dict,
        half_side_size: int = 4,
) -> xarray.DataArray:
    """slices blocks from data
    Args:
        filter_dict - things which should be included into the titles of files:
         {"years": ['2006', '2026'], "bands": ['max', 'Wind_']}
        rectangle_coords - border coordinates of the chosen territory:
         {'lat_min': 41.12, 'lat_max': 81.49,'lon_min': 19.38, 'lon_max': 169.40},
        target_res - {'lon_res': 0.25, 'lat_res': 0.25},
        half_side_size (int, optional) - square block half size. Defaults to 4.
    Returns:
        dict: {`block center pixel`: surrounding 3d tensor}
    """
    start_time = time.process_time()
    channels_stack = np.zeros(((8,) + dataset_as_xarray['Wind_'].shape))  # TODO channel number and base band from cfg
    for i, band in enumerate(dataset_as_xarray.keys()):
        if 'elev' in band:
            channels_stack[i] = np.repeat(dataset_as_xarray[band].to_numpy()[np.newaxis, :, :],
                                          channels_stack.shape[1],
                                          axis=0)
        else:
            channels_stack[i] = dataset_as_xarray[band]

    channels_stack = np.moveaxis(channels_stack, 0, 1)
    windows = np.lib.stride_tricks.sliding_window_view(channels_stack,
                                                       (28, 8, 2 * half_side_size + 1, 2 * half_side_size + 1))
    windows = np.squeeze(windows)
    windows = windows[::28]
    windows = np.moveaxis(windows, 0, 2)
    time_coords = dataset_as_xarray['Wind_'].time.data[:-28:28]
    lat_coords = dataset_as_xarray['Wind_'].lat.data[half_side_size:-half_side_size]
    lon_coords = dataset_as_xarray['Wind_'].lon.data[half_side_size:-half_side_size]

    X = xarray.DataArray(
        windows,
        dims=["lat", "lon", "time", "time_stack", "channels", "window_lat", "window_lon"],
        coords={"lat": lat_coords,
                "lon": lon_coords,
                "time": time_coords,
                "time_stack": list(range(28)),
                "channels": list(range(8)),
                "window_lat": list(range(2*half_side_size+1)),
                "window_lon": list(range(2*half_side_size+1))
                }
    )
    print(f"Numpy block preparation took {time.process_time() - start_time} seconds")

    return X.astype(np.float32)

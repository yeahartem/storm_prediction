import grp
from typing import Dict

from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import os
from src.data_utils import data_processing as dp

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import random
import pandas as pd
import xarray

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import warnings
import pickle

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
    
    if include_target:
        for k in tqdm(target.keys()):
            X_i = []
            for fn in blocks.keys():
                curr_pix = stations_pixs[k.casefold()]
                if curr_pix in blocks[fn].keys():
                    X_i.append(blocks[fn][curr_pix])
            if len(X_i) > 0:
                y_i = target[k]
                X_i.sort(key=lambda x: x.shape[0])
                for X_i_idx in range(len(X_i)):
                    if len(X_i[X_i_idx].shape) == 2:
                        X_i[X_i_idx] = xarray.concat([X_i[X_i_idx] for _ in range(X_i[-1].shape[0])], 'time')  # repeating elevation
                        X_i[X_i_idx] = X_i[X_i_idx].assign_coords({'time': X_i[-1].time})
                        
                X_i = xarray.concat(X_i, "channels").transpose('time', 'channels', 'lat', 'lon')

                inters = y_i.index.intersection(X_i.time.data) # !!!!!!!!!! Accidentally may be elevation        

                X_i = X_i.loc[inters]     

                X[k] = X_i

                y[k] = y_i.loc[inters]

                X_s_i = []
                target_ind = []
                for idx, (x_day, y_day) in enumerate(zip(X[k], y[k])):
                    if 3 < idx < ((len(X[k])) - 3):
                        x_stacked = xarray.concat(X[k].loc[X[k]['time'][idx - 3:idx + 4]], dim='stack').assign_coords({'time': X[k]['time'][idx]})
                        X_s_i.append(x_stacked)
                        target_ind.append(X[k]['time'][idx].values)
                    else:
                        continue
                X_s[k] = xarray.concat(X_s_i, "time")
                y_s[k] = y[k].loc[target_ind]
                assert len(X_s[k]) == len(y_s[k])
    else:
        some_key = list(blocks.keys())[0]
        for curr_pix in blocks[some_key].keys():
            X_i = []
            for fn in blocks.keys():

                X_i.append(blocks[fn][curr_pix])
                
            if len(X_i) > 0:
                X_i.sort(key=lambda x: x.shape[0])
                for X_i_idx in range(len(X_i)):
                    if len(X_i[X_i_idx].shape) == 2:
                        X_i[X_i_idx] = xarray.concat([X_i[X_i_idx] for _ in range(X_i[-1].shape[0])], 'time')  # repeating elevation
                        X_i[X_i_idx] = X_i[X_i_idx].assign_coords({'time': X_i[-1].time})       
                X_i = xarray.concat(X_i, "channels").transpose('time', 'channels', 'lat', 'lon')
                
                X[curr_pix] = X_i

    if include_target:
        return (X_s, y_s)
    else:
        return X


def get_y(
    df: pd.DataFrame,
    start: str,
    end: str,
    station_name_list: list,
    speed_th: float = 20.0,
    station_name: str = None,
) -> dict:
    """Returns target variable for objects

    Args:
        df (pd.DataFrame): weather stations data
        start (str): start date. YYYY-MM-DD
        end (str): end date. YYYY-MM-DD
        speed_th (float, optional): threshold value for binary classification. Defaults to 20..
        station_name (str, optional): name of the station in russian, not sensitive to case. Defaults to None.

    Returns:
        dict: {station_name: indicators of exceeding threshold}
    """
    
    df["Дата"] = pd.to_datetime((df["Дата"]), format="%Y/%m/%d")
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
        grpb = df_start_end.groupby(df_start_end["Название метеостанции"])
        assert (
            station_name in grpb.groups.keys()
        ), "No such station found. Available: " + "; ".join(list(grpb.groups.keys()))

        y = {station_name: grpb.get_group(station_name).y} # Should be Applied groupby as below
    else:
        grpb = df_start_end.groupby(df_start_end["Название метеостанции"])
        ks = grpb.groups.keys()

        y = {
            k: df_start_end.groupby(df_start_end["Название метеостанции"])
            .get_group(k)
            .y.groupby(df_start_end.groupby(df_start_end["Название метеостанции"]).get_group(k).y.index).max() #Group 8 days in 1
            for k in ks
        }

    return y


def get_pixel_stations(
    path_to_files: str,
    filter_dict: dict,
    station_names: list,
    station_list: pd.DataFrame,
    rectangle_coords: dict,
    target_res: dict
) -> dict:
    """maps stations to the pixels in .tifs

    Args:
        path_to_tifs (list): path to tif files to get sample dataset - for shape
        feature_names (list): feature names - to get sample dataset    = {"year": ['2006'], "band": ['max']}
        station_names (list): list of station names
        station_list (pd.DataFrame): pandas table with information about stations

    Returns:
        dict: {station_name: (pixel coords)}
    """
    if path_to_files.endswith('.tif'): 
        raise NotImplementedError('.tif files are not implemented yet, it will be done later. Please use .nc files.')

    elif path_to_files.endswith('.nc'):
        file_paths = path_to_files[:-4]
        bnds = list(filter(lambda x: x!='elevation', filter_dict['bands']))
        filtered_dict = {"years": filter_dict["years"], "bands": bnds}
        dataset = dp.get_xarrays(file_paths, rectangle_coords, target_res, filtered_dict)
    
    stations_pixs = {}
    for station_name in station_names:

        pix = dp.closest_pixel_for_station(
            station_name=station_name, dataset=dataset[bnds[0]], station_list=station_list
        )
        stations_pixs[station_name.casefold()] = pix
    return stations_pixs



def make_blocks(
    path_to_files: list,
    filter_dict: dict,
    rectangle_coords: dict,
    target_res: dict,
    half_side_size: int = 4,
    verbose: bool = False,
    time_limits: dict = {'t_start': np.datetime64('2005-01-01'), 't_end': np.datetime64('2020-01-01')},
) -> OrderedDict:
    """slices blocks from data

    Args:
        path_to_files - path to all files: ['../data/elev/*.tif', '../data/stash/WindProject/cmip_stash/*.nc'],
        filter_dict - things which should be included into the titles of files: {"years": ['2006', '2026'], "bands": ['max', 'Wind_']}
        rectangle_coords - border coordinates of the chosen territory: {'lat_min': 41.12, 'lat_max': 81.49,'lon_min': 19.38, 'lon_max': 169.40},
        target_res - {'lon_res': 0.25, 'lat_res': 0.25},
        half_side_size (int, optional) - square block half size. Defaults to 4.

    Returns:
        dict: {`block center pixel`: surrounding 3d tensor}
    """
    bands = {}
    for path_of_file in path_to_files:
        if 'elevation' not in path_of_file:
            path_of_file = path_of_file[:-4]
            if 'elevation' in filter_dict["bands"]:
                filter_dict["bands"].remove('elevation')
                bands = {**bands, **dp.get_xarrays(path_of_file, rectangle_coords, target_res, filter_dict, time_limits)} 
                cmip_xarray = dp.get_xarrays(path_of_file, rectangle_coords, target_res, filter_dict, time_limits)['Wind_']
                filter_dict["bands"].append('elevation')
            else: 
                bands = {**bands, **dp.get_xarrays(path_of_file, rectangle_coords, target_res, filter_dict, time_limits)} 
                cmip_xarray = dp.get_xarrays(path_of_file, rectangle_coords, target_res, filter_dict, time_limits)['Wind_']
     
    for path_of_file in path_to_files:        
        if 'elevation' in path_of_file:
            bands = {**bands, **dp.get_xarrays(path_of_file, rectangle_coords, 
                                               target_res, {"years": [filter_dict["years"][0]], "bands": ['elevation']},
                                               time_limits, cmip_xarray=cmip_xarray)}

    slices_dict = {k: {} for k in bands.keys()}  # key = center of block
    for k in tqdm(bands.keys()):
        np_ = bands[k]
        for i in range(half_side_size, len(np_.lat.data) - half_side_size):
            for j in range(half_side_size, len(np_.lon.data) - half_side_size):
                slices_dict[k][(i, j)] = np_.sel(lat=slice(np_.lat.data[i - half_side_size], np_.lat.data[i + half_side_size]),
                                                 lon=slice(np_.lon.data[j - half_side_size], np_.lon.data[j + half_side_size]))
    slices_dict = OrderedDict(sorted(slices_dict.items()))
    return slices_dict
    
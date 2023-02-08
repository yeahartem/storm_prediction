# from tkinter import Y
import logging
import sys

sys.path.append('..')
from src.data_utils.utils import check_leap_year

import matplotlib.pyplot as plt
import os, glob
import numpy as np
from tqdm import tqdm
import subprocess
import pandas as pd
from geopy.distance import great_circle
from sklearn import preprocessing
from scipy import interpolate
import xarray
import copy

# from geopy.distance import geodesic
from math import sin, cos, sqrt, atan2, radians


def get_xarrays(path_to_data: str, rectangle_coords: dict, target_res: dict,
                contains: dict = {"years": ['2006'], "bands": ['max', 'elevation']},
                time_limits: dict = {'t_start': np.datetime64('2005-01-01'), 't_end': np.datetime64('2020-01-01')},
                cmip_xarray: xarray.DataArray = None) -> dict:
    '''
    Returns reduced to rectangle_coords xarrays.

    rectangle_coords - {'lat_min': 41.12, 'lat_max': 81.49,'lon_min': 19.38, 'lon_max': 169.40},
    target_res - {'lon_res': 0.25, 'lat_res': 0.25},
    contains - things which should be included into the titles of .nc files: {"years": ['2016', '2026'], "bands": ['max', 'Wind_']}
    '''
    # unpack required geometry, target resolutions
    lat_min = rectangle_coords['lat_min']
    lat_max = rectangle_coords['lat_max']
    lon_max = rectangle_coords['lon_max']
    lon_min = rectangle_coords['lon_min']
    lat_res = target_res['lat_res']
    lon_res = target_res['lon_res']
    t_start = time_limits["t_start"]
    t_end = time_limits["t_end"]

    xarrays = {}
    for band in contains["bands"]:
        if band == 'elevation':
            print(band)
            f1_xarray = open_dataxarray(path_to_data, [contains["years"][0], band]).astype(np.float16)
            f1_xarray = reduce_to_area(f1_xarray, lat_min, lat_max, lon_min, lon_max)
            f1_xarray_refined = res_incr(X_elev=f1_xarray.lon.data, Y_elev=f1_xarray.lat.data,
                                         X_cmip=cmip_xarray.lon.data, Y_cmip=cmip_xarray.lat.data,
                                         elev_data=f1_xarray.data,
                                         etalon_data=np.zeros(cmip_xarray[0, :, :].data.shape),
                                         etalon=cmip_xarray[0, :, :].copy())
            for num, agr in enumerate(f1_xarray_refined.agregation.data):
                etalon = cmip_xarray[0, :, :].copy()
                etalon.data = f1_xarray_refined.data[num]
                xarrays[agr] = etalon
        else:
            band_year = []
            for year in contains["years"]:
                print(band, year)
                f1_xarray = open_dataxarray(path_to_data, [year, band]).astype(np.float16)
                f1_xarray = reduce_to_area(f1_xarray, lat_min, lat_max, lon_min, lon_max)
                f1_xarray_refined = interp_timewise_xarray(f1_xarray, lon_res=lon_res,
                                                           lat_res=lat_res, interp_method='linear',
                                                           plot_example=False)
                # del f1_xarray
                band_year.append(f1_xarray_refined)
            band_year_xarray = xarray.concat(band_year, dim="time")  # stack xarrays on time axis
            # del band_year
            band_year_xarray = band_year_xarray.sel(time=slice(t_start, t_end))
            xarrays[band] = band_year_xarray
    return xarrays


def open_dataxarray(path_to_data: str, contains: list = ['2006', 'max']) -> xarray.DataArray:
    '''
    Converts .nc files to xarrays. 29 of February are excluded as in the .nc files.
    '''
    # path_to_data = os.path.join('..', 'data', 'stash', 'WindProject', 'cmip_stash')
    # path_to_data = '../../../data/cmip_stash/elevation/elevation.nc'

    if 'elevation' in path_to_data:
        f1 = xarray.load_dataset(path_to_data, decode_times=False).astype(np.float16)
        f1_xarray = f1.to_array()
        f1_xarray = f1_xarray.reindex(Y=list(reversed(f1_xarray.Y)))
        f1_xarray = f1_xarray.roll(X=21600, roll_coords=True)
        f1_xarray = f1_xarray.assign_coords(X=[x if x > 0 else x + 360 for x in f1_xarray.X.values])
        f1_xarray = f1_xarray.rename({'X': 'lon', 'Y': 'lat'})
        np.nan_to_num(f1_xarray, copy=False)

    else:
        ncs = np.array(sorted(os.listdir(path_to_data)))
        ncs_filtered = [nc for nc in ncs if np.prod([cond in nc for cond in contains])]

        f1 = xarray.load_dataset(os.path.join(path_to_data, ncs_filtered[0]), decode_times=False)
        # Exclude 29 of February from date_range()
        first_time = pd.date_range(start=pd.to_datetime(ncs_filtered[0][-20:-12]), periods=f1.sizes['time'])
        t0 = np.apply_along_axis(check_leap_year, axis=0, arr=first_time)
        years_unique = np.unique(first_time[t0].year)
        second_time = pd.date_range(start=pd.to_datetime(ncs_filtered[0][-20:-12]),
                                    periods=f1.sizes['time'] + len(years_unique))
        for yea in years_unique:
            second_time = second_time.drop(pd.to_datetime(str(yea) + '-02-29'))
        f1['time'] = second_time
        f1_xarray = f1.to_array()
        # assert bool(np.prod([str(y) + '-02-29' in f1_xarray.time.loc[t0].data.astype('datetime64[D]').astype('str') for y in years_unique])), "no 29 feb on leap years"

    return f1_xarray


def reduce_to_area(data_arr: xarray.DataArray, lat_min: float = 24, lat_max: float = 31, lon_min: float = 272,
                   lon_max: float = 280) -> xarray.DataArray:
    """
    Reduces the area to the input frames +-1.5 on latitude axis and +-2 on longitude axis.
    """
    band_name = [v for v in dict(data_arr.coords)['variable'].data if 'bnds' not in v][0]
    output = data_arr.sel(lat=slice(lat_min - 1.5, lat_max + 1.5), lon=slice(lon_min - 2, lon_max + 2),
                          variable=band_name)
    output.data = np.float64(output.data)
    if band_name != 'topo':
        assert np.allclose(output.data[:, 0, :, :],
                           output.data[:, 1, :, :]), "`bnds` dim components of tensor are not identical"
    return output


def interp_timewise(data_arr: xarray.DataArray, lon_res: float = 0.25, lat_res: float = 0.25,
                    interp_method: str = 'linear', plot_example: bool = False) -> tuple:
    assert np.allclose(data_arr.data[:, 0, :, :],
                       data_arr.data[:, 1, :, :]), "2d components of tensor are not identical"
    timesteps = len(data_arr.time.data)
    plot_example_dice = np.random.randint(0, timesteps)
    znews = []
    x = data_arr.lon.data
    y = data_arr.lat.data
    xnew = np.arange(x[0], x[-1], lon_res)
    ynew = np.arange(y[0], y[-1], lat_res)
    for t in tqdm(range(timesteps)):
        # xx, yy = np.meshgrid(x, y)
        z = data_arr.data[t, 0, :, :]
        f = interpolate.interp2d(x, y, z, kind=interp_method)

        znew = f(xnew, ynew)
        znews.append(znew)
        if plot_example and t == plot_example_dice:
            plt.figure()
            plt.title('before')
            plt.imshow(z)
            plt.show()
            plt.figure()
            plt.title('after')
            plt.imshow(znew)
            plt.show()
    output = np.stack(znews)
    return (output, xnew, ynew)


def res_incr(X_elev, Y_elev, X_cmip, Y_cmip,
             elev_data, etalon_data, etalon,
             func_list=[np.mean, np.max, np.min, np.std]):
    data = []
    for func in func_list:
        x = 0
        y = 0
        k_start = 0
        i_start = 0
        for k in range(len(Y_elev)):
            if y < len(Y_cmip) and Y_elev[k] > Y_cmip[y]:
                k_end = k
                for i in range(len(X_elev)):
                    if x < len(X_cmip) and X_elev[i] > X_cmip[x] and i > 0:
                        i_end = i
                        val = func(elev_data[k_start:k_end, i_start:i_end])
                        etalon_data[y, x] = val
                        x += 1
                        i_start = i
                x = 0
                y += 1
                k_start = k
                i_start = 0
        etalon.data = etalon_data
        data.append(copy.deepcopy(etalon))

    name_list = ['elev_' + f.__name__ for f in func_list]
    xarray_refined = xarray.concat(data, pd.Index(name_list, name='agregation'))

    return xarray_refined


def interp_timewise_xarray(data_arr: xarray.DataArray, lon_res: float = 0.25, lat_res: float = 0.25,
                           interp_method: str = 'linear', plot_example: bool = False) -> xarray.DataArray:
    data_new, xnew, ynew = interp_timewise(data_arr, interp_method=interp_method, plot_example=plot_example)
    data_arr.coords
    coords_new = {'lon': ("lon", xnew), 'lat': ("lat", ynew), 'time': ("time", data_arr.time.data)}
    output = xarray.DataArray(data=data_new, coords=coords_new, dims=('time', 'lat', 'lon'), attrs=data_arr.attrs)

    return output


def get_file_paths(
        path_to_data: str = "drive/MyDrive/Belgorodskaya/*.tif",
        feature_names: list = ["tmax", "tmin", "pr"],
):
    """
    Filters out required features amongs terraclim dataset
    Arguments:
      path_to_data (str): path to directory that containts terraclim dataset
      feature_names (list): list of required features
    Returns:
      dict: key -- feature name; value -- list of related tif files
    """
    files_to_mosaic = glob.glob(path_to_data)
    files_to_mosaic = list(
        filter(lambda x: sum(fn in x for fn in feature_names) > 0, files_to_mosaic)
    )
    file_paths = {
        fn: list(filter(lambda x: fn in x, files_to_mosaic)) for fn in feature_names
    }
    return file_paths


def get_closest_pixel(dataset, coord):  # change gdal.Dataset to xarray
    """Finds the closest pixel indices in the dataset
  Args:
      dataset (gdal.Dataset): dataset with pixels
      coord (np.ndarray): coordinate for which the closest pixel's indices in the dataset will be found
  Returns:
      tuple: x, y indices among the dataset
  """

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx

    lon = np.array(dataset.lon)
    lat = np.array(dataset.lat[::-1])
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
    return closest_x_idx, closest_y_idx


def closest_pixel_for_station(station_name, dataset, station_list):  # Change gdal.Dataset to xarray
    """Finds the closest pixel indices in the dataset for the station
    Args:
      station_name (str): name of the station
      dataset (gdal.Dataset): dataset with pixels
      station_list (pd.DataFrame): table with stations' coordinates
    Returns:
      tuple: x, y indices among the dataset
    """
    station = station_list[station_list["Наименование станции"] == station_name]
    try:
        coord = [station["Широта"].values[0], station["Долгота"].values[0]]
    except IndexError:
        logging.debug(station_name)

    pix = get_closest_pixel(dataset=dataset, coord=coord)
    logging.debug(f'{station_name} in progress')
    return pix


def leap_years(ar, leap_idx):
    """
    Inserts 29 of February in a 3d-array.
    leap_idx - indices of 29 of February in descending order.
    """
    for i in leap_idx:
        feb = (ar[i] + ar[i + 1]) / 2
        feb = np.expand_dims(feb, axis=0)
        ar = np.vstack((ar[:i + 1], feb, ar[i + 1:]))

    return ar

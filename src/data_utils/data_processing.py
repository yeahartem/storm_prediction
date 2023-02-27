import logging
import sys

sys.path.append('..')
from src.data_utils.utils import check_leap_year
import glob
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from geopy.distance import great_circle
from scipy import interpolate
import xarray
import copy
import time


def filter_cmip_files(all_cmip_files, time_limits, band):
    band_cmip_files = []
    for file in all_cmip_files:
        start = np.datetime64(pd.to_datetime(file[-20:-12]))
        end = np.datetime64(pd.to_datetime(file[-11:-3]))
        if (end > time_limits[0]) and (start < time_limits[1]):
            if band in file.split('/')[-1]:
                band_cmip_files.append(file)
    return band_cmip_files


def get_xarrays(path_to_cmip_folder: str, rectangle_coords: dict, target_res: dict,
                time_limits: (np.datetime64, np.datetime64), bands: list) -> dict:
    """
    Returns reduced to rectangle_coords xarrays EXCEPT elevation.
    rectangle_coords - {'lat_min': 41.12, 'lat_max': 81.49,'lon_min': 19.38, 'lon_max': 169.40}, target_res - {
    'lon_res': 0.25, 'lat_res': 0.25}, contains - things which should be included into the titles of .nc files: {
    "years": ['2016', '2026'], "bands": ['max', 'Wind_']}
    """
    lat_res = target_res['lat_res']
    lon_res = target_res['lon_res']
    xarrays = {}
    start_time = time.process_time()

    all_cmip_files = [os.path.join(path_to_cmip_folder, fn) for fn in next(os.walk(path_to_cmip_folder))[2]]
    for band in bands:
        band_year = []
        cmip_files = filter_cmip_files(all_cmip_files, time_limits, band)
        print(f'Will be loaded {cmip_files}')
        for file in cmip_files:
            logging.info(f'''Loading {band} {file.split('/')[-1]}''')
            f1_xarray = open_cmip_as_xarray(file)
            f1_xarray = reduce_to_area(f1_xarray, rectangle_coords)
            f1_xarray_refined = interp_timewise_xarray(f1_xarray, lon_res=lon_res,
                                                       lat_res=lat_res, interp_method='linear',
                                                       plot_example=False)
            band_year.append(f1_xarray_refined)
            band_year_xarray = xarray.concat(band_year, dim="time")  # stack xarrays on time axis
            band_year_xarray = band_year_xarray.sel(time=slice(time_limits[0],
                                                               time_limits[1]))
            xarrays[band] = band_year_xarray

    logging.info(f"All xarrays preparation took {time.process_time() - start_time} seconds")

    return xarrays


def get_xarrays_elevation(path_to_data: str, rectangle_coords: dict,
                          reference_xarray: xarray.DataArray = None) -> dict:
    """
    Returns reduced to rectangle_coords elevation xarrays.
   target_res - {'lon_res': 0.25, 'lat_res': 0.25}, contains
    - things which should be included into the titles of .nc files: {
    """
    xarrays = {}
    start_time = time.process_time()
    band = 'elevation'
    logging.info(band)
    f1_xarray = open_elevation_as_xarray(path_to_data)
    f1_xarray = reduce_to_area(f1_xarray, rectangle_coords)
    f1_xarray_refined = res_incr(X_elev=f1_xarray.lon.data, Y_elev=f1_xarray.lat.data,
                                 X_cmip=reference_xarray.lon.data, Y_cmip=reference_xarray.lat.data,
                                 elev_data=f1_xarray.data,
                                 etalon_data=np.zeros(reference_xarray[0, :, :].data.shape),
                                 etalon=reference_xarray[0, :, :].copy())
    for num, agr in enumerate(f1_xarray_refined.agregation.data):
        etalon = reference_xarray[0, :, :].copy()
        etalon.data = f1_xarray_refined.data[num]
        xarrays[agr] = etalon

    logging.info(f"Elevation preparation took {time.process_time() - start_time} seconds")
    return xarrays


def open_cmip_as_xarray(path_to_file: str) -> xarray.DataArray:
    """
    Converts .nc files to xarrays. 29 of February are excluded as in the .nc files.
    """

    f1 = xarray.open_dataset(path_to_file, decode_times=False).astype(np.float32)
    # Exclude 29 of February from date_range()
    first_time = pd.date_range(start=pd.to_datetime(path_to_file[-20:-12]), periods=f1.sizes['time'])
    t0 = np.apply_along_axis(check_leap_year, axis=0, arr=first_time)
    years_unique = np.unique(first_time[t0].year)
    second_time = pd.date_range(start=pd.to_datetime(path_to_file[-20:-12]),
                                periods=f1.sizes['time'] + len(years_unique))
    for yea in years_unique:
        second_time = second_time.drop(pd.to_datetime(str(yea) + '-02-29'))
    f1['time'] = second_time
    f1_xarray = f1.to_array()

    return f1_xarray


def open_elevation_as_xarray(path_to_file: str) -> xarray.DataArray:
    """
    Converts .nc files to xarrays. 29 of February are excluded as in the .nc files.
    """
    f1 = xarray.open_dataset(path_to_file, decode_times=False).astype(np.float32)
    f1_xarray = f1.to_array()
    f1_xarray = f1_xarray.reindex(Y=list(reversed(f1_xarray.Y)))
    f1_xarray = f1_xarray.roll(X=21600, roll_coords=True)
    f1_xarray = f1_xarray.assign_coords(X=[x if x > 0 else x + 360 for x in f1_xarray.X.values])
    f1_xarray = f1_xarray.rename({'X': 'lon', 'Y': 'lat'})
    np.nan_to_num(f1_xarray, copy=False)

    return f1_xarray


def reduce_to_area(data_arr: xarray.DataArray, rectangle_coords: dict) -> xarray.DataArray:
    """
    Reduces the area to the input frames +-1.5 on latitude axis and +-2 on longitude axis.
    """
    lat_min = rectangle_coords['lat_min'] - 1.5 - 0.25 * 5  # TODO get numbers from config
    lat_max = rectangle_coords['lat_max'] + 1.5 + 0.25 * 5
    lon_max = rectangle_coords['lon_max'] + 2 + 0.25 * 5
    lon_min = rectangle_coords['lon_min'] - 2 - 0.25 * 5

    band_name = [v for v in dict(data_arr.coords)['variable'].data if 'bnds' not in v][0]
    output = data_arr.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max),
                          variable=band_name)
    # output.data = np.float32(output.data)
    if band_name != 'topo':
        assert np.allclose(output.data[:, 0, :, :],
                           output.data[:, 1, :, :]), "`bnds` dim components of tensor are not identical"
    return output


def old_reduce_to_area(data_arr: xarray.DataArray, lat_min: float = 24, lat_max: float = 31, lon_min: float = 272,
                       lon_max: float = 280) -> xarray.DataArray:
    """
    Reduces the area to the input frames +-1.5 on latitude axis and +-2 on longitude axis.
    """
    band_name = [v for v in dict(data_arr.coords)['variable'].data if 'bnds' not in v][0]
    output = data_arr.sel(lat=slice(lat_min - 1.5, lat_max + 1.5), lon=slice(lon_min - 2, lon_max + 2),
                          variable=band_name)
    # output.data = np.float32(output.data)
    if band_name != 'topo':
        assert np.allclose(output.data[:, 0, :, :],
                           output.data[:, 1, :, :]), "`bnds` dim components of tensor are not identical"
    return output


def interp_timewise(data_arr: xarray.DataArray, lon_res: float = 0.25, lat_res: float = 0.25,
                    interp_method: str = 'linear', plot_example: bool = False) -> tuple:
    assert np.allclose(data_arr.data[:, 0, :, :],
                       data_arr.data[:, 1, :, :]), "2d components of tensor are not identical"
    timesteps = len(data_arr.time.data)
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
            plot_example_dice = np.random.randint(0, timesteps)
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
    coords_new = {'lon': ("lon", xnew), 'lat': ("lat", ynew), 'time': ("time", data_arr.time.data)}
    output = xarray.DataArray(data=data_new, coords=coords_new, dims=('time', 'lat', 'lon'), attrs=data_arr.attrs)

    return output


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
    logging.debug(f'{station_name} pixel found')
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

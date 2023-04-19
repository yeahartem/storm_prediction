from itertools import islice

import numpy as np
from matplotlib import pyplot as plt
from calendar import isleap
import xarray
import json
from functools import wraps
import time

class Dict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Config(object):
    @staticmethod
    def __load__(data):
        if type(data) is dict:
            return Config.load_dict(data)
        elif type(data) is list:
            return Config.load_list(data)
        else:
            return data

    @staticmethod
    def load_dict(data: dict):
        result = Dict()
        for key, value in data.items():
            result[key] = Config.__load__(value)
        return result

    @staticmethod
    def load_list(data: list):
        result = [Config.__load__(item) for item in data]
        return result

    @staticmethod
    def load_json(path: str):
        with open(path, "r") as f:
            result = Config.__load__(json.loads(f.read()))
        return result




def check_leap_year(date):
    year = date.astype('datetime64[Y]').astype(int) + 1970

    return np.logical_not(
        np.logical_and(np.not_equal(year % 4, 0), np.logical_or(np.not_equal(year % 100, 0), np.equal(year % 400, 0))))


def test_check_leap_year(tmp: xarray.DataArray):
    t0 = np.apply_along_axis(check_leap_year, axis=0, arr=tmp.time.data)
    t1 = np.array([isleap(v) for v in (tmp.time.data.astype('datetime64[Y]').astype(int) + 1970)])
    assert (t1 == t0).all(), "check_leap_year does not work correctly"


def deg_min_to_dec(degrees, minutes):
    return degrees + minutes / 60.


def read_splits(train_path, val_path, test_path):
    with open(train_path) as f:
        train_list = f.read().split('\n')
    with open(test_path) as f:
        test_list = f.read().split('\n')
    if val_path:
        with open(test_path) as f:
            val_list = f.read().split('\n')
    else:
        val_list = test_list
    return train_list, val_list, test_list


def batched(iterable, n):
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper
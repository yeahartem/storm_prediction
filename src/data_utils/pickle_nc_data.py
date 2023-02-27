import json
import numpy as np
import xarray

path_to_config = 'conf/train_conf.json'
with open(path_to_config) as fs:
    conf = json.load(fs)

path_to_files = conf["path_to_files"]
for path in path_to_files:
    if 'elevation' not in path_to_files:
        path_to_cmip = path

path_to_cmip = path_to_files.endswith('.nc')
file_paths = path_to_files[:-4]
f1 = xarray.load_dataset(file_paths, decode_times=False).astype(np.float32)

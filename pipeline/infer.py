import numpy as np
import sys
sys.path.append('../')
import os
sys.path.append(os.path.realpath('.'))
import torch
import time
from pytorch_lightning.loggers import TensorBoardLogger
import warnings
import json
warnings.filterwarnings("ignore")
import xarray as xr
import geopandas as gpd

from src.data_assemble.assemble_ml import *
from src.data_assemble.assemble_conv import *
from src.models.utils import *
from src.data_utils.data_processing import *
from src.data_assemble.wrap_data import *
from src.models.WindCNN import *
from src.data_assemble.wrap_data import *
torch.manual_seed(112)
random.seed(112)

def infer(path_to_config="infer_conf.json"):
    with open(path_to_config) as jf:
        conf = json.load(jf)

    path_to_files = conf["path_to_files"]
    half_side_size = conf["half_side_size"]
    rectangle_coords = conf["rectangle_coords"]
    target_res = conf["target_res"]
    filter_dict = conf["filter_dict"]
    time_limits = conf["time_limits"]
    time_limits = {'t_start': np.datetime64(time_limits['t_start']), 't_end': np.datetime64(time_limits['t_end'])}
    nn_config_path = conf["nn_init_data"]["nn_config_path"]
    path_to_save = conf["path_to_save"]
    path_to_training_data = conf["nn_init_data"]["path_to_training_data"]
    chk_path = conf["nn_init_data"]["chk_path"]
    inf_file_name = conf['inf_file_name']
    

    print("Preparing blocks")
    blocks = make_blocks(path_to_files, filter_dict, rectangle_coords, target_res, half_side_size=half_side_size, time_limits=time_limits)
    print("Preparing blocks - done")

    print("Assembling dataset for inference")
    X = assemble_numpy_ds(blocks=blocks, target='', stations_pixs='', include_target=False)
    print("Assembling dataset for inference - done")

    #initialize and load model
    print("Initializing NN")
    batch_size = 1024
    with open(nn_config_path) as fs:
        args = json.load(fs)

    #os.path.join('..', 'data', 'nn_train')
    st_split_dict = train_val_test_split(path_to_training_data, train = 0.5, val = 0.25, test = 0.25, verbose = True)

    X_init, y_init = extract_splitted_data(path_to_training_data, st_split_dict)

    # logger = TensorBoardLogger(save_dir='../logs/wind', name='windnet')

    dm = WindDataModule(X=X_init, y=y_init, batch_size=batch_size, downsample=False)
    model = WindNetPL(args)

    
    chk_path = os.path.join(chk_path, "checkpoints", os.listdir(os.path.join(chk_path, "checkpoints"))[0])
    model2 = WindNetPL.load_from_checkpoint(chk_path, args=args)
    model2.eval()
    print("Initializing NN - done")

    print("Inference")
    lat_axis = []
    lon_axis = []
    for pix_idx, curr_X in X.items():
        curr_lat = curr_X.lat[half_side_size-1].data
        lat_axis.append(curr_lat)
        curr_lon = curr_X.lon[half_side_size-1].data
        lon_axis.append(curr_lon)
    time_axis = curr_X.time.data
    lat_axis = np.sort(np.unique(np.array(lat_axis)))
    lon_axis = np.sort(np.unique(np.array(lon_axis)))
    inference_xarray = xr.DataArray(
                                    data=np.ones((len(lon_axis), len(lat_axis), len(time_axis))),
                                    dims=["lon", "lat", "time"],
                                    coords=dict(
                                                lon=(["lon"], lon_axis),
                                                lat=(["lat"], lat_axis),
                                                time=(["time"], time_axis)
                                                ),
                                    attrs=curr_X.attrs
                                    )

    for pix_idx, curr_X in tqdm(X.items()):
        curr_lat  = curr_X.lat[half_side_size-1].data
        curr_lon  = curr_X.lon[half_side_size-1].data
        with torch.no_grad():
            if dm.transform is not None:
                inference_pix = model(dm.transform(torch.tensor(curr_X.data, device=model.device).double())).exp()[:,1].cpu().numpy()
            else:
                inference_pix = model(torch.tensor(curr_X.data, device=model.device).double()).exp()[:,1].cpu().numpy()
        inference_xarray.loc[dict(lat=curr_lat, lon=curr_lon)] = inference_pix
    inference_xarray.name = 'prob'
    print("Inference - done")
    tmp = inference_xarray.to_dataframe().reset_index()
    print("Converting to GeoDataFrame")
    gdf = gpd.GeoDataFrame(
        tmp.prob, geometry=gpd.points_from_xy(tmp.lon,tmp.lat), crs="EPSG:4326")
    gdf['time'] = tmp.time    
    print("Converting to GeoDataFrame - done")
    # path_to_save = "/home/s.lukashevich/Wind/data/nn_inference"
    
    print("Saving into ", os.path.join(path_to_save, inf_file_name))
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    if not os.path.exists(os.path.join(path_to_save, 'pics')):
        os.makedirs(os.path.join(path_to_save, 'pics'))
    
    print("Saving into - done")
    gdf.to_file(os.path.join(path_to_save, inf_file_name), driver="GeoJSON")  

    print("Sample maps (10)")
    for i, time in enumerate(gdf.time.unique()):
        f, ax = plt.subplots(1, figsize=(10, 5))
        ax = gdf[gdf['time'] == time].plot(column='prob', cmap='afmhot', ax=ax, legend=True)
        if i > 10:
            break
        plt.savefig(os.path.join(path_to_save, 'pics', str(time) + '.png'))
    print("Sample maps (10) - done")
    
    

if __name__ == "__main__":
    # assert len(sys.argv) > 1, "Provide path to config file"
    if len(sys.argv) == 1:
        sys.argv.append('pipeline/infer_conf.json')
    path_to_config = sys.argv[1]
    t1 = time.time()
    infer(path_to_config=path_to_config)
    t2 = time.time()
    print("Total time: ", t2 - t1)
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
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import calibration_curve
import matplotlib as mpl

from src.data_assemble.assemble_ml import *
from src.data_assemble.assemble_conv import *
from src.models.utils import *
from src.data_utils.data_processing import *
from src.data_assemble.wrap_data import *
from src.models.WindCNN import *
from src.models.temperature_scaling import *
from src.data_assemble.wrap_data import *
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s-%(message)s')

torch.manual_seed(112)
random.seed(112)


def infer(path_to_config="conf/infer_conf.json"):
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

    logging.info("Preparing blocks")
    blocks = make_blocks(path_to_files, filter_dict, rectangle_coords, target_res, half_side_size=half_side_size,
                         time_limits=time_limits)
    logging.info("Preparing blocks - done")

    logging.info("Assembling dataset for inference")
    X = assemble_numpy_ds(blocks=blocks, target='', stations_pixs='', include_target=False)
    file = open('infer_tmp.pkl', 'wb')
    # dump information to that file
    pickle.dump(X, file)
    file.close()

    logging.info("Assembling dataset for inference - done")

    file = open('infer_tmp.pkl', 'rb')
    X = pickle.load(file)
    file.close()

    # initialize and load model
    logging.info("Initializing NN")
    batch_size = 1024
    with open(nn_config_path) as fs:
        args = json.load(fs)

    # os.path.join('..', 'data', 'nn_train')
    stations_list = get_stations(all_stations_data='data_mounted/weather_stations/weatherstation_list.json',
                                 stations_allowed_path="conf/splits/time_split_stations.txt",
                                 max_lat=rectangle_coords['lat_max'], min_lat=rectangle_coords['lat_min'],
                                 max_lon=rectangle_coords['lon_max'], min_lon=rectangle_coords['lon_min'],
                                 max_height=300, min_height=-10)

    logging.info(f'Total stations: {len(stations_list)}')
    X_train, y_train = extract_splitted_data(os.path.join(conf["path_to_save"],"train"), stations_list)
    X_test, y_test = extract_splitted_data(os.path.join(conf["path_to_save"], "test"), stations_list)
    X_init = {"Train": X_train, "Val": X_test, "Test": X_test}
    y_init = {"Train": y_train, "Val": y_test, "Test": y_test}
    # logger = TensorBoardLogger(save_dir='../logs/wind', name='windnet')

    with open('data_mounted/X_backup_infer.npy', 'wb') as f:
        pickle.dump(X_init, f)
    with open('data_mounted/y_backup_infer', 'wb') as f:
        pickle.dump(y_init, f)

    dm = WindDataModule(X=X_init, y=y_init, batch_size=batch_size, downsample=False)
    dm.setup()
    optimizer = torch.optim.Adam
    net = WindNet()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    chk_path = os.path.join(chk_path, "checkpoints", os.listdir(os.path.join(chk_path, "checkpoints"))[0])
    model = WindNetPL.load_from_checkpoint(chk_path, args=args, net=net, optimizer=optimizer,
                                           scheduler=scheduler)  # WindNetPL(args, net=net, optimizer=optimizer, scheduler=scheduler)
    model.eval()

    # model2 = WindNetPL.load_from_checkpoint(chk_path, args=args)
    # model2.eval()
    logging.info("Initializing NN - done")
    logging.info("Calibrating on val")
    temp_scaled_model = ModelWithTemperature(model)

    # Tune the model temperature, and save the results
    temp_scaled_model.set_temperature(dm.val_dataloader())
    # model = temp_scaled_model
    logging.info('Done!')

    logging.info("Inference")
    lat_axis = []
    lon_axis = []
    for pix_idx, curr_X in X.items():
        curr_lat = curr_X.lat[half_side_size - 1].data
        lat_axis.append(curr_lat)
        curr_lon = curr_X.lon[half_side_size - 1].data
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
        curr_lat = curr_X.lat[half_side_size - 1].data
        curr_lon = curr_X.lon[half_side_size - 1].data
        with torch.no_grad():
            if dm.transform is not None:
                inference_pix = nn.Sigmoid()(temp_scaled_model(dm.transform(torch.tensor(curr_X.data, device=0)))).cpu()
            else:
                inference_pix = nn.Sigmoid()(temp_scaled_model(torch.tensor(curr_X.data, device=0))).cpu()
        inference_xarray.loc[dict(lat=curr_lat, lon=curr_lon)] = torch.squeeze(inference_pix).numpy()
    inference_xarray.name = 'prob'
    logging.info("Inference - done")
    tmp = inference_xarray.to_dataframe().reset_index()
    logging.info("Converting to GeoDataFrame")
    gdf = gpd.GeoDataFrame(
        tmp.prob, geometry=gpd.points_from_xy(tmp.lon, tmp.lat), crs="EPSG:4326")
    gdf['time'] = tmp.time
    logging.info("Converting to GeoDataFrame - done")
    # path_to_save = "/home/s.lukashevich/Wind/data/nn_inference"

    logging.info("Saving into ", os.path.join(path_to_save, inf_file_name))
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    if not os.path.exists(os.path.join(path_to_save, 'pics')):
        os.makedirs(os.path.join(path_to_save, 'pics'))

    logging.info("Saving into - done")
    gdf.to_file(os.path.join(path_to_save, inf_file_name), driver="GeoJSON")

    logging.info("Calibraiton curve")
    with torch.no_grad():
        y_pred_binary = nn.Sigmoid()(temp_scaled_model(dm.transform(torch.tensor(X_init['Val'], device=0)))).detach().cpu().numpy()  # binary_model.predict(x_val_binary)
        # y_pred_binary = temp_scaled_model(dm.transform(torch.tensor(X_init['Val'], device=model.device))).exp()[:, 1].detach().cpu().numpy()#binary_model.predict(x_val_binary)
        y_val_binary = y_init["Val"]
    acc_score = accuracy_score(y_val_binary, y_pred_binary >= args['threshold'])
    loss_score = log_loss(y_val_binary, y_pred_binary)
    logging.info('Binary metrics: validation accuracy is {0:.2f}, validation loss is {1:.2f}'.format(acc_score, loss_score))
    prob_true_binary, prob_pred_binary = calibration_curve(y_val_binary, y_pred_binary, n_bins=5, strategy='quantile')
    plot_reliability_diagram(prob_true_binary, prob_pred_binary, "WindNet")
    plt.savefig(os.path.join(path_to_save, 'pics', 'calibration_curve' + '.png'))
    logging.info("Calibraiton curve saved")

    # print("Sample maps")
    # for i, time in enumerate(gdf.time.unique()):
    #     f, ax = plt.subplots(1, figsize=(10, 5))
    #     ax = gdf[gdf['time'] == time].plot(column='prob', cmap='afmhot', ax=ax, legend=True)
    #     # if i > 10:
    #     #     break
    #     plt.savefig(os.path.join(path_to_save, 'pics', str(time) + '.png'))
    # print("Sample maps (10) - done")
    logging.info("Sample maps (10)")

    try:
        region_name = conf["region_name"]
        rus_bnd_gdf = gpd.read_file('pipeline/geo.json')
        state_df = rus_bnd_gdf[(rus_bnd_gdf.NAME_1 == region_name)]
        for i, time in enumerate(gdf.time.unique()):
            fig, gax = plt.subplots(1, figsize=(10, 10))
            g = gdf[gdf['time'] == time].plot(ax=gax, c=gdf[gdf['time'] == time]['prob'], marker='s', markersize=3000,
                                              alpha=0.95)
            state_df.plot(ax=gax, edgecolor="white", color="None", lw=3, alpha=1)
            cmap = gax.collections[-1].colorbar
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cbar = plt.colorbar(sm, orientation="horizontal", fraction=0.03, pad=0.009, ).set_label(
                label='Probability of Strong Wind', size=15)  # ,weight='bold'
            gax.axis('off')
            plt.savefig(os.path.join(path_to_save, 'pics', str(time) + '.png'))
            plt.close(fig)
            # plt.title(str(time)[:10], fontsize=25)
            if i >= 10:
                break

    except KeyError:
        for i, time in enumerate(gdf.time.unique()):
            f, ax = plt.subplots(1, figsize=(10, 5))
            ax = gdf[gdf['time'] == time].plot(column='prob', cmap='afmhot', ax=ax, legend=True)
            if i > 10:
                break
            plt.savefig(os.path.join(path_to_save, 'pics', str(time) + '.png'))

    logging.info("Sample maps (10) - done")


if __name__ == "__main__":
    # assert len(sys.argv) > 1, "Provide path to config file"

    if len(sys.argv) == 1:
        sys.argv.append('conf/infer_conf_chel.json')
    path_to_config = sys.argv[1]
    t1 = time.time()
    infer(path_to_config=path_to_config)
    t2 = time.time()
    logging.info(f"Total time: {t2 - t1}")
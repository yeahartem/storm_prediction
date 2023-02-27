import numpy as np
import sys

from src.data_assemble.create_dataset import load_dataset_as_xarray

sys.path.append('../')
import os

sys.path.append(os.path.realpath('.'))
import torch
import time
import warnings
import json

warnings.filterwarnings("ignore")
import xarray as xr
from sklearn.metrics import log_loss
from sklearn.calibration import calibration_curve
import matplotlib as mpl
from src.data_assemble.assemble_conv import *
from src.models.utils import *
from src.models.WindCNN import *
from src.models.temperature_scaling import *
from src.data_assemble.wrap_data import *
import logging
import copy
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s-%(message)s')

torch.manual_seed(112)
random.seed(112)


def infer(path_to_config):
    with open(path_to_config) as jf:
        conf = json.load(jf)
    rus_bnd_gdf = gpd.read_file('conf/geo.json')
    region_name = conf['region_name']
    override_region_rectangle = conf['override_region_rectangle']
    make_calibration_curve = conf['make_calibration_curve']
    load_data = conf['load_data']
    load_dump = conf['load_dump']
    save_dump = conf['save_dump']
    calibrate_model = conf['calibrate_model']
    temperature = conf['temperature']
    coord_offset = conf['coord_offset']
    rectangle_coords = {}
    if not override_region_rectangle:
        df = rus_bnd_gdf[(rus_bnd_gdf.NAME_1 == region_name)]
        rectangle_coords['lon_min'] = df.bounds['minx'].values[0] - coord_offset
        rectangle_coords['lon_max'] = df.bounds['maxx'].values[0] + coord_offset
        rectangle_coords['lat_min'] = df.bounds['miny'].values[0] - coord_offset
        rectangle_coords['lat_max'] = df.bounds['maxy'].values[0] + coord_offset
    else:
        rectangle_coords = conf["rectangle_coords"]
    path_to_files = conf["path_to_files"]
    half_side_size = conf["half_side_size"]
    target_res = conf["target_res"]
    time_limits = copy.deepcopy(conf["time_limits"])
    time_limits[0] = np.datetime64(time_limits[0])
    time_limits[1] = np.datetime64(time_limits[1])
    chk_path = conf["nn_init_data"]["chk_path"]
    inf_file_name = conf['inf_file_name']
    path_to_save = os.path.join(conf['path_to_save'], region_name)
    conf['actual rectangle'] = rectangle_coords
    bands = conf['bands']
    args = {'lr': 1e-4, 'threshold': 0.5}

    os.makedirs(path_to_save, exist_ok=True)
    with open(os.path.join(path_to_save, 'infer_conf.json'), 'w') as fp:
        json.dump(conf, fp)

    logging.info("Preparing blocks")
    if not load_data:
        dataset_as_xarray = load_dataset_as_xarray(file_paths=path_to_files,
                                                   rectangle_coords=rectangle_coords,
                                                   target_res=target_res,
                                                   bands=bands,
                                                   time_limits=time_limits)

        X = make_blocks_no_target(dataset_as_xarray=dataset_as_xarray,
                                  half_side_size=half_side_size)
        logging.info("Preparing blocks - done")
        logging.info("Assembling dataset for inference")
        file = open(save_dump, 'wb')
        pickle.dump(X, file)
        file.close()
        logging.info("Assembling dataset for inference - done")
    else:
        file = open(load_dump, 'rb')
        X = pickle.load(file)
        file.close()

    logging.info("Initializing NN")
    batch_size = 1024
    stations_list = get_stations(all_stations_data='data_mounted/weather_stations/weatherstation_list.json',
                                 stations_allowed_path="conf/splits/time_split_stations.txt",
                                 max_lat=80.52, min_lat=36.38,
                                 max_lon=181.45, min_lon=32.12,
                                 max_height=300, min_height=-10)

    logging.info(f'Total stations: {len(stations_list)}')
    X_train, y_train = extract_splitted_data(os.path.join(conf["path_to_init_data"], "train"), stations_list)
    X_test, y_test = extract_splitted_data(os.path.join(conf["path_to_init_data"], "test"), stations_list)
    X_init = {"Train": X_train, "Val": X_test, "Test": X_test}
    y_init = {"Train": y_train, "Val": y_test, "Test": y_test}

    dm = WindDataModule(X=X_init, y=y_init, batch_size=batch_size, downsample=False)
    dm.setup()
    optimizer = torch.optim.Adam
    net = WindNet()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    chk_path = os.path.join(chk_path, "checkpoints", os.listdir(os.path.join(chk_path, "checkpoints"))[0])
    model = WindNetPL.load_from_checkpoint(chk_path, args=args, net=net, optimizer=optimizer,
                                           scheduler=scheduler)
    model.eval()
    logging.info("Initializing NN - done")

    temp_scaled_model = ModelWithTemperature(model)
    if calibrate_model:
        logging.info("Calibrating on val")
        temp_scaled_model.set_temperature(dm.val_dataloader())
    else:
        temp_scaled_model.temperature = torch.nn.Parameter(torch.tensor([temperature]))
        temp_scaled_model = temp_scaled_model.cuda()
        logging.info(f'Temperature set {temperature}')

    logging.info("Inference")

    result_xarray = xr.DataArray(
        data=np.ones((len(X.lat.data), len(X.lon.data), len(X.time.data))),
        dims=["lat", "lon", "time"],
        coords=dict(
            lat=(["lat"], X.lat.data),
            lon=(["lon"], X.lon.data),
            time=(["time"], X.time.data)
        ),
        attrs=X.attrs
    )

    for (curr_lat, curr_lon) in tqdm(zip(X.lat.data, X.lon.data)):
        with torch.no_grad():
            if dm.transform is not None:
                inference_pix = nn.Sigmoid()(temp_scaled_model(dm.transform(torch.tensor(X.sel(lat=curr_lat, lon=curr_lon).data, device=0)))).cpu()
            else:
                inference_pix = nn.Sigmoid()(temp_scaled_model(torch.tensor(X.sel(lat=curr_lat, lon=curr_lon).data, device=0))).cpu()
        result_xarray.loc[dict(lat=curr_lat, lon=curr_lon)] = torch.squeeze(inference_pix).numpy()

    np_res = result_xarray.values
    result_xarray.name = 'prob'
    logging.info("Inference - done")
    tmp = result_xarray.to_dataframe().reset_index()
    logging.info("Converting to GeoDataFrame")
    gdf = gpd.GeoDataFrame(
        tmp.prob, geometry=gpd.points_from_xy(tmp.lon, tmp.lat), crs="EPSG:4326")
    gdf['time'] = tmp.time
    logging.info("Converting to GeoDataFrame - done")

    logging.info(f"Saving into {os.path.join(path_to_save, inf_file_name)}")
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    if not os.path.exists(os.path.join(path_to_save, 'pics')):
        os.makedirs(os.path.join(path_to_save, 'pics'))

    logging.info("Saving into - done")
    gdf.to_file(os.path.join(path_to_save, inf_file_name), driver="GeoJSON")

    if make_calibration_curve:
        logging.info("Calibraiton curve")
        with torch.no_grad():
            y_pred_binary = nn.Sigmoid()(temp_scaled_model(dm.transform(
                torch.tensor(X_init['Val'], device=0)))).detach().cpu().numpy()  # binary_model.predict(x_val_binary)
            # y_pred_binary = temp_scaled_model(dm.transform(torch.tensor(X_init['Val'], device=model.device)))
            # .exp()[:, 1].detach().cpu().numpy()#binary_model.predict(x_val_binary)
            y_val_binary = y_init["Val"]
        acc_score = accuracy_score(y_val_binary, y_pred_binary >= args['threshold'])
        loss_score = log_loss(y_val_binary, y_pred_binary)
        logging.info(
            'Binary metrics: validation accuracy is {0:.2f}, validation loss is {1:.2f}'.format(acc_score, loss_score))
        prob_true_binary, prob_pred_binary = calibration_curve(y_val_binary, y_pred_binary, n_bins=5,
                                                               strategy='quantile')
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
    # logging.info("Sample maps (10)")
    try:
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
            # gax.axis('off')
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
    path_to_config = 'conf/infer_conf.json'
    t1 = time.time()
    infer(path_to_config=path_to_config)
    t2 = time.time()
    logging.info(f"Total time: {t2 - t1}")

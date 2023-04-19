import sys

sys.path.append('../../')
import os

sys.path.append(os.path.realpath('../pipeline'))
import warnings

warnings.filterwarnings("ignore")
import matplotlib as mpl

from src.data_assemble.assemble_data import *
from src.binary_target.models.utils import *
from src.binary_target.models.WindCNN import *
from src.binary_target.models.temperature_scaling import *
from src.binary_target.datamodule import *
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s-%(message)s')

torch.manual_seed(112)
random.seed(112)


def infer(path_to_config):
    Configuration = conf_utils.Config()
    cfg = Configuration.load_json(path_to_config)
    path_to_save = os.path.join(cfg.path_to_save, cfg.region_name)
    inf_file_name = cfg.inf_file_name
    inf_file_name_parquet = cfg.inf_file_name_parquet
    region_name = cfg.region_name
    rus_bnd_gdf = gpd.read_file('/wind/configs/geo.json')

    logging.info("Preparing blocks")
    climate_file_paths = [os.path.join(cfg.data_dir, var + '.nc') for var in cfg.variables]
    print(f'loading {climate_file_paths}')
    lat_lon_bnds = cfg['rectangle_coords']
    time_bnds = cfg['time_limits']
    def _cut_lan_lot_time(x, lat_lon_bnds, time_bnds):
        return x.sel(lon=slice(*(lat_lon_bnds['lon_min'], lat_lon_bnds['lon_max'])), lat=slice(*(lat_lon_bnds['lat_min'], lat_lon_bnds['lat_max'])), time=slice(*(time_bnds[0], time_bnds[1])))
    _cut = partial(_cut_lan_lot_time, lat_lon_bnds=lat_lon_bnds, time_bnds=time_bnds)
    dataset_as_xarray = xr.open_mfdataset(climate_file_paths,  combine="by_coords", parallel=True, engine='scipy', compat='override', preprocess=_cut)
    var_names_to_drop = [v for v in dataset_as_xarray.keys() if v not in cfg.variables + ['lat', 'lon', 'time'] ] 
    coords_names_to_drop = [v for v in dataset_as_xarray.coords.keys() if v not in cfg.variables + ['lat', 'lon', 'time'] ] 
    dataset_as_xarray = dataset_as_xarray.drop_vars(var_names_to_drop)
    dataset_as_xarray = dataset_as_xarray.drop_vars(coords_names_to_drop)
    dataset_as_blocks = make_blocks_numpy(dataset_as_xarray, cfg.half_side_size)
    X = dataset_as_blocks

    optimizer = torch.optim.Adam
    args = {'lr': 1e-4, 'threshold': 0.5}
    chk_path = cfg["nn_init_data"]["chk_path"]
    net = WindNet(cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau
    chk_path = os.path.join(chk_path, "checkpoints", os.listdir(os.path.join(chk_path, "checkpoints"))[0])
    model = WindNetPL.load_from_checkpoint(chk_path, args=args, net=net, optimizer=optimizer,
                                            scheduler=scheduler)
    model.eval()
    logging.info("Initializing NN - done")

    file = open(os.path.join(cfg["nn_init_data"]["chk_path"], 'transform.pkl'), 'rb')
    transform = pickle.load(file)
    file.close()

    logging.info("Inference")
    # data_pix = X.sel(lat=curr_lat, lon=curr_lon)
    days = len(X.time.data)
    days_full_4weeks = (days // 28) * 28 
    result = np.zeros((len(X.lat.data), len(X.lon.data), days_full_4weeks // 28))
    for i, curr_lat in enumerate(X.lat.data):
        for j, curr_lon in enumerate(X.lon.data):
            with torch.no_grad():
                
                data_pix = X.sel(lat=curr_lat, lon=curr_lon)
                data_pix = model(transform(torch.tensor(data_pix.data[:days_full_4weeks].reshape(-1, 28, dataset_as_blocks.shape[-3], dataset_as_blocks.shape[-2], dataset_as_blocks.shape[-1]))))
                inference_pix = nn.Sigmoid()(data_pix).cpu()
                
            result[i, j] = torch.squeeze(inference_pix).numpy()

    result_mean_prob = result.mean()
    result_xarray = xr.DataArray(
        data=result,
        dims=["lat", "lon", "time"],
        coords=dict(
            lat=X.lat.data,
            lon=X.lon.data,
            time=X.time.data[:days_full_4weeks][::28]
        ))

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

    # gdf.to_file(os.path.join(path_to_save, inf_file_name), driver="GeoJSON")
    gdf.to_parquet(os.path.join(path_to_save, inf_file_name_parquet))
    logging.info("Saving into - done")

    # state_df = rus_bnd_gdf[(rus_bnd_gdf.NAME_1 == region_name)]
    for i, time in enumerate(gdf.time.unique()):
        fig, gax = plt.subplots(1, figsize=(10, 10))
        g = gdf[gdf['time'] == time].plot(ax=gax, c=gdf[gdf['time'] == time]['prob'], marker='s', markersize=3000,
                                            alpha=0.95)
        rus_bnd_gdf.plot(ax=gax, edgecolor="white", color="None", lw=3, alpha=1)
        cmap = gax.collections[-1].colorbar
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, orientation="horizontal", fraction=0.03, pad=0.009, ).set_label(
            label='Probability of Strong Wind', size=15)  # ,weight='bold'
        # gax.axis('off')
        plt.ylim((result_xarray.lat.min().data, result_xarray.lat.max().data))
        plt.xlim((result_xarray.lon.min().data, result_xarray.lon.max().data))
        plt.savefig(os.path.join(path_to_save, 'pics', str(time) + '.png'))
        plt.close(fig)
        # plt.title(str(time)[:10], fontsize=25)
        if i >= 1:
            break

    # print('RESULT PROB:', result_mean_prob)

if __name__ == "__main__":
    path_to_config = '/wind/configs/infer_conf.json'
    t1 = time.time()
    infer(path_to_config=path_to_config)
    t2 = time.time()
    logging.info(f"Total time: {t2 - t1}")

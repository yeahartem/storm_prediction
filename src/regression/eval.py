import sys,os
sys.path.append(os.getcwd())
import warnings
import torch
import logging
from tqdm import tqdm
from src.regression.data_load import make_padding
from src.regression.models.pl_module import WindNetPL
import hydra
from omegaconf import DictConfig
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import torch.nn as nn


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'

def plot_prediction(cfg: DictConfig, predictions_df: np.ndarray, days: int) -> None:
    given_days = predictions_df['date'].unique()
    logging.info(f"Plot in coords:")
    logging.info(f"Lat {predictions_df['lat'].min()}-{predictions_df['lat'].max()}")
    logging.info(f"Lon {predictions_df['lon'].min()}-{predictions_df['lon'].max()}")
    for day in given_days[:days]:
        logging.info(f"Day: {day}")
        current_data = predictions_df[predictions_df['date'] == day]
        pivot_table = current_data.pivot(index='lat', columns='lon', values='prediction')    
        image_array = pivot_table.values
        fig, ax = plt.subplots(figsize=(12, 12))
        plt.title(f'Max wind speed prediction for {pd.to_datetime(day).date()}')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        img = ax.imshow(image_array, interpolation='nearest', extent=[predictions_df['lon'].min(),predictions_df['lon'].max(),
                                                                      predictions_df['lat'].max(), predictions_df['lat'].min()])
        plt.gca().invert_yaxis()
        cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        fig.colorbar(img, cax=cax)
        savepath = os.path.join(*cfg.eval.path_to_predictions.split('/')[:-1], f'wind_max_{pd.to_datetime(day).date()}.png')
        plt.savefig(savepath, dpi=200) 
        logging.info(f"Plot saved")
        plt.clf()
        ax.cla()

    
class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.prepare_data()
        self.total_index = list(itertools.product(self.time_indexes, self.lat_indexes, self.lon_indexes))
        logging.info(f"Coordinates to be predicted")
        logging.info(f"Lat: {min(self.lat_coords_full)} - {max(self.lat_coords_full)}")
        logging.info(f"Lon: {min(self.lon_coords_full)} - {max(self.lon_coords_full)}")
        logging.info(f"Time: {min(self.time_coords)} - {max(self.time_coords)}")


    def load_dataset(self):
        """Load climate data from given folder"""
        self.time_coords_full = np.load(os.path.join(self.cfg.eval.data_dir, 'time.npy')).astype('datetime64[D]')
        self.lat_coords_full = np.load(os.path.join(self.cfg.eval.data_dir, 'lat.npy'))
        self.lon_coords_full = np.load(os.path.join(self.cfg.eval.data_dir, 'lon.npy'))
        self.var_data = np.empty((len(self.cfg.process.variables), len(self.time_coords_full), len(self.lat_coords_full), len(self.lon_coords_full)), dtype=np.float16)
        for i, var in enumerate(self.cfg.process.variables):
            self.var_data[i] = np.load(os.path.join(self.cfg.eval.data_dir, var + f'_{self.cfg.process.precision}.npy'))


    def limit_inference_space(self):
        """Crop data by spatial coordinates"""
        time_start = pd.to_datetime(self.cfg.eval.time_start)
        time_end = pd.to_datetime(self.cfg.eval.time_end)
        self.time_min_idx = np.searchsorted(self.time_coords_full, time_start)
        self.time_max_idx = np.searchsorted(self.time_coords_full, time_end)

        self.time_coords = self.time_coords_full[self.time_min_idx:self.time_max_idx]
        self.time_indexes = list(range(self.time_min_idx + self.cfg.time_window//2, self.time_max_idx + self.cfg.time_window//2)) 
        self.lat_indexes = list(range(self.cfg.half_side_size, len(self.lat_coords_full) + self.cfg.half_side_size))
        self.lon_indexes = list(range(self.cfg.half_side_size, len(self.lon_coords_full) + self.cfg.half_side_size))
        
        self.var_data = self.var_data[:, self.time_min_idx:self.time_max_idx, :, :]
        self.time_indexes = [t_idx - self.time_min_idx for t_idx in self.time_indexes]
        assert len(self.time_indexes) == len(self.time_coords), f"{len(self.time_indexes)} {len(self.time_coords)}"
        assert self.var_data.shape[1] == len(self.time_coords), f"{len(self.var_data.shape[1])} {len(self.time_coords)}"
        logging.info(f"Bounded data shape: {self.var_data.shape}")


    def prepare_data(self):
        """Prepare data for inference"""
        self.load_dataset()
        self.limit_inference_space()
        self.var_data, self.shift = make_padding(self.var_data, self.cfg.half_side_size)
        assert len(self.lat_indexes) == len(self.lat_coords_full), f" indexes is {len(self.lat_indexes)} while coords is {len(self.lat_coords_full)}"
        assert len(self.lon_indexes)== len(self.lon_coords_full), f" indexes is {len(self.lon_indexes)} while coords is  {len(self.lon_coords_full)}"
        assert len(self.time_indexes)== len(self.time_coords), f" indexes is {len(self.time_indexes)} while coords is {len(self.time_coords)}"
    
    def __len__(self):
        return len(self.total_index)

    def __getitem__(self, idx):
        t, lat, lon  = self.total_index[idx]
        item = self.var_data[:,
                            slice(t - self.cfg.time_window//2, t + self.cfg.time_window//2 + 1),
                            slice(lat - self.cfg.half_side_size, lat + self.cfg.half_side_size + 1),
                            slice(lon - self.cfg.half_side_size, lon + self.cfg.half_side_size + 1),
                            ]
        item = torch.from_numpy(item).to(torch.float16) #torch.float32
        coord = np.array([t, lat, lon])
        return item, coord



class EvalDatasetElev(torch.utils.data.Dataset):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.prepare_data()
        self.total_index = list(itertools.product(self.time_indexes, self.lat_indexes, self.lon_indexes))
        logging.info(f"Coordinates to be predicted")
        logging.info(f"Lat: {min(self.lat_coords_full)} - {max(self.lat_coords_full)}")
        logging.info(f"Lon: {min(self.lon_coords_full)} - {max(self.lon_coords_full)}")
        logging.info(f"Time: {min(self.time_coords)} - {max(self.time_coords)}")


    def load_dataset(self):
        """Load climate data from given folder"""
        self.time_coords_full = np.load(os.path.join(self.cfg.eval.data_dir, 'time.npy')).astype('datetime64[D]')
        self.lat_coords_full = np.load(os.path.join(self.cfg.eval.data_dir, 'lat.npy'))
        self.lon_coords_full = np.load(os.path.join(self.cfg.eval.data_dir, 'lon.npy'))
        self.var_data = np.empty((len(self.cfg.process.variables), len(self.time_coords_full), len(self.lat_coords_full), len(self.lon_coords_full)), dtype=np.float16)
        for i, var in enumerate(self.cfg.process.variables):
            self.var_data[i] = np.load(os.path.join(self.cfg.eval.data_dir, var + f'_{self.cfg.process.precision}.npy'))
        logging.info(f"Climate shape: {self.var_data.shape}")


    def load_evevation(self):
        """Load elevation data from given folder"""
        lat_elev_coords = np.load(os.path.join(self.cfg.eval.data_dir, 'elev_lat.npy'))
        lon_elev_coords = np.load(os.path.join(self.cfg.eval.data_dir, 'elev_lon.npy'))
        dtype = np.float16
        self.elev_data = np.empty((len(lat_elev_coords), len(lon_elev_coords)), dtype=dtype)
        elevation_path = os.path.join(self.cfg.eval.data_dir, f'elev_{16}.npy')
        assert os.path.isfile(elevation_path), f"Elevation file {elevation_path} does not exist"
        self.elev_data[...] = np.load(elevation_path)
        self.r = np.max((np.abs(np.diff(self.lat_coords_full)).max(), np.abs(np.diff(self.lon_coords_full)).max())) / np.min((np.abs(np.diff(lat_elev_coords)).min(), np.abs(np.diff(lon_elev_coords)).min()))
        self.r_lat = np.abs(np.diff(self.lat_coords_full)).max() / np.abs(np.diff(lat_elev_coords)).min()
        self.r_lon = np.abs(np.diff(self.lon_coords_full)).max() / np.abs(np.diff(lon_elev_coords)).min()
        self.elev_hss = int(self.r * self.cfg.half_side_size) // 2
        self.r = torch.from_numpy(np.atleast_1d(self.r))
        self.r_lat = torch.from_numpy(np.atleast_1d(self.r_lat))
        self.r_lon = torch.from_numpy(np.atleast_1d(self.r_lon))
        logging.info(f"Elevation shape: {self.elev_data.shape}")


    def limit_inference_space(self):
        """Crop data by spatial coordinates"""
        time_start = pd.to_datetime(self.cfg.eval.time_start)
        time_end = pd.to_datetime(self.cfg.eval.time_end)
        self.time_min_idx = np.searchsorted(self.time_coords_full, time_start)
        self.time_max_idx = np.searchsorted(self.time_coords_full, time_end)

        self.time_coords = self.time_coords_full[self.time_min_idx:self.time_max_idx]
        self.time_indexes = list(range(self.time_min_idx + self.cfg.time_window//2, self.time_max_idx + self.cfg.time_window//2)) 
        self.lat_indexes = list(range(self.cfg.half_side_size, len(self.lat_coords_full) + self.cfg.half_side_size))
        self.lon_indexes = list(range(self.cfg.half_side_size, len(self.lon_coords_full) + self.cfg.half_side_size))
        
        self.var_data = self.var_data[:, self.time_min_idx - self.cfg.time_window//2:self.time_max_idx + self.cfg.time_window//2, :, :]
        self.time_indexes = [t_idx - self.time_min_idx for t_idx in self.time_indexes]
        if isinstance(self.time_coords, list) and isinstance(self.time_indexes, list):
            assert len(self.time_indexes) == len(self.time_coords), f"{len(self.time_indexes)} {len(self.time_coords)}"
            assert self.var_data.shape[1] == len(self.time_coords), f"{len(self.var_data.shape[1])} {len(self.time_coords)}"
        logging.info(f"Bounded data shape: {self.var_data.shape}")


    def prepare_data(self):
        """Prepare data for inference"""
        self.load_dataset()
        self.load_evevation()
        self.limit_inference_space()
        self.var_data, self.shift = make_padding(self.var_data, self.cfg.half_side_size)
        self.elev_data, self.shift_elev = make_padding(self.elev_data, self.elev_hss)
        assert len(self.lat_indexes) == len(self.lat_coords_full), f" indexes is {len(self.lat_indexes)} while coords is {len(self.lat_coords_full)}"
        assert len(self.lon_indexes)== len(self.lon_coords_full), f" indexes is {len(self.lon_indexes)} while coords is  {len(self.lon_coords_full)}"
        assert len(self.time_indexes)== len(self.time_coords), f" indexes is {len(self.time_indexes)} while coords is {len(self.time_coords)}"
    
    def __len__(self):
        return len(self.total_index)

    def __getitem__(self, idx):
        t, lat, lon  = self.total_index[idx]
        X = self.var_data[:,
                            slice(t - self.cfg.time_window//2, t + self.cfg.time_window//2 + 1),
                            slice(lat - self.cfg.half_side_size, lat + self.cfg.half_side_size + 1),
                            slice(lon - self.cfg.half_side_size, lon + self.cfg.half_side_size + 1),
                            ]
        
        X = torch.from_numpy(X).to(torch.float16) 

        lat_index_elev = int((lat - self.shift[0]) * self.r_lat) + self.shift_elev[0]
        lon_index_elev = int((lon - self.shift[1]) * self.r_lon) + self.shift_elev[1]
        X_elev = self.elev_data[
                                slice(lat_index_elev - self.elev_hss, lat_index_elev + self.elev_hss + 1),
                                slice(lon_index_elev - self.elev_hss, lon_index_elev + self.elev_hss + 1),
                                ]
        X_elev = torch.from_numpy(X_elev).to(torch.float16) 
        X_elev = X_elev.view(1, X_elev.shape[-2], X_elev.shape[-1])
        coord = np.array([t, lat, lon])
        return (X, X_elev), coord

    
def load_model(cfg: DictConfig):
    return WindNetPL.load_from_checkpoint(cfg.eval.path_to_checkpoint, cfg=cfg, eval=True).half().eval()


def predict(model, dataset, use_elevation, batch_size=1, distributed=False, device_num=0, num_workers=0):    
    if torch.cuda.is_available():
        torch.cuda.set_device(device_num)
        device = torch.device("cuda") 
    else:
        device = torch.device("cpu")
    logging.info(f'Using {device}')
    if distributed:
        model = nn.DataParallel(model)
    else:
        model.to(device)
    
    number_of_points = len(dataset.lat_indexes) * len(dataset.lon_indexes) * len(dataset.time_indexes)
    logging.info((f"Number of points to predict: {number_of_points}"))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
    predictions_list = []
    coords_list = []  
    coords_indxs_list =  []
    num_items_to_predict = len(dataloader)
    logging.info(f"Number of batches to predict: {num_items_to_predict}")

    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_items_to_predict, desc="Inference"):
            data = batch[0]
            coords_idxs = batch[1]
            if not distributed:
                if use_elevation:
                    data = [t.to(device) for t in data]
                else:
                    data = data.to(device)
            prediction = model(data)
            prediction = prediction.detach().cpu()
            prediction = prediction.numpy()
            predictions_list = predictions_list + list(prediction)
            coords_indxs_list.append(coords_idxs)
    logging.info("Inference finished")

    for coords_idxs in coords_indxs_list:
        for t, lat, lon, in coords_idxs:
            coords = [dataset.lat_coords_full[lat - dataset.shift[0]],
                      dataset.lon_coords_full[lon - dataset.shift[1]],
                      dataset.time_coords[t - dataset.cfg.time_window//2]]
            coords_list.append(coords)

    predictions = np.concatenate(predictions_list, axis=0)
    logging.info(f"Predictions max value: {predictions.max()}, min value: {predictions.min()}")
    logging.info(f"Predictions mean value: {predictions.mean()}, std value: {predictions.std()}")
    result_df = pd.DataFrame({"date": [item[2] for item in coords_list],
                              "lat": [item[0] for item in coords_list],
                              "lon": [item[1] for item in coords_list],
                              "prediction": predictions.flatten().astype(np.float64)
                           })
    return result_df


def eval(cfg: DictConfig) -> None:        
    model = load_model(cfg)
    if cfg.eval.use_elevation:
        dataset = EvalDatasetElev(cfg)
    else:
        dataset = EvalDataset(cfg)

    result_df = predict(model, dataset,
                        use_elevation=cfg.eval.use_elevation,\
                        batch_size=cfg.eval.batch_size_test,
                        distributed=cfg.eval.distributed_test,
                        num_workers=cfg.eval.num_workers_eval)
                        
    os.makedirs(os.path.join(*cfg.eval.path_to_predictions.split('/')[:-1]), exist_ok=True)
    result_df.to_csv(cfg.eval.path_to_predictions, index=False)
    plot_prediction(cfg, result_df, 1)


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="cmip5_WindNet41x41.yaml")
def main(cfg: DictConfig):    
    eval(cfg)

if __name__ == "__main__":      
    main()
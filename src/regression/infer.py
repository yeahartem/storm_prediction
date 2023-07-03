import sys,os
sys.path.append(os.getcwd())
import warnings
import torch
import logging
from tqdm import tqdm
from src.data_assemble.assemble_eval import load_dataset, load_target
from src.regression.data_load import DataPreLoader
from src.regression.models.pl_module import WindNetPL
import hydra
from omegaconf import DictConfig
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')


def plot_prediction(cfg: DictConfig, predictions_df: np.ndarray, days: int) -> None:

    given_days = predictions_df['time'].unique()
    for day in given_days[:days]:
        current_data = predictions_df[predictions_df['time'] == day]
        pivot_table = current_data.pivot(index='lat', columns='lon', values='prediction')    
        image_array = pivot_table.values
        
        plt.figure(figsize=(14,14))
        plt.title(f'Max wind speed prediction for {pd.to_datetime(day).date()}')
        plt.imshow(image_array, cmap='magma', interpolation='lanczos')
        plt.gca().invert_yaxis()
        # cb = plt.colorbar() 
        savepath = os.path.join(cfg.path_to_predictions, f'wind_max_{pd.to_datetime(day).date()}.png')
        plt.savefig(savepath)
        # cb.remove()
        

class DataLoader:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.prepare_data()
        self.cr_batch = cfg.cube_root_of_batch_size

    def prepare_data(self):
        """Prepare data for inference"""
        var_data, time_coords, lat_coords, lon_coords = load_dataset(self.cfg, time_slices = 44)
        self.var_data_blocks = DataPreLoader.data_to_blocks(var_data, self.cfg.time_window, self.cfg.half_side_size)
        
        self.time_coords = time_coords[self.cfg.time_window//2:len(time_coords) - self.cfg.time_window//2]
        self.lat_coords = lat_coords[self.cfg.half_side_size:len(lat_coords) - self.cfg.half_side_size]
        self.lon_coords = lon_coords[self.cfg.half_side_size:len(lon_coords) - self.cfg.half_side_size]
        assert self.var_data_blocks.shape[0] == len(self.lat_coords)
        assert self.var_data_blocks.shape[1] == len(self.lon_coords)
        assert self.var_data_blocks.shape[2] == len(self.time_coords)

    def single_loader(self):
        """Generate items for inference"""
        for index in np.ndindex(self.var_data_blocks.shape[0], self.var_data_blocks.shape[1], self.var_data_blocks.shape[2]):
            item = self.var_data_blocks[index[0], index[1], index[2]]
            item = torch.from_numpy(item).to(torch.float16) #torch.float32
            item = item.unsqueeze(0)
            yield item, [self.lat_coords[index[0]], self.lon_coords[index[1]], self.time_coords[index[2]]] 

    def batch_loader(self): # TODO - not working
        """Generate batches for inference"""
        cr_batch = self.cfg.cr_batch

        for i in range(0, self.var_data_blocks.shape[0] + self.cfg.cr_batch - 1, self.cfg.cr_batch):
            for j in range(0, self.var_data_blocks.shape[1]+ self.cfg.cr_batch - 1, self.cfg.cr_batch):
                for k in range(0, self.var_data_blocks.shape[2]+ self.cfg.cr_batch - 1, self.cfg.cr_batch):
                    item = self.var_data_blocks[i:i+self.cfg.cr_batch, j:j+self.cfg.cr_batch, k:k+self.cfg.cr_batch]
                    item = torch.from_numpy(item).to(torch.float16)
                    if item.shape[0] >0:
                        yield item, [self.lat_coords[i:i+self.cfg.cr_batch], self.lon_coords[j:j+self.cfg.cr_batch], self.time_coords[k:k+self.cfg.cr_batch]]


def test(cfg: DictConfig) -> None:        

    model = WindNetPL.load_from_checkpoint(cfg.path_to_checkpoint, cfg=cfg).half()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    DL = DataLoader(cfg)
    dataloader = DL.single_loader()
    num_items_to_predict = DL.var_data_blocks.shape[0] * DL.var_data_blocks.shape[1] * DL.var_data_blocks.shape[2]
    logging.info(f"Number of items to predict: {num_items_to_predict}")
    predictions_list = []
    coords_list = []    
    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_items_to_predict, desc="Inference"):
            data = batch[0]
            coords = batch[1]
            data = data.to(device)
            prediction = model(data)
            predictions_list.append(prediction.cpu().numpy())
            coords_list.append(coords)
    
    predictions = np.concatenate(predictions_list, axis=0)

    result_df = pd.DataFrame({"time": [item[2] for item in coords_list],
                            "lat": [item[0] for item in coords_list],
                            "lon": [item[1] for item in coords_list],
                            "prediction": predictions.flatten()
                           })
    

    os.makedirs(cfg.path_to_predictions, exist_ok=True)
    result_df.to_csv(os.path.join(cfg.path_to_predictions, "result.csv"), index=False)
    logging.info(f"Predictions shape: {predictions.shape}")
    logging.info(f"Predictions max value: {predictions.max()}, min value: {predictions.min()}")
    logging.info(f"Predictions mean value: {predictions.mean()}, std value: {predictions.std()}")
   
    plot_prediction(cfg, result_df, 3)


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs/infer_configs"), config_name="cmip5_w_eval.yaml")
def main(cfg: DictConfig):    
    test(cfg)
    logging.info('Inference finished!')


if __name__ == "__main__":      
    main()
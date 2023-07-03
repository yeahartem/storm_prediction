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
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')



def prepare_data(cfg: DictConfig):
    """Prepare data for inference"""
    var_data, time_coords, lat_coords, lon_coords = load_dataset(cfg, time_slices = 500)
    var_data_blocks = DataPreLoader.data_to_blocks(var_data, cfg.time_window, cfg.half_side_size)
    
    time_coords = time_coords[cfg.time_window//2:len(time_coords) - cfg.time_window//2]
    lat_coords = lat_coords[cfg.half_side_size:len(lat_coords) - cfg.half_side_size]
    lon_coords = lon_coords[cfg.half_side_size:len(lon_coords) - cfg.half_side_size]
    assert var_data_blocks.shape[0] == len(lat_coords)
    assert var_data_blocks.shape[1] == len(lon_coords)
    assert var_data_blocks.shape[2] == len(time_coords)
    return var_data_blocks, time_coords, lat_coords, lon_coords


def batch_generator(var_data_blocks, time_coords, lat_coords, lon_coords, batch_size):
    """Generate batches for inference"""

    for index in np.ndindex(var_data_blocks.shape[0], var_data_blocks.shape[1], var_data_blocks.shape[2]):
        batch = var_data_blocks[index[0], index[1], index[2]]
        batch = torch.from_numpy(batch).to(torch.float16) #torch.float32
        batch = batch.unsqueeze(0)
        yield batch, lat_coords[index[0]], lon_coords[index[1]], time_coords[index[2]], 



def test(cfg: DictConfig) -> None:        

    model = WindNetPL.load_from_checkpoint(cfg.path_to_checkpoint, cfg=cfg).half()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    var_data_blocks, time_coords, lat_coords, lon_coords = prepare_data(cfg)
    dataloader = batch_generator(var_data_blocks, time_coords, lat_coords, lon_coords, cfg.batch_size)
    num_items_to_predict = var_data_blocks.shape[0] * var_data_blocks.shape[1] * var_data_blocks.shape[2]
    logging.info(f"Number of items to predict: {num_items_to_predict}")
    predictions_list = []
    coords_list = []    
    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_items_to_predict, desc="Inference"):
            data, lat, lon, t = batch
            data = data.to(device)
            prediction = model(data)
            predictions_list.append(prediction.cpu().numpy())
            coords_list.append([lat, lon, t])
    

@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs/infer_configs"), config_name="cmip5_w_eval.yaml")
def main(cfg: DictConfig):    
    test(cfg)
    logging.info('Inference finished!')


if __name__ == "__main__":      
    main()
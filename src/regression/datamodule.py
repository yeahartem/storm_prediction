import sys,os
sys.path.append(os.getcwd())
import logging
import numpy as np
import pytorch_lightning as pl
import torchvision  
from torch.utils.data import DataLoader, Dataset
import torch
from omegaconf import DictConfig
from src.regression.data_load import DataPreLoader
from src.utils.norm_values import mean_channels_cmip6, std_channels_cmip6


class WindDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, test=False):
        super().__init__()
        self.cfg = cfg      
        self.DPL = DataPreLoader(cfg)
        
        if self.cfg.train.normalize:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Normalize(mean=mean_channels_cmip6, std=std_channels_cmip6),
                ]
            )
        else: self.transform = None
        
    def setup(self, stage=None):
        if self.cfg.train.use_elevation:
            logging.info("Using elevation data")
            DatasetClass = XarrayDatasetElev
        else:
            logging.info("Not using elevation data")
            DatasetClass = XarrayDataset

        if stage == "fit" or stage is None:
            self.dataset_train = DatasetClass(self.DPL, test=False)
            self.dataset_val = DatasetClass(self.DPL, test=True)
        if stage == "test" or stage is None:
            self.dataset_test = DatasetClass(self.DPL, test=True)


    def train_dataloader(self):
        return DataLoader(dataset=self.dataset_train, batch_size=self.cfg.train.batch_size, num_workers=self.cfg.train.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.dataset_val, batch_size=self.cfg.train.batch_size, num_workers=self.cfg.train.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.dataset_test, batch_size=self.cfg.train.batch_size, num_workers=self.cfg.train.num_workers, pin_memory=True)


class XarrayDataset(Dataset):
    def __init__(self, DPL, test=False, dtype=torch.float16,):
        self.cfg = DPL.cfg
        self.dataset_torch = DPL.dataset_torch
        if test:
            self.data_idxs = DPL.test_data_idxs
        else:
            self.data_idxs = DPL.train_data_idxs
        self.dtype = dtype

        logging.info(f"Sample shape is {self.get_sample_shape(10)}")

    def get_sample_shape(self, idx): 
        lat_index, lon_index, time_index, y = self.data_idxs[:, idx]
        X = self.dataset_torch[:,
                               slice(time_index - self.cfg.time_window//2, time_index + self.cfg.time_window//2 + 1),
                               slice(lat_index - self.cfg.half_side_size, lat_index + self.cfg.half_side_size + 1),
                               slice(lon_index - self.cfg.half_side_size, lon_index + self.cfg.half_side_size + 1),
                               ]
        return X.shape

    def __len__(self):
        return self.data_idxs.shape[1]

    def __getitem__(self, idx):
        lat_index, lon_index, time_index, y = self.data_idxs[:, idx]
        X = self.dataset_torch[:,
                               slice(time_index - self.cfg.time_window//2, time_index + self.cfg.time_window//2 + 1),
                               slice(lat_index - self.cfg.half_side_size, lat_index + self.cfg.half_side_size + 1),
                               slice(lon_index - self.cfg.half_side_size, lon_index + self.cfg.half_side_size + 1),
                               ]
        y = torch.tensor(y, dtype=self.dtype)
        return X, y
    

class XarrayDatasetElev(XarrayDataset):
    def __init__(self, DPL, test=False, dtype=torch.float16, ):
        super(XarrayDatasetElev, self).__init__(DPL, dtype)
        self.elevation_torch = DPL.elevation_torch
        self.elev_hss = DPL.elev_hss
        self.r_lat = DPL.r_lat
        self.r_lon = DPL.r_lon
        self.shift_clim = DPL.shift
        self.shift_elev = DPL.shift_elev

    def __getitem__(self, idx):
        lat_index, lon_index, time_index, y = self.data_idxs[:, idx]
        X = self.dataset_torch[:,
                               slice(time_index - self.cfg.time_window//2, time_index + self.cfg.time_window//2 + 1),
                               slice(lat_index - self.cfg.half_side_size, lat_index + self.cfg.half_side_size + 1),
                               slice(lon_index - self.cfg.half_side_size, lon_index + self.cfg.half_side_size + 1),
                               ]
        # print([time_index, lat_index, lon_index])
        lat_index_elev = int((lat_index - self.shift_clim[0]) * self.r_lat) + self.shift_elev[0]
        lon_index_elev = int((lon_index - self.shift_clim[1]) * self.r_lon) + self.shift_elev[1]
        X_elev = self.elevation_torch[
                                    slice(lat_index_elev - self.elev_hss, lat_index_elev + self.elev_hss + 1),
                                    slice(lon_index_elev - self.elev_hss, lon_index_elev + self.elev_hss + 1),
                                    ]
        X_elev = X_elev.view(1, X_elev.shape[-2], X_elev.shape[-1])
        y = torch.tensor(y, dtype=self.dtype)
        return (X, X_elev), y    
    
if __name__ == '__main__':
    pass
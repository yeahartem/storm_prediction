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


class WindDataModuleAlt(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig, test=False):
        super().__init__()
        self.cfg = cfg      
        self.DPL = DataPreLoader(cfg)
        
        if self.cfg.normalize:
            mean_channels = np.load(os.path.join(self.cfg.data_dir, f"mean_{cfg.precision}.npy"))
            std_channels = np.load(os.path.join(self.cfg.data_dir, f"std_{cfg.precision}.npy"))
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Normalize(mean=mean_channels, std=std_channels),
                ]
            )
        else: self.transform = None
        
    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset_train = XarrayDatasetAlt(cfg=self.cfg, data_idxs=self.DPL.train_data_idxs, 
                                                 dataset_torch=self.DPL.dataset_torch,
                                                 transforms=self.DPL.transform)
            self.dataset_val = XarrayDatasetAlt(cfg=self.cfg, data_idxs=self.DPL.test_data_idxs,
                                                dataset_torch=self.DPL.dataset_torch,
                                                transforms=self.DPL.transform)
        if stage == "test" or stage is None:
            self.dataset_test = XarrayDatasetAlt(cfg=self.cfg, data_idxs=self.DPL.test_data_idxs,
                                                dataset_torch=self.DPL.dataset_torch,
                                                transforms=self.DPL.transform)

    def train_dataloader(self):
        return DataLoader(dataset=self.dataset_train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(dataset=self.dataset_val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.dataset_test, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers, pin_memory=True)


class XarrayDatasetAlt(Dataset):
    def __init__(self, cfg, data_idxs, dataset_torch, dtype=torch.float16, transforms=None):
        self.cfg = cfg
        self.dataset_torch = dataset_torch
        self.data_idxs = data_idxs
        self.dtype = dtype
        self.transforms = transforms

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
    
if __name__ == '__main__':
    pass
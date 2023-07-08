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


class WindDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
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
            self.dataset_train = XarrayDataset(self.DPL.train_data_idxs, self.DPL.dataset_as_blocks,
                                                     transforms=self.DPL.transform)
            self.dataset_val = XarrayDataset(self.DPL.test_data_idxs, self.DPL.dataset_as_blocks,
                                                   transforms=self.DPL.transform)
        if stage == "test" or stage is None:
            self.dataset_test = XarrayDataset(self.DPL.test_data_idxs, self.DPL.dataset_as_blocks,
                                                    transforms=self.DPL.transform)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.cfg.batch_size, num_workers=self.cfg.num_workers)


class XarrayDataset(Dataset):
    def __init__(self, data_idxs, dataset_as_blocks, dtype=torch.float16, transforms=None):
        self.dataset_as_blocks = dataset_as_blocks
        self.data_idxs = data_idxs
        self.dtype = dtype
        self.transforms = transforms

    def __len__(self):
        return self.data_idxs.shape[1]

    def __getitem__(self, idx):
        lat, lon, date, y = self.data_idxs[:, idx]
        X = self.dataset_as_blocks[lat, lon, date]
        y = torch.tensor(y, dtype=self.dtype)
        # y = torch.add(torch.div(y, self.target_max), -1.0 * self.target_min)  
        X = torch.tensor(X, dtype=self.dtype)
        if self.transforms:
            X = self.transforms(X)
        return X, y


if __name__ == '__main__':
    pass
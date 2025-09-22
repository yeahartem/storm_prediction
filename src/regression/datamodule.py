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
from denseweight import DenseWeight

class WindDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
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
            # DatasetClass = XarrayDatasetElev
            raise NotImplementedError
        else:
            logging.info("Not using elevation data")
            DatasetClass = XarrayDataset

        if stage == "fit" or stage is None:
            self.dataset_train = DatasetClass(DPL=self.DPL, test=False)
            self.dataset_val = DatasetClass(DPL=self.DPL, test=True)
        if stage == "test":
            self.dataset_test = DatasetClass(DPL=self.DPL, test=True)


    def train_dataloader(self):
        return DataLoader(dataset=self.dataset_train,
                          batch_size=self.cfg.train.batch_size,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True,
                          drop_last=True)   # Чтобы не было ошибки ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 70])

    def val_dataloader(self):
        return DataLoader(dataset=self.dataset_val,
                          batch_size=self.cfg.train.batch_size,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.dataset_test,
                          batch_size=self.cfg.train.batch_size,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          )


class XarrayDataset(Dataset):
    def __init__(self, DPL, test=False, dtype=torch.float32):
        self.cfg = DPL.cfg
        self.dataset_torch = DPL.dataset_torch
        if test:
            self.data_idxs = DPL.test_data_idxs
            self.dense_weighter = None # Для тестового набора нам это не нужно
            logging.info("Test dataloader init")
        else:
            self.data_idxs = DPL.train_data_idxs
            # np.save("data_idxs_for_debug.npy", self.data_idxs)
            if self.cfg.train.loss_name=='MSELoss_Dense' or self.cfg.train.loss_name=='L1Loss_Dense':
                y_denseweight = self.data_idxs[7, :]
                # y_denseweight = self.data_idxs[7:, :].flatten() # (Не подходит ->) для Quantile Regression квантильная регрессия. Потому что надо обучать на распределении 0.96 квантиля, чтобы на выбросы не обращать внимания
                # --- ОТЛАДОЧНЫЙ ПРИНТ №1 ---
                print("\n--- DEBUG: Data for DenseWeight.fit() ---")
                print(f"Shape of targets: {y_denseweight.shape}")
                print(f"Min: {y_denseweight.min()}, Max: {y_denseweight.max()}")
                print(f"Sample 10 targets: {y_denseweight[:10]}")
                print("----------------------------------------\n")
                # --- КОНЕЦ ПРИНТА ---
                dw = DenseWeight(alpha=1.0)
                dw.fit(y_denseweight)
                self.dense_weighter = dw
            else:
                self.dense_weighter = None
            
            logging.info("Train dataloader init")
        self.dtype = dtype

        logging.info(f"Sample shape is {self.get_sample_shape(10)}")
# ====================================================
        # lat_index, lon_index, time_index, y_denseweight = self.data_idxs
        # # Define DenseWeight
        # dw = DenseWeight(alpha=1.0)
        # # Fit DenseWeight and get the weights for the 1000 samples
        # self.weights = dw.fit(y_denseweight)
        
    def get_sample_shape(self, idx): 
        lat_index, lon_index, time_index, *y = self.data_idxs[:, idx]
        lat_index = int(lat_index)
        lon_index = int(lon_index)
        time_index = int(time_index)
        X = self.dataset_torch[:,
                               slice(time_index - self.cfg.time_window//2, time_index + self.cfg.time_window//2 + 1),
                               slice(lat_index - self.cfg.half_side_size, lat_index + self.cfg.half_side_size + 1),
                               slice(lon_index - self.cfg.half_side_size, lon_index + self.cfg.half_side_size + 1),
                               ]
        return X.shape

    def __len__(self):
        return self.data_idxs.shape[1] # у должны быть равны длине вот этого

    def __getitem__(self, idx):
        # lat_index, lon_index, time_index, time_pos, time_pos_m, lat_pos, lon_pos, y = self.data_idxs[:8, idx]
        lat_index, lon_index, time_index, time_pos, time_pos_m, lat_pos, lon_pos, *y = self.data_idxs[:, idx] # Quantile regression

        lat_index = int(lat_index)
        lon_index = int(lon_index)
        time_index = int(time_index)
        X = self.dataset_torch[:,
                               slice(time_index - self.cfg.time_window//2, time_index + self.cfg.time_window//2 + 1),
                               slice(lat_index - self.cfg.half_side_size, lat_index + self.cfg.half_side_size + 1),
                               slice(lon_index - self.cfg.half_side_size, lon_index + self.cfg.half_side_size + 1),
                               ]
        # assert X.shape[-1] == 95, f"idxs {lat_index, lon_index, time_index}"
        pos = torch.tensor([time_pos, time_pos_m, lat_pos, lon_pos], dtype=self.dtype)
        pos = pos.expand(self.cfg.time_window, 4)
        y = torch.tensor(y, dtype=self.dtype)
        
        # Проверяем, есть ли у нас обученный "взвешиватель"
        if self.dense_weighter is not None:
            # Получаем веса для таргета текущего примера
            weights = self.dense_weighter(y.cpu().numpy())
            weights_tensor = torch.tensor(weights, dtype=self.dtype)
            # --- ОТЛАДОЧНЫЙ ПРИНТ №2 ---
            if idx < 5: # Печатаем только для первых 5 примеров
                print(f"\n--- DEBUG: __getitem__ idx={idx} ---")
                print(f"Target values (shape {y.shape}): {np.round(y.numpy(), 2)}")
                print(f"Calculated weights (shape {weights_tensor.shape}): {np.round(weights_tensor.numpy(), 2)}")
                print("----------------------------------")
            # --- КОНЕЦ ПРИНТА ---
        else:
            # Если не используем DenseWeight, создаем тензор-пустышку
            weights_tensor = torch.tensor([1.0], dtype=self.dtype) # Просто чтобы что-то вернуть

        return [X, pos], y, weights_tensor
    

if __name__ == '__main__':
    pass
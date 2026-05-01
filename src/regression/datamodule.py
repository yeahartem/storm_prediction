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
        
        if self.cfg.train.normalize: # Похоже нигде не используется
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

        # split='train' → DPL.train_data_idxs (time < start_of_val)
        # split='val'   → DPL.val_data_idxs   (start_of_val <= time < start_of_test)
        # split='test'  → DPL.test_data_idxs  (time >= start_of_test)
        # If start_of_val is not set in the config, val_data_idxs == test_data_idxs
        # (legacy leaky behaviour). data_load.py emits a warning in that case.
        if stage == "fit" or stage is None:
            self.dataset_train = DatasetClass(DPL=self.DPL, split='train')
            self.dataset_val   = DatasetClass(DPL=self.DPL, split='val')
        if stage == "test":
            self.dataset_test  = DatasetClass(DPL=self.DPL, split='test')


    def train_dataloader(self):
        return DataLoader(dataset=self.dataset_train,
                          batch_size=self.cfg.train.batch_size,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)   # Чтобы не было ошибки ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 70])

    def val_dataloader(self):
        # shuffle=False: the dataset already holds a deterministic subset selected
        # via val_subset_size/val_subset_seed (see XarrayDataset.__init__). This
        # gives a stable, reproducible val/loss every epoch — clean signal for
        # early stopping.
        return DataLoader(dataset=self.dataset_val,
                          batch_size=self.cfg.train.batch_size,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True,
                          shuffle=False,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(dataset=self.dataset_test,
                          batch_size=self.cfg.train.batch_size,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          )


# РАЗБИРАЕМ ТУТ, СЕЙЧАС РАЗБИРАЕМ DataPreLoader , это DPL, там происходит разбивка на тест по split_date


class XarrayDataset(Dataset):
    def __init__(self, DPL, split='train', test=None, dtype=torch.float32):
        """
        split: 'train' | 'val' | 'test'
        test:  legacy boolean kept for backward compatibility. If supplied, it
               overrides `split` (test=True -> 'test', test=False -> 'train').
        """
        if test is not None:
            split = 'test' if test else 'train'
        if split not in ('train', 'val', 'test'):
            raise ValueError(f"split must be 'train' | 'val' | 'test', got {split!r}")

        self.cfg = DPL.cfg
        self.dataset_torch = DPL.dataset_torch
        self.split = split

        if split == 'test':
            self.data_idxs = DPL.test_data_idxs
            self.dense_weighter = None
            self._DPL = None
            logging.info(f"Test dataloader init: {self.data_idxs.shape[1]} samples")
        elif split == 'val':
            # val_data_idxs is created by DataPreLoader.target_df_to_array().
            # Falls back to test_data_idxs if start_of_val was not configured.
            raw = getattr(DPL, 'val_data_idxs', DPL.test_data_idxs)
            # ---- Deterministic stratified val subset ----
            # Two motivations:
            #   1. Determinism. shuffle=True + limit_val_batches sampled fresh
            #      32K random rows every epoch, so val/loss was noisy and
            #      early-stop selected almost-randomly. Picking a fixed subset
            #      once removes that noise.
            #   2. Geographic balance. Uniform sampling over rows is biased
            #      toward stations that have more observations (dense network
            #      regions dominate). Stratifying by (lat_idx, lon_idx) gives
            #      every station the same vote and matches the geography of
            #      the test set more faithfully.
            #
            # Default: per-station random K samples — every val station
            # contributes, K is chosen so total ≈ desired size.
            # Fallback (val_samples_per_station <= 0): old uniform sampling
            # by val_subset_size.
            val_samples_per_station = int(self.cfg.train.get('val_samples_per_station', 6) or 0)
            val_subset_size = int(self.cfg.train.get('val_subset_size', 0) or 0)
            val_subset_seed = int(self.cfg.train.get('val_subset_seed', 42))
            n_total = raw.shape[1]

            if val_samples_per_station > 0:
                # Stratify by station key. Sort once, walk groups, pick K per group.
                keys = raw[0].astype(np.int64) * 1_000_000 + raw[1].astype(np.int64)
                order = np.argsort(keys, kind='stable')
                sorted_keys = keys[order]
                change = np.flatnonzero(np.diff(sorted_keys)) + 1
                edges = np.concatenate(([0], change, [n_total]))
                rng = np.random.default_rng(val_subset_seed)
                picks = []
                K = val_samples_per_station
                for i in range(len(edges) - 1):
                    s, e = edges[i], edges[i + 1]
                    n = e - s
                    if n <= K:
                        picks.append(order[s:e])
                    else:
                        sel = rng.choice(n, size=K, replace=False)
                        picks.append(order[s:e][sel])
                final_idx = np.sort(np.concatenate(picks))
                self.data_idxs = raw[:, final_idx]
                n_stations = len(edges) - 1
                logging.info(
                    f"Val dataloader init: stratified subset — "
                    f"{self.data_idxs.shape[1]} samples from {n_stations} stations "
                    f"(K={K}/station, seed={val_subset_seed}, full val={n_total})"
                )
            elif val_subset_size > 0 and n_total > val_subset_size:
                rng = np.random.default_rng(val_subset_seed)
                perm = rng.permutation(n_total)[:val_subset_size]
                self.data_idxs = raw[:, np.sort(perm)]
                logging.info(
                    f"Val dataloader init: uniform random subset {self.data_idxs.shape[1]} of {n_total} "
                    f"(seed={val_subset_seed})"
                )
            else:
                self.data_idxs = raw
                logging.info(f"Val dataloader init: full {self.data_idxs.shape[1]} samples")
            self.dense_weighter = None
            self._DPL = None
        else:  # train
            self._DPL = DPL  # keep reference for per-epoch resampling
            self.data_idxs = DPL.train_data_idxs
            # np.save("data_idxs_for_debug.npy", self.data_idxs)
            if self.cfg.train.loss_name=='MSELoss_Dense' or self.cfg.train.loss_name=='L1Loss_Dense':
                y_denseweight = self.data_idxs[7, :]
                # y_denseweight = self.data_idxs[7:, :].flatten() # для Quantile Regression квантильная регрессия
                dw = DenseWeight(alpha=1.0)
                dw.fit(y_denseweight)
                self.dense_weighter = dw
            else:
                self.dense_weighter = None
            logging.info(f"Train dataloader init: {self.data_idxs.shape[1]} samples")
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

    @property
    def _active_idxs(self):
        """Return current data_idxs, dynamically updated each epoch if resampling."""
        if self._DPL is not None:
            return self._DPL.train_data_idxs
        return self.data_idxs

    def __len__(self):
        return self._active_idxs.shape[1]

    def __getitem__(self, idx):
        data_idxs = self._active_idxs
        lat_index, lon_index, time_index, time_pos, time_pos_m, lat_pos, lon_pos, y = data_idxs[:8, idx]
        # lat_index, lon_index, time_index, time_pos, time_pos_m, lat_pos, lon_pos, *y = self.data_idxs[:, idx] # Quantile regression
        # Row 14: per-station effective threshold (max(p95_station, abs_threshold))
        station_threshold = float(data_idxs[8, idx]) if data_idxs.shape[0] > 8 else float(self.cfg.train.target_threshold)

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
        else:
            # Если не используем DenseWeight, создаем тензор-пустышку
            weights_tensor = torch.tensor([1.0], dtype=self.dtype) # Просто чтобы что-то вернуть

        return [X, pos], y, weights_tensor, torch.tensor(station_threshold, dtype=self.dtype)

class XarrayDatasetElev(XarrayDataset):
    """Same as XarrayDataset but appends elevation as a 5th channel.

    Elevation is regridded to the CMIP6 grid, so indices are identical.
    DPL.elevation_torch shape: (lat_padded, lon_padded) – no time dim.
    Output X shape: (5, time_window, patch_h, patch_w)
    """
    def __init__(self, DPL, split='train', test=None, dtype=torch.float32):
        super().__init__(DPL, split=split, test=test, dtype=dtype)
        self.elevation_torch = DPL.elevation_torch  # (lat_padded, lon_padded)

    def __getitem__(self, idx):
        data_idxs = self._active_idxs
        lat_index, lon_index, time_index, time_pos, time_pos_m, lat_pos, lon_pos, y = data_idxs[:8, idx]
        station_threshold = float(data_idxs[8, idx]) if data_idxs.shape[0] > 8 else float(self.cfg.train.target_threshold)
        lat_index = int(lat_index)
        lon_index = int(lon_index)
        time_index = int(time_index)

        X = self.dataset_torch[:,
                               slice(time_index - self.cfg.time_window//2, time_index + self.cfg.time_window//2 + 1),
                               slice(lat_index - self.cfg.half_side_size, lat_index + self.cfg.half_side_size + 1),
                               slice(lon_index - self.cfg.half_side_size, lon_index + self.cfg.half_side_size + 1),
                               ]  # (4, time_window, patch_h, patch_w)

        X_elev = self.elevation_torch[
                               slice(lat_index - self.cfg.half_side_size, lat_index + self.cfg.half_side_size + 1),
                               slice(lon_index - self.cfg.half_side_size, lon_index + self.cfg.half_side_size + 1),
                               ]  # (patch_h, patch_w)
        # broadcast elevation across time window -> (1, time_window, patch_h, patch_w)
        X_elev = X_elev.unsqueeze(0).unsqueeze(0).expand(1, X.shape[1], -1, -1)
        X = torch.cat([X, X_elev], dim=0)  # (5, time_window, patch_h, patch_w)

        pos = torch.tensor([time_pos, time_pos_m, lat_pos, lon_pos], dtype=self.dtype)
        pos = pos.expand(self.cfg.time_window, 4)
        y = torch.tensor(y, dtype=self.dtype)
        weights_tensor = torch.tensor([1.0], dtype=self.dtype)
        return [X, pos], y, weights_tensor, torch.tensor(station_threshold, dtype=self.dtype)


if __name__ == '__main__':
    pass

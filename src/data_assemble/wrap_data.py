import os
import random
import pickle
from click import pass_context
import numpy as np
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import torch
from tqdm import tqdm

import geopandas as gpd
from shapely.geometry import Point, Polygon, box, LineString

import pandas as pd
import matplotlib.pyplot as plt


class WindDataModule(pl.LightningDataModule):
    def __init__(
        self, X: dict, y: dict, batch_size: int = 128, downsample: bool = True
    ):
        super().__init__()
        self.batch_size = batch_size
        self.X_train, self.X_val, self.X_test = (
            torch.tensor(X["Train"], dtype=torch.double),
            torch.tensor(X["Val"], dtype=torch.double),
            torch.tensor(X["Test"], dtype=torch.double),
        )
        self.y_train, self.y_val, self.y_test = (
            torch.tensor(y["Train"], dtype=torch.double),
            torch.tensor(y["Val"], dtype=torch.double),
            torch.tensor(y["Test"], dtype=torch.double),
        )
        mean_channels = self.X_train.mean(dim=[0, -1, -2])
        std_channels = self.X_train.std(dim=[0, -1, -2])
        self.transform = transforms.Compose(
            [
                transforms.Normalize(mean=mean_channels, std=std_channels),
            ]
        )

        self.dl_dict = {"batch_size": self.batch_size}

        if downsample:
            def make_weights_for_balanced_classes(images, nclasses):
                n_images = len(images)
                count_per_class = [0] * nclasses
                for _, image_class in images:
                    count_per_class[image_class] += 1
                weight_per_class = [0.] * nclasses
                for i in range(nclasses):
                    weight_per_class[i] = float(n_images) / float(count_per_class[i])
                weights = [0] * n_images
                for idx, (image, image_class) in enumerate(images):
                    weights[idx] = weight_per_class[image_class]
                return weights
            weights = make_weights_for_balanced_classes(self.X_train, 2)                                                                
            weights = torch.tensor(weights, dtype=self.X_train.dtype, device=self.X_train.device)                                       
            self.sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))                     
                                                                                            
        else:
            self.sampler = None

    def prepare_data(self):
        if type(self.y_train) == torch.Tensor and len(self.y_train.shape) == 2:
            pass
        else:
            self.y_train = torch.tensor(self.y_train, dtype=torch.long)
            self.y_val = torch.tensor(self.y_val, dtype=torch.long)
            self.y_test = torch.tensor(self.y_test, dtype=torch.long)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.dataset_train = TensorDataset(
                self.transform(self.X_train), torch.tensor(self.y_train)
            )
            self.dataset_val = TensorDataset(
                self.transform(self.X_val), torch.tensor(self.y_val)
            )

        if stage == "test" or stage is None:
            self.dataset_test = TensorDataset(
                self.transform(self.X_test), torch.tensor(self.y_test)
            )

    def train_dataloader(self):
        return DataLoader(self.dataset_train, sampler=self.sampler, **self.dl_dict)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, sampler=self.sampler, **self.dl_dict)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, sampler=self.sampler, **self.dl_dict)


def train_val_test_split(
    path_to_data: str,
    train: float = 0.5,
    val: float = 0.25,
    test: float = 0.25,
    verbose: bool = False,
) -> dict:
    """randomly splits weather stations to train, val, test in proportions given

    Args:
        path_to_data (str): path to folder which contains folder with preparsed numpy objects from stations
        train (float, optional): train weather stations share. Defaults to 0.5.
        val (float, optional): val weather stations share. Defaults to 0.25.
        test (float, optional): test weather stations share. Defaults to 0.25.
        verbose (bool, optional): if to print out the result of split. Defaults to False.

    Returns:
        dict: [description]
    """
    stations = os.listdir(path_to_data)
    random.shuffle(stations)
    partition = {"train_share": train, "val_share": val, "test_share": test}
    train_len = int(len(stations) * partition["train_share"])
    val_len = int(len(stations) * partition["val_share"])
    test_len = int(len(stations) * partition["test_share"])
    train_sts, val_sts, test_sts = (
        stations[:train_len],
        stations[train_len : train_len + val_len],
        stations[train_len + val_len :],
    )
    st_split_dict = {"Train": train_sts, "Val": val_sts, "Test": test_sts}
    if verbose:
        print(st_split_dict)
    return st_split_dict


def extract_splitted_data(path_to_dump: str, st_split_dict: dict, return_xarray: bool = False) -> tuple:
    """extracts X, y, splitted into train, val, test

    Args:
        path_to_dump (str): path to folder which contains folder with preparsed numpy objects from stations
        st_split_dict (dict): division by stations' names into train, val, test

    Returns:
        tuple: (X - keys = train, val, test. values = objects; y - similarly)
    """
    X = {}
    y = {}
    for split_part, sts in st_split_dict.items():
        X_split = []
        y_split = []
        for st in sts:
            st_dir = os.path.join(path_to_dump, st)
            with open(os.path.join(st_dir, "objects.npy"), "rb") as f:
                X_ = pickle.load(f)

            X_split.append(X_)
            try:
                with open(os.path.join(st_dir, "target.npy"), "rb") as f:
                    y_ = pickle.load(f)
                y_split.append(y_)
            except FileNotFoundError:
                y_split.append([])
        if return_xarray:
            X[split_part] = X_split
        else:
            X[split_part] = np.concatenate(X_split)
        y[split_part] = np.concatenate(y_split)
    return X, y


def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out

def ewma_vectorized_2d(data, alpha, axis=None, offset=None, dtype=None, order='C', out=None):
    """
    Calculates the exponential moving average over a given axis.
    :param data: Input data, must be 1D or 2D array.
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param axis: The axis to apply the moving average on.
        If axis==None, the data is flattened.
    :param offset: optional
        The offset for the moving average. Must be scalar or a
        vector with one element for each row of data. If set to None,
        defaults to the first value of each row.
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Ignored if axis is not None.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the desired output. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    assert data.ndim <= 2

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if axis is None or data.ndim < 2:
        # use 1D version
        if isinstance(offset, np.ndarray):
            offset = offset[0]
        print(order)
        return ewma_vectorized(data, alpha, offset, dtype=dtype, order=order,
                               out=out)

    assert -data.ndim <= axis < data.ndim

    # create reshaped data views
    out_view = out
    if axis < 0:
        axis = data.ndim - int(axis)

    if axis == 0:
        # transpose data views so columns are treated as rows
        data = data.T
        out_view = out_view.T

    if offset is None:
        # use the first element of each row as the offset
        offset = np.copy(data[:, 0])
    elif np.size(offset) == 1:
        offset = np.reshape(offset, (1,))

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # calculate the moving average
    row_size = data.shape[1]
    row_n = data.shape[0]
    scaling_factors = np.power(1. - alpha, np.arange(row_size + 1, dtype=dtype),
                               dtype=dtype)
    # create a scaled cumulative sum array
    np.multiply(
        data,
        np.multiply(alpha * scaling_factors[-2], np.ones((row_n, 1), dtype=dtype),
                    dtype=dtype)
        / scaling_factors[np.newaxis, :-1],
        dtype=dtype, out=out_view
    )
    np.cumsum(out_view, axis=1, dtype=dtype, out=out_view)
    out_view /= scaling_factors[np.newaxis, -2::-1]

    if not (np.size(offset) == 1 and offset == 0):
        offset = offset.astype(dtype, copy=False)
        # add the offsets to the scaled cumulative sums
        out_view += offset[:, np.newaxis] * scaling_factors[np.newaxis, 1:]

    return out


def map_to_pandas(grid, x_axis, y_axis, t_axis, start_date, day_interval=1):
    df = pd.DataFrame(np.zeros((len(list(range(grid.shape[0]))[::day_interval])*grid[0,3:-3,3:-3].shape[0]*grid[0,3:-3,3:-3].shape[1],4)))
    df.rename(columns={0:'date', 1:'lon', 2:'lat', 3:'value'}, inplace=True)
    k=0
    for t in tqdm(list(range(grid.shape[0]))[::day_interval]):
        for i in range(3,grid[t,:,:].shape[0]-3):
            df.iloc[k:k+grid[t,3:-3,3:-3].shape[1],0]=t_axis[t] #str(start_date + pd.DateOffset(1) * t)
            df.iloc[k:k+grid[t,3:-3,3:-3].shape[1],1]=x_axis[:]
            df.iloc[k:k+grid[t,3:-3,3:-3].shape[1],2]=y_axis[i-3] 
            df.iloc[k:k+grid[t,3:-3,3:-3].shape[1],3]=grid[t,i,3:-3]
            k+=grid[t,3:-3,3:-3].shape[1]
    df=df.reset_index()
    return df


def plot_map(df_year, column, epsg=3035, part_world_to_plot='world', img_path=None, text=None, show=True, vmin=None, vmax=None):
    
    dx, dy = .25, .25
    geometry = [Polygon([(x-dx, y-dy),
                         (x+dx, y-dy),
                         (x+dx, y+dy),
                         (x-dx, y+dy)]) for x, y in zip(df_year.lon, df_year.lat)]

    geodf = gpd.GeoDataFrame(df_year, crs = 4326, geometry=geometry)
    
    
    polygon = box(-180, 23, 180, 90)

    
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = gpd.clip(world, polygon)
    
    if part_world_to_plot == 'Florida':
        Fl_polygon = box(-92, 22, -75, 34)
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world = gpd.clip(world, Fl_polygon)
        basemap = world
    if part_world_to_plot == 'KK_Belg_Rost':
        Fl_polygon = box(23, 40, 46, 53)
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world = gpd.clip(world, Fl_polygon)
        basemap = world
    if part_world_to_plot == 'KK':
        KK_polygon = box(23, 40, 45, 50)
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world = gpd.clip(world, KK_polygon)
        basemap = world
    if part_world_to_plot == 'nothern':
        basemap = world.cx[:, 52:]
    elif part_world_to_plot == 'world':
        basemap = world
    elif part_world_to_plot == 'europe':
        basemap = world[world.continent == 'Europe'] 
        
        a = basemap.geometry.iloc[0]
        def restriction(x):
            x = x.bounds
            l1, r1 = x[0], x[2]
            l2, r2 = x[1], x[3]

            return (l1 > a.bounds[0] and r1 < a.bounds[2]) and (l2 > a.bounds[1] and r2 < a.bounds[3])

        geodf = geodf[geodf.geometry.apply(restriction) == True]
        
    elif part_world_to_plot == 'russia':
        europe = world[world.continent == 'Europe'] 
        basemap = europe[europe.name =='Russia']
        
        a = basemap.geometry.iloc[0]
        print(a.bounds)
        def restriction(x):
            x = x.bounds
            l1, r1 = x[0], x[2]
            l2, r2 = x[1], x[3]

            return (l1 > a.bounds[0] and r1 < a.bounds[2]) and (l2 > a.bounds[1] and r2 < a.bounds[3])

        geodf = geodf[geodf.geometry.apply(restriction) == True]
        
    basemap = basemap.to_crs(epsg=epsg)
    geodf = geodf.to_crs(epsg=epsg)
    
    f = plt.figure(dpi = 200)
    ax = f.gca()
    ax.set_facecolor('white')

    if vmin is None:
        vmin = df_year[column].min()
    
    if vmax is None:
        vmax = df_year[column].max()

    basemap.plot(ax=ax, color = '#CACACA')
    geodf.plot(ax = ax, column = column, legend=True,
              legend_kwds={'orientation': "horizontal",
                            'shrink': 0.8}, #cax=cax,
              vmin=vmin, vmax=vmax, aspect=1, edgecolor="white", linewidth=0.05, alpha=1.)

    if text is None:
        text = column
    plt.title(text)
    plt.axis('equal')
    plt.axis('off')
    if img_path is not None:
        plt.savefig(img_path, transparent=False, facecolor='w')
    if show:
        plt.show()

def plot_grid(ax, crs, color='black', linewidth=.5, alpha=0.2, xmin=-180, 
              xmax=179.999, ymin=-90, ymax=90, n=10, width=15, height=15):
    cols = list(np.arange(xmin, xmax + width, width))
    rows = list(np.arange(ymin, ymax + height, height))

    xlin = np.linspace(xmin, xmax, n*len(cols))
    ylin = np.linspace(ymin, ymax, n*len(rows))
    lines  = [LineString(zip([x]*(n*len(rows)), ylin)) for x in cols]
    lines += [LineString(zip(xlin, [y]*(n*len(cols)))) for y in rows]

    grid = gpd.GeoDataFrame(lines, crs=4326, geometry=lines)
    grid.to_crs(crs, inplace=True)

    grid.plot(ax=ax, color=color, linewidth=linewidth, alpha=alpha)

def plot_points(df, epsg=3035):
    polygon = box(-180, 23, 180, 90)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = gpd.clip(world, polygon)

    basemap = world.cx[:, 52:]
    basemap = basemap.to_crs(epsg=epsg)

    f = plt.figure(dpi = 200)
    ax = f.gca()
    basemap.plot(ax=ax, color = '#CACACA')

    pts = gpd.points_from_xy(df.lon, df.lat)
    pts = gpd.GeoDataFrame(pts)
    pts = pts.set_geometry(0)
    pts.crs = 4326

    pts = pts.to_crs(basemap.crs)
    pts.plot(ax=ax, markersize=1, color='#373F51')
    plt.axis('off')
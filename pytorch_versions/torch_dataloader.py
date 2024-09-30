import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle as pkl


def NormalizeData(data, data_max=6,
                  data_min=-3):
    data_new = (data - data_min) / (data_max - data_min)
    return data_new


class prepare_pytorch_Dataset(Dataset):
    def __init__(self, opts, verbose=False, transform=None, target_transform=None):
        self.opts = opts
        list_IDs = np.loadtxt(self.opts['list_IDs_directory'], usecols=(0, 1), dtype=int)
        self.len_data = self.opts['data_size']
        self.list_IDs = list_IDs[:, 0]
        self.list_orders = list_IDs[:, 1]
        self.transform = transform
        self.target_transform = target_transform
        self.metadata = pd.read_csv(self.opts['metadata_directory'])
        self.batch_size = self.opts['batch_size']
        self.res_scale = self.opts['res_scale']
        self.crop_scale = self.opts['crop_scale']
        self.path = self.opts['data_path']
        self.indexes = np.arange(len(self.list_IDs), dtype=int)
        self.output_format = self.opts['output_format']
        self.include_lens_pos = self.opts['include_lens_pos']
        self.include_map_units = self.opts['include_map_units']
        self.dim = (int((self.opts['dim'] / self.res_scale) * self.crop_scale),
                    int((self.opts['dim'] / self.res_scale) * self.crop_scale))
        self.lens_pos_units = np.linspace(-500, 500, 1001)
        if self.include_map_units:
            self.dim_input = self.dim + 1
        else:
            self.dim_input = self.dim
        self.n_channels = self.opts['n_channel']
        self.shuffle = self.opts['shuffle']

        if self.include_lens_pos:
            self.n_channels = 2
        elif self.include_lens_pos and self.crop_scale != 1:
            print('Lens positions cannot be included for cropped maps ')
            exit()
        elif self.include_map_units and self.crop_scale != 1:
            print('Map units cannot be included for cropped maps ')
            exit()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        # Find list of IDs
        self.list_IDs_temp = self.list_IDs[index]
        conv_const = self.metadata.loc[self.metadata['ID'].isin(self.list_IDs_temp)]['const'].values
        X = []
        for i, ID in enumerate(self.list_IDs_temp):
            f = open(self.path + str(ID) + "/map.bin", "rb")
            map_tmp = np.fromfile(f, 'i', -1, "")
            maps = (np.reshape(map_tmp, (-1, 10000)))
            # order = self.list_orders_temp[i]
            tmp = maps[0::self.res_scale, 0::self.res_scale]
            tmp = np.log10(tmp * conv_const[i] + 0.004)
            X.append(NormalizeData(tmp, 6, -3))
        X = np.asarray(X).reshape((len(self.list_IDs_temp), 1, self.dim[0], self.dim[1]))
        return X, X

    def get_generator_indexes(self):
        return self.indexes

    def percent_selector(self, map, order):

        dim1 = int(map.shape[0] * self.crop_scale)
        dim2 = int(map.shape[0] - (map.shape[0] * self.crop_scale))

        if order == 1:
            return map[:dim1, :dim1]
        elif order == 2:
            return map[:dim1, dim2:]
        elif order == 3:
            return map[dim2:, :dim1]
        elif order == 4:
            return map[dim2:, dim2:]

    def lens_pos_map(self, lens_pos):
        lens_pos_map_ = np.zeros((len(self.lens_pos_units), len(self.lens_pos_units)))

        if len(lens_pos.shape) > 1:
            lens_poses = lens_pos[:, 0:2]
        elif lens_pos.shape[0] == 1:
            coordinate = lens_pos[0:2]
            lens_pos_map_[np.where(self.lens_pos_units == int(coordinate[0]))[0][0],
                          np.where(self.lens_pos_units == int(coordinate[1]))[0][0]] += 1.

            lens_pos_map_[np.where(self.lens_pos_units == int(coordinate[0]) - 1)[0][0],
                          np.where(self.lens_pos_units == int(coordinate[1]) + 1)[0][0]] += 1.

            lens_pos_map_[np.where(self.lens_pos_units == int(coordinate[0]) + 1)[0][0],
                          np.where(self.lens_pos_units == int(coordinate[1]) + 1)[0][0]] += 1.

            lens_pos_map_[np.where(self.lens_pos_units == int(coordinate[0]) - 1)[0][0],
                          np.where(self.lens_pos_units == int(coordinate[1]) - 1)[0][0]] += 1.

            lens_pos_map_[np.where(self.lens_pos_units == int(coordinate[0]) + 1)[0][0],
                          np.where(self.lens_pos_units == int(coordinate[1]) - 1)[0][0]] += 1.
            lens_pos_map_ = lens_pos_map_ / np.max(lens_pos_map_)
            return lens_pos_map_[:1000, :1000]
        else:
            return lens_pos_map_[:1000, :1000]
        for c, coordinate in enumerate(lens_poses):
            lens_pos_map_[np.where(self.lens_pos_units == int(coordinate[0]))[0][0],
                          np.where(self.lens_pos_units == int(coordinate[1]))[0][0]] += 1.

            lens_pos_map_[np.where(self.lens_pos_units == int(coordinate[0]) - 1)[0][0],
                          np.where(self.lens_pos_units == int(coordinate[1]) + 1)[0][0]] += 1.

            lens_pos_map_[np.where(self.lens_pos_units == int(coordinate[0]) + 1)[0][0],
                          np.where(self.lens_pos_units == int(coordinate[1]) + 1)[0][0]] += 1.

            lens_pos_map_[np.where(self.lens_pos_units == int(coordinate[0]) - 1)[0][0],
                          np.where(self.lens_pos_units == int(coordinate[1]) - 1)[0][0]] += 1.

            lens_pos_map_[np.where(self.lens_pos_units == int(coordinate[0]) + 1)[0][0],
                          np.where(self.lens_pos_units == int(coordinate[1]) - 1)[0][0]] += 1.

            lens_pos_map_ = lens_pos_map_ / np.max(lens_pos_map_)
            return lens_pos_map_[:1000, :1000]


def Data_Loader(opts, dict, params, verbose=False):  # transform = T.Resize(28)

    if verbose:
        print('Preparing the pytorch dataset')
    dataset = prepare_pytorch_Dataset2(dict, params)
    n = dataset.len_data
    batch = opts['batch_size']
    frac = opts['test_frac']
    train_set_size = ((int((1 - frac) * n)) - ((int((1 - frac) * n)) % batch))

    if verbose:
        print('Total sample size: ', n)
        print('Training sample size: ', train_set_size)
        print('Testing sample size: ', n - train_set_size)

    train_loader = DataLoader(dataset=dataset[:train_set_size][0],
                              batch_size=opts['batch_size'],
                              shuffle=True,
                              num_workers=opts['n_workers'])

    test_loader = DataLoader(dataset=dataset[train_set_size:][0],
                             batch_size=opts['batch_size'],
                             shuffle=False,
                             num_workers=opts['n_workers'])
    return train_loader, test_loader


def datareader(index, opts):
    path = opts['data_path_bulk'] + 'all_maps_batch' + str(index) + '.pkl'
    data_dict = pkl.load(open(path, "rb"))

    all_params = pd.read_csv(opts['metadata_directory'])

    true_params = []
    # maps = np.zeros((len(data_dict.keys()), 1000, 1000))
    for i, key in enumerate(data_dict.keys()):
        true_params.append([int(key), all_params.k[all_params.ID == key].values[0],
                            all_params.g[all_params.ID == key].values[0],
                            all_params.s[all_params.ID == key].values[0],
                            all_params.const[all_params.ID == key].values[0]])
        # data = data_dict[key].reshape((1000,1000))
        # data = np.log10(data * all_params.const[all_params.ID == key].values[0] + 0.001)
        # maps[i, :, :] = NormalizeData(data, 6, -3)

    return data_dict, np.asarray(true_params)

# (maps.reshape((len(maps), 1, 1000, 1000))).astype(np.float32)
class prepare_pytorch_Dataset2(Dataset):
    def __init__(self, data_dict, ids, transform=None, target_transform=None):
        self.ids = ids
        self.data_dict = data_dict
        self.len_data = len(ids)
        self.batch_size = 8
        self.input_keys = list(data_dict.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return (self.len_data % self.batch_size)

    def __getitem__(self, idx):
        keys = self.input_keys[idx]

        return self.__generatedata(keys)

    def __generatedata(self, keys):
        # print(len(keys))
        X = np.zeros((len(keys), 1, 1000, 1000))
        Y = np.zeros((len(keys), 1, 1000, 1000))
        for i, key in enumerate(keys):
            data = np.asarray(self.data_dict[key]).reshape((1000, 1000))
            data = np.log10(data * self.ids[i, 4] + 0.001)
            X[i, 0, :, :] = NormalizeData(data, 6, -3).astype(np.float32)
            Y[i, 0, :, :] = X[i, 0, :, :].astype(np.float32)
            if self.transform:
                X[i, 0, :, :] = self.transform(X[i, 0, :, :])
            if self.target_transform:
                Y[i, 0, :, :] = self.target_transform(Y[i, 0, :, :])
        return X, Y


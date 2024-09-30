"""Code from:

https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

"""

import keras
import numpy as np
import pandas as pd
from maps_util import NormalizeData, read_all_lens_pos
from training_utils import prepare_input_to_fit_keras_ADs, prepare_input_for_kgs_bt


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, list_IDs, batch_size=8, dim=10000, n_channels=1,
                 res_scale=10, crop_scale=1, path='./../../../fred/oz108/GERLUMPH_project/DATABASES/gerlumph_db/',
                 shuffle=True, conv_const='./../data/all_maps_meta_kgs.csv',
                 output_format='xx',
                 include_lens_pos=False,
                 include_map_units=False):
        """Initialization"""

        self.dim = (int((dim / res_scale) * crop_scale), int((dim / res_scale) * crop_scale))
        self.batch_size = batch_size
        self.list_IDs = list_IDs[:, 0]
        self.list_orders = list_IDs[:, 1]
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.res_scale = res_scale
        self.crop_scale = crop_scale
        self.path = path
        self.indexes = np.arange(len(self.list_IDs), dtype=int)
        self.on_epoch_end()
        self.conv_const_path = conv_const
        self.output_format = output_format
        self.include_lens_pos = include_lens_pos
        self.include_map_units = include_map_units
        self.lens_pos_units = np.linspace(-500, 500, 1001)
        if self.include_map_units:
            self.dim_input = self.dim + 1
        else:
            self.dim_input = self.dim

        if self.include_lens_pos:
            self.n_channels = 2
        elif self.include_lens_pos and self.crop_scale != 1:
            print('Lens positions cannot be included for cropped maps ')
            exit()
        elif self.include_map_units and self.crop_scale != 1:
            print('Map units cannot be included for cropped maps ')
            exit()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        params = all_params[indexes]
        X = np.zeros((self.batch_size, 2000))
        y = np.zeros((self.batch_size, 1))
        for i, p in enumerate(param):
            X[i] = get_lc(p)
            y[i] = 1
        # Find list of IDs
        self.list_IDs_temp = [self.list_IDs[k] for k in indexes]
        self.list_orders_temp = [self.list_orders[k] for k in indexes]
        meta = pd.read_csv(self.conv_const_path)
        self.conv_const = meta.loc[meta['ID'].isin(self.list_IDs_temp)]['const'].values
        self.true_params = meta.loc[meta['ID'].isin(self.list_IDs_temp)][['k', 'g', 's']].values
        if self.n_channels == 4:
            self.k = meta.loc[meta['ID'].isin(self.list_IDs_temp)]['k'].values
            self.g = meta.loc[meta['ID'].isin(self.list_IDs_temp)]['g'].values
            self.s = meta.loc[meta['ID'].isin(self.list_IDs_temp)]['s'].values

        # Generate data
        if self.output_format == 'xx' or \
                self.output_format == 'x1x2Y' or \
                self.output_format == 'xxy':
            X, y = self.__data_generation()
            return X, y
        else:
            X = self.__data_generation()
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_input[0], self.dim_input[1], self.n_channels))
        Y = np.empty((self.batch_size, self.dim[0], self.dim[1], 1))
        X2 = np.empty((self.batch_size, 3))

        # Generate data
        for i, ID in enumerate(self.list_IDs_temp):
            # Store sample
            f = open(self.path + str(ID) + "/map.bin", "rb")
            map_tmp = np.fromfile(f, 'i', -1, "")
            maps = (np.reshape(map_tmp, (-1, 10000)))
            order = self.list_orders_temp[i]
            tmp = maps[0::self.res_scale, 0::self.res_scale]
            if self.crop_scale != 1:
                tmp = self.percent_selector(tmp, order)

            if self.n_channels == 4:
                tmp = tmp.reshape((self.dim[0], self.dim[1]))
                tmp = np.log10(tmp * self.conv_const[i] + 0.004)
                X[i, :, :, 0] = NormalizeData(tmp)
                X[i, :, :, 1] = self.k[i] / 2.
                X[i, :, :, 2] = self.g[i] / 2.
                X[i, :, :, 3] = self.s[i]
                Y[i, :, :, 0] = NormalizeData(tmp)
            if self.n_channels == 2:
                tmp = tmp.reshape((self.dim[0], self.dim[1]))
                tmp = np.log10(tmp * self.conv_const[i] + 0.004)
                if self.include_lens_pos:
                    X[i, :, :, 0] = NormalizeData(tmp)
                    lens_pos = np.loadtxt(self.path + str(ID) + "/lens_pos.dat")
                    X[i, :, :, 1] = self.lens_pos_map(lens_pos)
                else:
                    X[i, :, :, :] = NormalizeData(tmp.reshape((self.dim[0], self.dim[1], 2)))

                Y[i, :, :, 0] = NormalizeData(tmp)
            else:
                # tmp = tmp.reshape((self.dim[0], self.dim[1], 1))
                tmp = np.log10(tmp * self.conv_const[i] + 0.004)
                X[i, :, :, 0] = NormalizeData(tmp)
                X2[i,] = self.true_params[i]
                Y[i, :, :, 0] = NormalizeData(tmp)

        if self.output_format == 'xx':
            # This is the default for most cases. X can have many channels, but Y will be always the normalized map.
            return X, Y
        if self.output_format == 'xxy':
            # This is the default for passing two channels separately.
            # X has two channels, but they will be considered two inputs.
            return [X[:, :, :, 0].reshape((X.shape[0], X.shape[1], X.shape[2], 1)),
                    X[:, :, :, 1].reshape((X.shape[0], X.shape[1], X.shape[2], 1))], \
                   Y
        elif self.output_format == 'x':
            # This is for the VAE that only takes data and assumes it should return the same input.
            return X
        elif self.output_format == 'x1x2':
            # This is for VAE but when we give the three params k,g,s along with the map as input.
            return [X, X2]
        elif self.output_format == 'x1x2Y':
            # This is the default format but when we give the three params k,g,s along with the map as input.
            return [X, X2], Y

    def get_generator_indexes(self):
        return self.indexes

    def map_reader(self, list_IDs_temp_, output='map'):
        '''
        This function is written to help see if the pipeline for reading the maps is working fine.
        For a given set of IDs, It reads the map, and takes log10, and normalized them to be between -3 and 6.
        It also does the convolution with source profile and returns that if the option is chosen.

        :param list_IDs_temp: A list of IDs of maps that you want to read
        :param output: What the output should be. The options are:
                "map": If you want the maps in units of magnification
                "map_norm": If you want the maps normalized.
        :return:
        '''
        list_order_temp = list_IDs_temp_[:, 1]
        list_IDs_temp = list_IDs_temp_[:, 0]
        X = np.empty((len(list_IDs_temp), self.dim[0], self.dim[1], 1))
        meta = pd.read_csv(self.conv_const_path)
        conv_const = meta.loc[meta['ID'].isin(list_IDs_temp)]['const'].values

        # macro_mag = meta.loc[meta['ID'].isin(list_IDs_temp)]['const'].values

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # try:
            f = open(self.path + str(ID) + "/map.bin", "rb")
            map_tmp = np.fromfile(f, 'i', -1, "")
            maps = (np.reshape(map_tmp, (-1, 10000)))
            tmp = maps[0::self.res_scale, 0::self.res_scale]
            tmp = tmp * conv_const[i]
            order = list_order_temp[i]
            if self.crop_scale != 1:
                tmp = self.percent_selector(tmp, order)

            if output == 'map':
                if self.n_channels == 4:
                    X[i, :, :, 0] = tmp
                    X[i, :, :, 1] = self.k[i] / 2.
                    X[i, :, :, 2] = self.g[i] / 2.
                    X[i, :, :, 3] = self.s[i]
                if self.n_channels == 2:
                    X[i, :, :, :] = tmp.reshape((self.dim[0], self.dim[1], 1))
                else:
                    X[i, :, :, :] = NormalizeData(tmp).reshape((self.dim[0], self.dim[1], 1))

            elif output == 'lamda':

                if self.n_channels == 4:
                    tmp = np.log10(tmp + 0.004)
                    X[i, :, :, 0] = NormalizeData(tmp)
                    X[i, :, :, 1] = self.k[i] / 2.
                    X[i, :, :, 2] = self.g[i] / 2.
                    X[i, :, :, 3] = self.s[i]
                if self.n_channels == 2:
                    tmp = tmp.reshape((self.dim[0], self.dim[1], 1))
                    tmp = np.log10(tmp + 0.004)
                    X[i, :, :, :] = NormalizeData(tmp)
                else:
                    tmp = np.log10(tmp + 0.004)
                    X[i, :, :, :] = NormalizeData(tmp).reshape((self.dim[0], self.dim[1], 1))

        return X

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


class DataGenerator2(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, list_IDs, dict, dict2=None, batch_size=8, dim=10000, n_channels=1,
                 res_scale=10, crop_scale=1, path='./../../../fred/oz108/GERLUMPH_project/DATABASES/gerlumph_db/',
                 shuffle=True, conv_const='./../data/all_maps_meta_kgs.csv',
                 output_format='xx',
                 include_lens_pos=False,
                 include_map_units=False):
        """Initialization"""

        self.dim = (int((dim / res_scale) * crop_scale), int((dim / res_scale) * crop_scale))
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.data = dict
        self.list_orders = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.res_scale = res_scale
        self.crop_scale = crop_scale
        self.path = path
        self.indexes = np.arange(len(self.list_IDs), dtype=int)
        self.on_epoch_end()
        self.conv_const_path = conv_const
        self.output_format = output_format
        self.include_lens_pos = include_lens_pos
        self.include_map_units = include_map_units
        self.lens_pos_units = np.linspace(-500, 500, 1001)
        if self.include_map_units:
            self.dim_input = self.dim + 1
        else:
            self.dim_input = self.dim

        if self.include_lens_pos:
            self.data_extra = dict2
        #     self.n_channels = 2
        # elif self.include_lens_pos and self.crop_scale != 1:
        #     print('Lens positions cannot be included for cropped maps ')
        #     exit()
        # elif self.include_map_units and self.crop_scale != 1:
        #     print('Map units cannot be included for cropped maps ')
        #     exit()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        self.list_IDs_temp = [self.list_IDs[k] for k in indexes]
        self.list_orders_temp = [self.list_orders[k] for k in indexes]
        self.data_tmp = [self.data[ID] for ID in self.list_IDs_temp]
        if self.include_lens_pos:
            self.data_extra_tmp = [self.data_extra[ID] for ID in self.list_IDs_temp]
        meta = pd.read_csv(self.conv_const_path)
        self.conv_const = meta.loc[meta['ID'].isin(self.list_IDs_temp)]['const'].values
        self.true_params = meta.loc[meta['ID'].isin(self.list_IDs_temp)][['k', 'g', 's']].values
        if self.n_channels == 4:
            self.k = meta.loc[meta['ID'].isin(self.list_IDs_temp)]['k'].values
            self.g = meta.loc[meta['ID'].isin(self.list_IDs_temp)]['g'].values
            self.s = meta.loc[meta['ID'].isin(self.list_IDs_temp)]['s'].values

        # Generate data
        if self.output_format == 'xx' or \
                self.output_format == 'x1x2Y' or \
                self.output_format == 'xxy':
            X, y = self.__data_generation()
            return X, y
        else:
            X = self.__data_generation()
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim_input[0], self.dim_input[1], self.n_channels))
        Y = np.empty((self.batch_size, self.dim[0], self.dim[1], 1))
        X2 = np.empty((self.batch_size, 3))
        if self.include_lens_pos:
            X3 = np.empty((self.batch_size, 50, 50, 1))

        # Generate data
        for i, ID in enumerate(self.list_IDs_temp):
            # Store sample
            tmp = self.data_tmp[i].reshape((self.dim[0], self.dim[1]))
            order = self.list_orders_temp[i]
            if self.crop_scale != 1:
                tmp = self.percent_selector(tmp, order)

            if self.n_channels == 4:
                tmp = tmp.reshape((self.dim[0], self.dim[1]))
                tmp = np.log10(tmp * self.conv_const[i] + 0.004)
                X[i, :, :, 0] = NormalizeData(tmp)
                X[i, :, :, 1] = self.k[i] / 2.
                X[i, :, :, 2] = self.g[i] / 2.
                X[i, :, :, 3] = self.s[i]
                Y[i, :, :, 0] = NormalizeData(tmp)
            if self.n_channels == 2:
                tmp = tmp.reshape((self.dim[0], self.dim[1]))
                tmp = np.log10(tmp * self.conv_const[i] + 0.004)
                if self.include_lens_pos:
                    X[i, :, :, 0] = NormalizeData(tmp)
                    lens_pos = np.loadtxt(self.path + str(ID) + "/lens_pos.dat")
                    X[i, :, :, 1] = self.lens_pos_map(lens_pos)
                else:
                    X[i, :, :, :] = NormalizeData(tmp.reshape((self.dim[0], self.dim[1], 2)))

                Y[i, :, :, 0] = NormalizeData(tmp)
            else:
                # tmp = tmp.reshape((self.dim[0], self.dim[1], 1))
                tmp = np.log10(tmp * self.conv_const[i] + 0.004)
                X[i, :, :, 0] = NormalizeData(tmp)
                X2[i,] = self.true_params[i]
                Y[i, :, :, 0] = NormalizeData(tmp)
                if self.include_lens_pos:
                    data_tmp = self.data_extra_tmp[i]
                    if data_tmp.shape[0] == 0:
                        pass
                    elif data_tmp.shape[0] != 0 and len(data_tmp.shape) == 1:
                        X3[i, :, :, 0] = np.histogram2d(np.asarray([data_tmp])[:, 0],
                                                        np.asarray([data_tmp])[:, 1],
                                                        bins=50, density=True)[0]
                    else:
                        X3[i, :, :, 0] = np.histogram2d(data_tmp[:, 0],
                                                        data_tmp[:, 1],
                                                        bins=50, density=True)[0]

        if self.output_format == 'xx':
            # This is the default for most cases. X can have many channels, but Y will be always the normalized map.
            return X, Y
        if self.output_format == 'xxy':
            # This is the default for passing two channels separately.
            # X has two channels, but they will be considered two inputs.
            return [X[:, :, :, 0].reshape((X.shape[0], X.shape[1], X.shape[2], 1)),
                    X[:, :, :, 1].reshape((X.shape[0], X.shape[1], X.shape[2], 1))], \
                   Y
        elif self.output_format == 'x':
            # This is for the VAE that only takes data and assumes it should return the same input.
            return X
        elif self.output_format == 'x1x2':
            # This is for VAE but when we give the three params k,g,s along with the map as input.
            return [X, X2]
        elif self.output_format == 'x1x2Y':
            if self.include_lens_pos:
                return [X, X3], Y
            # This is the default format but when we give the three params k,g,s along with the map as input.
            else:
                return [X, X2], Y

    def get_generator_indexes(self):
        return self.indexes

    def map_reader(self, list_IDs_temp_, output='map'):
        '''
        This function is written to help see if the pipeline for reading the maps is working fine.
        For a given set of IDs, It reads the map, and takes log10, and normalized them to be between -3 and 6.
        It also does the convolution with source profile and returns that if the option is chosen.

        :param list_IDs_temp: A list of IDs of maps that you want to read
        :param output: What the output should be. The options are:
                "map": If you want the maps in units of magnification
                "map_norm": If you want the maps normalized.
        :return:
        '''
        list_order_temp = list_IDs_temp_[:, 1]
        list_IDs_temp = list_IDs_temp_[:, 0]
        X = np.empty((len(list_IDs_temp), self.dim[0], self.dim[1], 1))
        meta = pd.read_csv(self.conv_const_path)
        conv_const = meta.loc[meta['ID'].isin(list_IDs_temp)]['const'].values

        # macro_mag = meta.loc[meta['ID'].isin(list_IDs_temp)]['const'].values

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # try:
            f = open(self.path + str(ID) + "/map.bin", "rb")
            map_tmp = np.fromfile(f, 'i', -1, "")
            maps = (np.reshape(map_tmp, (-1, 10000)))
            tmp = maps[0::self.res_scale, 0::self.res_scale]
            tmp = tmp * conv_const[i]
            order = list_order_temp[i]
            if self.crop_scale != 1:
                tmp = self.percent_selector(tmp, order)

            if output == 'map':
                if self.n_channels == 4:
                    X[i, :, :, 0] = tmp
                    X[i, :, :, 1] = self.k[i] / 2.
                    X[i, :, :, 2] = self.g[i] / 2.
                    X[i, :, :, 3] = self.s[i]
                if self.n_channels == 2:
                    X[i, :, :, :] = tmp.reshape((self.dim[0], self.dim[1], 1))
                else:
                    X[i, :, :, :] = NormalizeData(tmp).reshape((self.dim[0], self.dim[1], 1))

            elif output == 'map_norm':

                if self.n_channels == 4:
                    tmp = np.log10(tmp + 0.004)
                    X[i, :, :, 0] = NormalizeData(tmp)
                    X[i, :, :, 1] = self.k[i] / 2.
                    X[i, :, :, 2] = self.g[i] / 2.
                    X[i, :, :, 3] = self.s[i]
                if self.n_channels == 2:
                    tmp = tmp.reshape((self.dim[0], self.dim[1], 1))
                    tmp = np.log10(tmp + 0.004)
                    X[i, :, :, :] = NormalizeData(tmp)
                else:
                    tmp = np.log10(tmp + 0.004)
                    X[i, :, :, :] = NormalizeData(tmp).reshape((self.dim[0], self.dim[1], 1))

        return X

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


def data_gererators_set_up_AD(running_params, model_params):
    partition, data_dict, running_params, model_params = prepare_input_to_fit_keras_ADs(running_params,
                                                                                            model_params)
    selected_params = {'dim': running_params['dim'],
                       'batch_size': running_params['batch_size'],
                       'n_channels': model_params['n_channels'],
                       'res_scale': running_params['res_scale'],
                       'crop_scale': model_params['crop_scale'],
                       'path': running_params['path'],
                       'shuffle': running_params['shuffle'],
                       'conv_const': running_params['conv_const'],
                       'output_format': model_params['output_format'],
                       'include_lens_pos': model_params['include_lens_pos'],
                       'include_map_units': model_params['include_map_units']}
    if data_dict is None:
        if model_params['include_lens_pos']:
            print('Adding lens position requires setting train_set_selection=random.')
            return
        else:
            training_generator = DataGenerator(partition['train'], data_dict, **selected_params)
            validation_generator = DataGenerator(partition['validation'], data_dict, **selected_params)
            selected_params['shuffle'] = False
            test_generator = DataGenerator(partition['test'], data_dict, **selected_params)
    else:
        if model_params['include_lens_pos']:
            all_lens_pos = read_all_lens_pos()
            training_generator = DataGenerator2(partition['train'], data_dict, dict2=all_lens_pos, **selected_params)
            validation_generator = DataGenerator2(partition['validation'], data_dict, dict2=all_lens_pos,
                                                  **selected_params)
            selected_params['shuffle'] = False
            test_generator = DataGenerator2(partition['test'], data_dict, dict2=all_lens_pos, **selected_params)
        else:
            training_generator = DataGenerator2(partition['train'], data_dict, **selected_params)
            validation_generator = DataGenerator2(partition['validation'], data_dict, **selected_params)
            selected_params['shuffle'] = False
            test_generator = DataGenerator2(partition['test'], data_dict, **selected_params)

    return training_generator, validation_generator, test_generator, running_params, model_params, partition


def data_generator_set_up_kgs_bt(model_design_key, running_params):
    bt_train, bt_test, kgs_train, kgs_test, lens_pos_train, lens_pos_test, ids_train, ids_test, IDs = prepare_input_for_kgs_bt(model_design_key, running_params)

    if model_design_key == 'kgs_lens_pos_to_bt':
        train_object_X = [kgs_train, lens_pos_train]
        train_object_Y = bt_train
        test_object_X = [kgs_test, lens_pos_test]
        test_object_Y = bt_test

    elif model_design_key == 'bt_to_kgs':
        train_object_X = bt_train
        train_object_Y = kgs_train
        test_object_X = bt_test
        test_object_Y = kgs_test

    elif model_design_key == 'kgs_to_bt':
        train_object_X = kgs_train
        train_object_Y = bt_train
        test_object_X = kgs_test
        test_object_Y = bt_test

    return train_object_X, train_object_Y, test_object_X, test_object_Y, ids_train, ids_test
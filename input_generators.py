"""Code from:

https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

"""

import keras
import numpy as np
import pandas as pd
from maps_util import NormalizeData, read_all_lens_pos, read_binary_map, norm_maps, process_read_lens_pos
from training_utils import prepare_input_to_fit_keras_ADs, prepare_input_for_kgs_bt
from concurrent.futures import ThreadPoolExecutor
import tensorflow as tf
import os
from functools import lru_cache


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, list_IDs, batch_size=8, input_size=10000, n_channels=1,
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

    import numpy as np
    import tensorflow as tf
    import os
    from concurrent.futures import ThreadPoolExecutor

    class CustomDataGenerator(tf.keras.utils.Sequence):
        def __init__(self, data_dir, batch_size=32, shuffle=True, num_workers=4):
            """
            Custom Data Generator with parallel file loading.

            Args:
                data_dir (str): Path to the dataset directory.
                batch_size (int): Number of samples per batch.
                shuffle (bool): Whether to shuffle data at the end of each epoch.
                num_workers (int): Number of threads for parallel loading.
            """
            self.data_dir = data_dir
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.num_workers = num_workers
            self.object_folders = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if
                                   os.path.isdir(os.path.join(data_dir, f))]
            self.file_paths = self._get_file_list()
            self.on_epoch_end()  # Shuffle at initialization if needed

        def _get_file_list(self):
            """Collects all file paths from object directories."""
            file_paths = []
            for folder in self.object_folders:
                files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.npy')]
                file_paths.extend(files)
            return file_paths

        def __len__(self):
            """Denotes the number of batches per epoch."""
            return int(np.floor(len(self.file_paths) / self.batch_size))

        def __getitem__(self, index):
            """Generates one batch of data."""
            batch_files = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]
            X, y = self.__data_generation(batch_files)
            return X, y

        def __data_generation(self, batch_files):
            """Loads data in parallel using multiple threads."""
            X = []
            y = []

            # Use ThreadPoolExecutor for parallel loading
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(self._load_file, batch_files))

            for data, label in results:
                X.append(data)
                y.append(label)

            return np.array(X), np.array(y)

        def _load_file(self, file_path):
            """Loads a single file and extracts its label."""
            data = np.load(file_path)  # Load .npy file
            label = self._extract_label_from_path(file_path)
            return data, label

        def _extract_label_from_path(self, file_path):
            """Derives label from folder name or filename if necessary."""
            object_name = os.path.basename(os.path.dirname(file_path))  # Extract object folder name
            label = int(object_name.split('_')[-1])  # Example: extracting number from 'object_3'
            return label

        def on_epoch_end(self):
            """Shuffles data at the end of each epoch."""
            if self.shuffle:
                np.random.shuffle(self.file_paths)

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

    def __init__(self, list_IDs, dict, dict2=None, batch_size=8, input_size=10000, n_channels=1,
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
    selected_params = {'input_size': running_params['input_size'],
                       'output_size': running_params['output_size'],
                       'batch_size': running_params['batch_size'],
                       'n_channels': model_params['n_channels'],
                       'res_scale': running_params['res_scale'],
                       'crop_scale': model_params['crop_scale'],
                       'path': running_params['path'],
                       'shuffle': running_params['shuffle'],
                       'conv_const': running_params['conv_const'],
                       'input_output_format': model_params['input_output_format'],
                       'include_lens_pos': model_params['include_lens_pos'],
                       'include_map_units': model_params['include_map_units']}
    if data_dict is None:
        if model_params['include_lens_pos']:
            print('Adding lens position requires setting train_set_selection=random.')
            training_generator = OptimizedDataGeneratorByID(partition['train'], **selected_params)
            validation_generator = OptimizedDataGeneratorByID(partition['validation'], **selected_params)
            selected_params['shuffle'] = False
            test_generator = OptimizedDataGeneratorByID(partition['test'], **selected_params)
        else:
            training_generator = OptimizedDataGeneratorByID(partition['train'], **selected_params)
            validation_generator = OptimizedDataGeneratorByID(partition['validation'], **selected_params)
            selected_params['shuffle'] = False
            test_generator = OptimizedDataGeneratorByID(partition['test'], **selected_params)
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
    bt_train, bt_test, kgs_train, kgs_test, lens_pos_train, lens_pos_test, ids_train, ids_test, IDs = prepare_input_for_kgs_bt(
        model_design_key, running_params)

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


class OptimizedDataGeneratorByID(tf.keras.utils.Sequence):
    def __init__(self,
                 id_list,
                 path='./../../../fred/oz108/GERLUMPH_project/DATABASES/gerlumph_db/',
                 input_size=10000,
                 output_size=1000,
                 batch_size=8,
                 shuffle=True,
                 num_workers=4,
                 cache_size=100,
                 n_channels=1,
                 res_scale=10,
                 crop_scale=1,
                 conv_const='./../data/all_maps_meta_kgs.csv',
                 input_output_format='xx',
                 include_lens_pos=False,
                 include_map_units=False
                 ):
        """
        Optimized Data Generator for datasets with folders named by IDs.

        Args:
            data_dir (str): Path to the dataset directory.
            id_list (list): List of IDs corresponding to subfolders.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle data at the end of each epoch.
            num_workers (int): Number of threads for parallel loading.
            cache_size (int): Number of recently accessed files to cache in memory.
        """
        self.data_dir = path
        self.input_size = input_size
        self.output_size = output_size
        self.id_list = id_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.cache_size = cache_size
        self.file_paths = self._get_file_list()
        self.include_lens_pos = include_lens_pos
        self.input_output_format = input_output_format
        self.on_epoch_end()  # Shuffle at initialization if needed
        self.lens_pos_max = 244784

    def _get_file_list(self):
        """Collects all file paths from ID directories."""
        file_paths = []
        for object_id in self.id_list:
            folder = os.path.join(self.data_dir, str(object_id))
            if os.path.isdir(folder):
                files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.bin')]
                file_paths.extend(files)
        return file_paths

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.file_paths) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data."""
        batch_files = self.file_paths[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(batch_files)
        return X, y

    def __data_generation(self, batch_files):
        """Loads data in parallel using multiple threads and caching."""
        X1 = np.zeros((self.batch_size, self.output_size, self.output_size, 1), dtype=np.float32)
        X2 = np.zeros((self.batch_size, self.lens_pos_max, 2), dtype=np.float32)

        # Use ThreadPoolExecutor for parallel loading
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self._load_file_with_cache, batch_files))

        for i, (data, label) in enumerate(results):
            if self.include_lens_pos:
                X1[i, :, :, 0] = data[0]
                X2[i, :, :] = data[1]
            else:
                X1[i, :, :, 0] = data

        if self.include_lens_pos:
            return (X1, X2), X1
        else:
            return X1, X1

    @lru_cache(maxsize=100)
    def _load_file_with_cache(self, file_path):
        """Loads a single file and caches frequently accessed data."""
        label = self._extract_label_from_path(file_path)
        data = read_binary_map(label, self.input_size)
        if self.input_output_format == 'norm_maps':
            data = norm_maps(data)
        if self.include_lens_pos:
            lens_pos = process_read_lens_pos(label)
            return [data, lens_pos], label
        else:
            return data, label

    def _extract_label_from_path(self, file_path):
        """Derives label from the ID folder name."""
        object_id = os.path.basename(os.path.dirname(file_path))  # Extract object folder (ID) name
        label = int(object_id)  # Example: if ID is '3', label = 3. Adjust as needed.
        return label

    def on_epoch_end(self):
        """Shuffles data at the end of each epoch."""
        if self.shuffle:
            np.random.shuffle(self.file_paths)

    def to_tf_dataset(self):
        """Converts the generator into a TensorFlow dataset with two inputs and one output."""
        if self.include_lens_pos:
            dataset = tf.data.Dataset.from_generator(
                lambda: self,
                output_signature=(
                    (
                        tf.TensorSpec(shape=(self.batch_size, self.output_size, self.output_size, 1), dtype=tf.float32),  # X1 (image input)
                        tf.TensorSpec(shape=(self.batch_size, self.lens_pos_max, 2), dtype=tf.float32)  # X2 (array input)
                    ),
                    tf.TensorSpec(shape=(self.batch_size, self.output_size, self.output_size, 1), dtype=tf.float32)  # y (target)
                )
            )
        else:
            dataset = tf.data.Dataset.from_generator(
                lambda: self,
                output_signature=(
                        tf.TensorSpec(shape=(self.batch_size, self.output_size, self.output_size, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size, self.output_size, self.output_size, 1), dtype=tf.float32)
                )
            )
        return dataset.prefetch(tf.data.AUTOTUNE)

    def inspect_example_batch(self):
        """Fetches and prints an example batch to inspect the data."""
        # Get the first batch from the generator
        example_batch = self[0]

        # Unpack the batch into input (X) and output (y)
        X, y = example_batch

        # Print the shape of the input and output
        print("Shape of input data1 (X1):", X[0].shape)
        print("Shape of input data2 (X2):", X[1].shape)
        print("Shape of output data (y):", np.array(y).shape)

        # Show a sample input and output
        print("Example input (first sample):\n", np.mean(X[1]))
        # print("Example input (second sample):\n", np.mean(X[1][1]))
        # print("Example input (third sample):\n", np.mean(X[1][2]))
        # print("Example output (label for the first sample):", y[0])

        return X, y

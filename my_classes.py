"""Code from:

https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

"""
from tensorflow import keras
from keras import layers
from tensorflow.keras.layers import Flatten, Reshape
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Add, \
    BatchNormalization
import sys
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.saving import saving_api

from sf_fft import sf_fft

sys.path += ['..']
# path_to_module = '~/codes/normalizing_flows/normalizing_flows/'
# sys.path.append(path_to_module)
# from codes.normalizing_flows.normalizing_flows.layers import FlowLayer
# from codes.normalizing_flows.normalizing_flows.flows.affine import Planar, Radial
# from codes.normalizing_flows.normalizing_flows.flows import flow
# from codes.normalizing_flows.normalizing_flows.flows.sylvester import TriangularSylvester
# from codes.normalizing_flows.normalizing_flows.models.vae2 import GatedConvVAE


def tweedie_loss_func(p):
    def tweedie_loglikelihood(y, y_hat):
        loss = - y * tf.pow(y_hat, 1 - p) / (1 - p) + \
               tf.pow(y_hat, 2 - p) / (2 - p)
        return tf.reduce_mean(loss)

    return tweedie_loglikelihood


from scipy import interpolate


def lc_maker_mse(map, distance=15, n_samp=100):
    mapp, map_hat = map[0], map[1]

    batch_size = mapp.shape[0]

    nxpix = 1000
    nypix = 1000
    factor = 1
    tot_r_e = 25
    xr = [-(tot_r_e / factor), (tot_r_e / factor)]
    yr = [-(tot_r_e / factor), (tot_r_e / factor)]

    x0 = xr[0]
    dx = xr[1] - x0  # map units in R_E

    mappix = 1000 / dx
    nsamp = n_samp
    distance = distance  # in maps units
    dt = (22 * 365) / 1.5  # from Kochanek (2004) for Q2237+0305: The quasar takes 22 years to go through 1.5 R_Ein
    tmax = distance * dt

    t = np.linspace(0, tmax, nsamp, dtype='int')
    metric = []

    for n in range(batch_size):
        c = 0
        # if origin==None:
        mmap = mapp[n]
        mmap_hat = map_hat[n]
        origin = [i // 2 for i in mmap.shape]

        for ang in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
            # -------- determine the x and y pixel values
            rad = ang * np.pi / 180.0
            crad = np.cos(rad)
            srad = np.sin(rad)
            pixsz = dx / float(nxpix)  # size of one pixel in R_E
            drpix = distance / dx * float(nxpix) / float(nsamp)  # step's size in pixels
            ix = [i * drpix * crad + origin[0] for i in range(nsamp)]
            iy = [i * drpix * srad + origin[1] for i in range(nsamp)]

            # -------- interpolate onto light curve pixels
            #          (using interp2d!!!)
            x, y = np.arange(float(nxpix)), np.arange(float(nypix))
            mapint = interpolate.interp2d(x, y, mmap, kind='cubic')
            mapint2 = interpolate.interp2d(x, y, mmap_hat, kind='cubic')

            lc = np.array([mapint(i, j)[0] for i, j in zip(*[iy, ix])])
            lc2 = np.array([mapint2(i, j)[0] for i, j in zip(*[iy, ix])])

            c += np.sum((lc - lc2) ** 2)
        metric.append(c / 12.)
    return np.ndarray.astype(np.asarray(metric), np.float32)


@tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
def tf_lc_function_mse(input):
    y = tf.numpy_function(lc_maker_mse, [input], tf.float32)
    return y


def lc_maker_sf_median(map, distance=15, n_samp=100):
    mapp, map_hat = map[0], map[1]

    batch_size = mapp.shape[0]
    #
    # nxpix = 1000
    # nypix = 1000
    # factor = 1
    # tot_r_e = 25
    # xr = [-(tot_r_e / factor), (tot_r_e / factor)]
    # yr = [-(tot_r_e / factor), (tot_r_e / factor)]
    #
    # x0 = xr[0]
    # dx = xr[1] - x0  # map units in R_E
    #
    # mappix = 1000 / dx
    # nsamp = n_samp
    # distance = distance  # in maps units
    # dt = (22 * 365) / 1.5  # from Kochanek (2004) for Q2237+0305: The quasar takes 22 years to go through 1.5 R_Ein
    # tmax = distance * dt
    #
    # t = np.linspace(0, tmax, nsamp, dtype='int')
    metric = []

    for n in range(batch_size):
        c = 0
        # if origin==None:
        mmap = mapp[n]
        mmap_hat = map_hat[n]
        # origin = [i // 2 for i in mmap.shape]

        # for ang in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
        #     # -------- determine the x and y pixel values
        #     rad = ang * np.pi / 180.0
        #     crad = np.cos(rad)
        #     srad = np.sin(rad)
        #     pixsz = dx / float(nxpix)  # size of one pixel in R_E
        #     drpix = distance / dx * float(nxpix) / float(nsamp)  # step's size in pixels
        #     ix = [i * drpix * crad + origin[0] for i in range(nsamp)]
        #     iy = [i * drpix * srad + origin[1] for i in range(nsamp)]
        #
        #     # -------- interpolate onto light curve pixels
        #     #          (using interp2d!!!)
        #     x, y = np.arange(float(nxpix)), np.arange(float(nypix))
        #     mapint = interpolate.interp2d(x, y, mmap, kind='cubic')
        #     mapint2 = interpolate.interp2d(x, y, mmap_hat, kind='cubic')
        #
        #     lc = np.array([mapint(i, j)[0] for i, j in zip(*[iy, ix])])
        #     lc2 = np.array([mapint2(i, j)[0] for i, j in zip(*[iy, ix])])
        #
        #     c += np.median(np.abs(sf_fft(lc) - sf_fft(lc2)))
        c += np.median(np.abs(sf_fft(mmap) - sf_fft(mmap_hat)))

        metric.append(c / batch_size)
    return np.ndarray.astype(np.asarray(metric), np.float32)


@tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
def tf_lc_function_sf_median(input):
    y = tf.numpy_function(lc_maker_sf_median, [input], tf.float32)
    return y


def lc_maker_sf_max(map, distance=15, n_samp=100):
    mapp, map_hat = map[0], map[1]

    batch_size = mapp.shape[0]

    nxpix = 1000
    nypix = 1000
    factor = 1
    tot_r_e = 25
    xr = [-(tot_r_e / factor), (tot_r_e / factor)]
    yr = [-(tot_r_e / factor), (tot_r_e / factor)]

    x0 = xr[0]
    dx = xr[1] - x0  # map units in R_E

    mappix = 1000 / dx
    nsamp = n_samp
    distance = distance  # in maps units
    dt = (22 * 365) / 1.5  # from Kochanek (2004) for Q2237+0305: The quasar takes 22 years to go through 1.5 R_Ein
    tmax = distance * dt

    t = np.linspace(0, tmax, nsamp, dtype='int')
    metric = []

    for n in range(batch_size):
        c = 0
        # if origin==None:
        mmap = mapp[n]
        mmap_hat = map_hat[n]
        origin = [i // 2 for i in mmap.shape]

        for ang in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
            # -------- determine the x and y pixel values
            rad = ang * np.pi / 180.0
            crad = np.cos(rad)
            srad = np.sin(rad)
            pixsz = dx / float(nxpix)  # size of one pixel in R_E
            drpix = distance / dx * float(nxpix) / float(nsamp)  # step's size in pixels
            ix = [i * drpix * crad + origin[0] for i in range(nsamp)]
            iy = [i * drpix * srad + origin[1] for i in range(nsamp)]

            # -------- interpolate onto light curve pixels
            #          (using interp2d!!!)
            x, y = np.arange(float(nxpix)), np.arange(float(nypix))
            mapint = interpolate.interp2d(x, y, mmap, kind='cubic')
            mapint2 = interpolate.interp2d(x, y, mmap_hat, kind='cubic')

            lc = np.array([mapint(i, j)[0] for i, j in zip(*[iy, ix])])
            lc2 = np.array([mapint2(i, j)[0] for i, j in zip(*[iy, ix])])

            c += np.max(np.abs(sf_fft(lc) - sf_fft(lc2)))
            c += np.max(np.abs(sf_fft(mmap) - sf_fft(mmap_hat)))

        metric.append(c / 12.)
    return np.ndarray.astype(np.asarray(metric), np.float32)


@tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
def tf_lc_function_sf_max(input):
    y = tf.numpy_function(lc_maker_sf_max, [input], tf.float32)
    return y


def custom_loss_func(up=99, low=10):
    def custom_loglikelihood(y, y_hat):
        # batch_size = tf.shape(y)[0]
        # up_lim = tfp.stats.percentile(y, up, axis=[1, 2, 3])
        # low_lim = tfp.stats.percentile(y, low, axis=[1, 2, 3])
        # huber = tf.cast(0, tf.float32)
        # huber1 = tf.cast(0, tf.float32)

        # for i in range(batch_size):
        #   huber += tf.keras.losses.MeanSquaredError()(y[i][tf.math.greater(y[i], up_lim[i])],
        #                                       y_hat[i][tf.math.greater(y[i], up_lim[i])])
        #   huber1 += tf.keras.losses.MeanSquaredError()(y[tf.math.less(y[i], low_lim[i])],
        #                                       y_hat[tf.math.less(y[i], low_lim[i])])

        # huber = tf.keras.losses.MeanSquaredError()(
        #     y[tf.math.logical_and(tf.math.greater(y, up_lim), tf.math.less(y, low_lim))],
        #     y_hat[tf.math.logical_and(tf.math.greater(y, up_lim), tf.math.less(y, low_lim))]
        # )

        # huber = tf.keras.losses.BinaryCrossentropy()(y[tf.math.greater(y, 0.6)],
        #                                         y_hat[tf.math.greater(y, 0.6)])
        # if tf.math.is_nan(huber):
        #     huber = tf.cast(0, tf.float32)
        # # huber1 = tf.keras.losses.BinaryCrossentropy()(y[tf.math.less(y, 0.35)],
        # #                                         y_hat[tf.math.less(y, 0.35)])
        # if tf.math.is_nan(huber1):
        #     huber = tf.cast(0, tf.float32)
        kl_loss = tf.keras.losses.kl_divergence(y, y_hat)
        huber2 = tf.keras.losses.BinaryCrossentropy()(y, y_hat)
        # coeff = tf.Variable(1000)
        return tf.reduce_mean(huber2 + tf.math.abs(kl_loss))  # +\
        # tf.cast(huber/tf.cast(batch_size, tf.float32), tf.float32)+\
        # tf.cast(huber1/tf.cast(batch_size, tf.float32), tf.float32)
        # huber2 + tf.cast(2, tf.float32)*huber1)

    return custom_loglikelihood


def lc_loss_func(metric='mse', lc_loss_coeff=1):
    def lc_loglikelihood(y, y_hat):
        batch_size = tf.shape(y)[0]
        data = tf.concat([tf.reshape(y, [1, batch_size, tf.shape(y)[1], tf.shape(y)[1]]),
                          tf.reshape(y_hat, [1, batch_size, tf.shape(y)[1], tf.shape(y)[1]])],
                         0)
        if metric == 'mse':
            c = K.sum(tf_lc_function_mse(data))
        elif metric == 'sf_median':
            c = K.sum(tf_lc_function_sf_median(data))
        elif metric == 'sf_max':
            c = K.sum(tf_lc_function_sf_max(data))
        tf.cast(y_hat, tf.float32)
        tf.cast(y, tf.float32)
        tf.cast(c, tf.float32)
        lc_loss_coeff_tf = tf.constant(lc_loss_coeff, dtype=tf.float32)
        lc_loss = lc_loss_coeff_tf * c

        return lc_loss + tf.reduce_mean(tf.keras.losses.BinaryCrossentropy(from_logits=True)(y, y_hat))

    return lc_loglikelihood


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


def NormalizeData(data, data_max=6, data_min=-3):
    data_new = (data - data_min) / (data_max - data_min)
    return data_new



def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2])
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1])
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)

    return (ch1, ch2), (cw1, cw2)


# Design model
def model_design_2l(input_side):
    input_img = keras.Input(shape=(input_side, input_side, 1))

    x = layers.Conv2D(128, (5, 5), activation='relu',
                      strides=(4, 4), padding="same")(input_img)
    x = layers.Conv2D(128, (6, 6), activation='relu')(x)
    x = layers.MaxPooling2D((5, 5))(x)
    x = layers.Conv2D(1, (5, 5), activation='relu')(x)
    encoded = layers.MaxPooling2D((5, 5))(x)

    x = layers.UpSampling2D((5, 5))(encoded)
    x = layers.Conv2DTranspose(128, (5, 5), activation='relu')(x)
    x = layers.UpSampling2D((5, 5))(x)
    x = layers.Conv2DTranspose(128, (6, 6), activation='relu')(x)
    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid',
                                     strides=(4, 4), padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def model_design_3l(input_side):
    input_img = keras.Input(shape=(input_side, input_side, 1))

    x = layers.Conv2D(32, (4, 4), activation='relu',
                      strides=1, padding="same")(input_img)
    x = layers.MaxPooling2D((5, 5))(x)
    x = layers.Conv2D(128, (4, 4), activation='relu',
                      strides=1, padding="same")(x)
    x = layers.MaxPooling2D((5, 5))(x)
    x = layers.Conv2D(256, (4, 4), activation='relu',
                      strides=1, padding="same")(x)
    x = layers.MaxPooling2D((5, 5))(x)
    x = layers.Conv2D(1, (4, 4), activation='relu',
                      strides=1, padding="same")(x)
    encoded = x  # layers.MaxPooling2D((5, 5))(x)

    # x = layers.UpSampling2D((5, 5))(encoded)
    x = layers.Conv2DTranspose(256, (4, 4), activation='relu',
                               strides=1, padding="same")(encoded)
    x = layers.UpSampling2D((5, 5))(x)
    x = layers.Conv2DTranspose(128, (4, 4), activation='relu',
                               strides=1, padding="same")(x)
    x = layers.UpSampling2D((5, 5))(x)
    x = layers.Conv2DTranspose(32, (4, 4), activation='relu',
                               strides=1, padding="same")(x)
    x = layers.UpSampling2D((5, 5))(x)
    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid',
                                     padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def model_design_Unet(input_side, n_channels=1, first_down_sampling=2):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(input_img)
    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((first_down_sampling, first_down_sampling))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((5, 5))(x)

    x = layers.Conv2D(1, (4, 4), activation='relu', strides=1, padding="same")(x)

    encoded = x

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(encoded)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((5, 5))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((first_down_sampling, first_down_sampling))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def Unet_lens_pos(input_side1, input_side2, n_channels=1, af='relu'):
    input_img1 = keras.Input(shape=(input_side1, input_side1, n_channels))
    input_img2 = keras.Input(shape=(input_side2, input_side2, n_channels))

    x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img1)
    # x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((5, 5))(x)

    x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    y = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(input_img2)
    x = layers.Add()([x, y])

    # x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    # # x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.MaxPooling2D((5, 5))(x)

    x = layers.Conv2D(1, (4, 4), activation=af, strides=1, padding="same")(x)
    encoded = x

    # x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x6)
    # x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    # # x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.UpSampling2D((5, 5))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((5, 5))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model([input_img1, input_img2], decoded)

    return autoencoder_model


def Unet_take_two_channels_separately(input_side, n_channels=1, first_down_sampling=2):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))
    input_img2 = keras.Input(shape=(input_side, input_side, n_channels))

    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(input_img)
    x2 = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(input_img2)
    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x2 = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x2)
    x3 = layers.Add()([x, x2])
    x = layers.MaxPooling2D((2, 2))(x3)

    x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((5, 5))(x)

    x = layers.Conv2D(1, (4, 4), activation='relu', strides=1, padding="same")(x)

    encoded = x

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(encoded)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((5, 5))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((first_down_sampling, first_down_sampling))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def Unet_sobel_edges1(input_side, n_channels=1):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    grad_components = tf.image.sobel_edges(input_img)
    grad_mag_components = grad_components ** 2
    grad_mag_square = tf.math.reduce_sum(grad_mag_components, axis=-1)  # sum all magnitude components
    grad_mag_img = tf.sqrt(grad_mag_square)
    # grad_mag_img = BatchNormalization()(grad_mag_img)
    mask = tf.greater(grad_mag_img, 0.0001 * tf.ones_like(grad_mag_img))
    grad_mag_img = tf.multiply(grad_mag_img, tf.cast(mask, tf.float32))

    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(input_img)
    x2 = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(grad_mag_img)
    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = Add()([x, x2])
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((5, 5))(x)

    x = layers.Conv2D(1, (4, 4), activation='relu', strides=1, padding="same")(x)

    encoded = x

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(encoded)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((5, 5))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def Unet_sobel_edges2(input_side, n_channels=1):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    grad_components = tf.image.sobel_edges(input_img)
    grad_mag_components = grad_components ** 2
    grad_mag_square = tf.math.reduce_sum(grad_mag_components, axis=-1)  # sum all magnitude components
    grad_mag_img = tf.sqrt(grad_mag_square)
    # grad_mag_img = BatchNormalization()(grad_mag_img)
    mask = tf.greater(grad_mag_img, 0.0001 * tf.ones_like(grad_mag_img))
    grad_mag_img = tf.multiply(grad_mag_img, tf.cast(mask, tf.float32))

    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(input_img)
    x2 = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(grad_mag_img)
    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = Add()([x, x2])
    x = layers.MaxPooling2D((2, 2))(x)
    x2 = layers.MaxPooling2D((2, 2))(x2)

    x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x3 = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x2)
    x = Add()([x, x3])
    x = layers.MaxPooling2D((2, 2))(x)
    x3 = layers.MaxPooling2D((2, 2))(x3)

    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x4 = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x3)
    x = Add()([x, x4])
    x = layers.MaxPooling2D((2, 2))(x)
    x4 = layers.MaxPooling2D((2, 2))(x4)

    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x5 = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x4)
    x = Add()([x, x5])
    x = layers.MaxPooling2D((5, 5))(x)
    x5 = layers.MaxPooling2D((2, 2))(x5)

    x = layers.Conv2D(1, (4, 4), activation='relu', strides=1, padding="same")(x)

    encoded = x

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(encoded)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((5, 5))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def model_design_Unet_resnet(input_side, n_channels=1, af='relu'):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    bl1_conv1 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img)
    bl1_conv2 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(bl1_conv1)

    bl1_conv3 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img)
    bl1_add = Add()([bl1_conv3, bl1_conv2])
    bl1_pl = layers.MaxPooling2D((2, 2))(bl1_add)

    bl2_conv1 = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(bl1_pl)
    bl2_conv2 = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(bl2_conv1)

    bl2_conv3 = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(bl1_pl)
    bl2_add = Add()([bl2_conv3, bl2_conv2])
    bl2_pl = layers.MaxPooling2D((2, 2))(bl2_add)

    bl3_conv1 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl2_pl)
    bl3_conv2 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl3_conv1)

    bl3_conv3 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl2_pl)
    bl3_add = Add()([bl3_conv3, bl3_conv2])
    bl3_pl = layers.MaxPooling2D((2, 2))(bl3_add)

    bl4_conv1 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl3_pl)
    bl4_conv2 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl4_conv1)

    bl4_conv3 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl3_pl)
    bl4_add = Add()([bl4_conv3, bl4_conv2])
    bl4_pl = layers.MaxPooling2D((5, 5))(bl4_add)

    encoded = layers.Conv2D(1, (4, 4), activation=af, strides=1, padding="same")(bl4_pl)

    bl5_conv1 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(encoded)
    bl5_conv2 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl5_conv1)

    bl5_conv3 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(encoded)
    bl5_add = Add()([bl5_conv3, bl5_conv2])
    bl5_pl = layers.UpSampling2D((5, 5))(bl5_add)

    bl6_conv1 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl5_pl)
    bl6_conv2 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl6_conv1)

    bl6_conv3 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl5_pl)
    bl6_add = Add()([bl6_conv3, bl6_conv2])
    bl6_pl = layers.UpSampling2D((2, 2))(bl6_add)

    bl7_conv1 = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(bl6_pl)
    bl7_conv2 = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(bl7_conv1)

    bl7_conv3 = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(bl6_pl)
    bl7_add = Add()([bl7_conv3, bl7_conv2])
    bl7_pl = layers.UpSampling2D((2, 2))(bl7_add)

    bl8_conv1 = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(bl7_pl)
    bl8_conv2 = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(bl8_conv1)

    bl8_conv3 = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(bl7_pl)
    bl8_add = Add()([bl8_conv3, bl8_conv2])
    bl8_pl = layers.UpSampling2D((2, 2))(bl8_add)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(bl8_pl)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def Unet_resnet_3param(input_side1, input2, n_channels=1, af='relu'):
    input_img1 = keras.Input(shape=(input_side1, input_side1, n_channels))
    input_img2 = keras.Input(shape=(input2,))

    dense_1 = layers.Dense(input_side1, activation="relu")(input_img2)
    dense_2 = layers.Dense(input_side1 ** 2, activation="relu")(dense_1)
    input_2 = layers.Reshape((input_side1, input_side1, 1))(dense_2)

    bl1_conv1 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img1)
    bl1_conv2 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(bl1_conv1)

    bl1_conv3 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img1)
    bl1_add = Add()([bl1_conv3, bl1_conv2, input_2])
    bl1_pl = layers.MaxPooling2D((2, 2))(bl1_add)

    bl2_conv1 = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(bl1_pl)
    bl2_conv2 = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(bl2_conv1)

    bl2_conv3 = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(bl1_pl)
    bl2_add = Add()([bl2_conv3, bl2_conv2])
    bl2_pl = layers.MaxPooling2D((2, 2))(bl2_add)

    bl3_conv1 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl2_pl)
    bl3_conv2 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl3_conv1)

    bl3_conv3 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl2_pl)
    bl3_add = Add()([bl3_conv3, bl3_conv2])
    bl3_pl = layers.MaxPooling2D((2, 2))(bl3_add)

    bl4_conv1 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl3_pl)
    bl4_conv2 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl4_conv1)

    bl4_conv3 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl3_pl)
    bl4_add = Add()([bl4_conv3, bl4_conv2])
    bl4_pl = layers.MaxPooling2D((5, 5))(bl4_add)

    encoded = layers.Conv2D(1, (4, 4), activation=af, strides=1, padding="same")(bl4_pl)

    bl5_conv1 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(encoded)
    bl5_conv2 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl5_conv1)

    bl5_conv3 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(encoded)
    bl5_add = Add()([bl5_conv3, bl5_conv2])
    bl5_pl = layers.UpSampling2D((5, 5))(bl5_add)

    bl6_conv1 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl5_pl)
    bl6_conv2 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl6_conv1)

    bl6_conv3 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl5_pl)
    bl6_add = Add()([bl6_conv3, bl6_conv2])
    bl6_pl = layers.UpSampling2D((2, 2))(bl6_add)

    bl7_conv1 = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(bl6_pl)
    bl7_conv2 = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(bl7_conv1)

    bl7_conv3 = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(bl6_pl)
    bl7_add = Add()([bl7_conv3, bl7_conv2])
    bl7_pl = layers.UpSampling2D((2, 2))(bl7_add)

    bl8_conv1 = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(bl7_pl)
    bl8_conv2 = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(bl8_conv1)

    bl8_conv3 = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(bl7_pl)
    bl8_add = Add()([bl8_conv3, bl8_conv2])
    bl8_pl = layers.UpSampling2D((2, 2))(bl8_add)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(bl8_pl)

    autoencoder_model = keras.Model(inputs=([input_img1, input_img2]), outputs=decoded)

    return autoencoder_model


def model_design_Unet_resnet2(input_side, n_channels=1, af='relu'):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    bl1_conv1 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img)
    bl1_conv2 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(bl1_conv1)

    # bl1_conv3 = layers.InputLayer(input_shape=(input_side, input_side, n_channels))
    # bl1_add = Add()([bl1_conv1, bl1_conv2])

    bl1_pl = layers.MaxPooling2D((2, 2))(bl1_conv2)

    bl2_conv1 = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(bl1_pl)
    bl2_conv2 = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(bl2_conv1)

    # bl2_conv3 = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(bl1_pl)
    # bl2_add = Add()([bl1_pl, bl2_conv2])
    bl2_concat = concatenate([bl1_pl, bl2_conv2], axis=3)
    bl2_pl = layers.MaxPooling2D((2, 2))(bl2_concat)

    bl3_conv1 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl2_pl)
    bl3_conv2 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl3_conv1)

    # bl3_conv3 = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(bl2_pl)
    # bl3_add = Add()([bl2_pl, bl3_conv2])
    bl3_concat = concatenate([bl2_pl, bl3_conv2], axis=3)
    bl3_pl = layers.MaxPooling2D((2, 2))(bl3_concat)

    bl4_conv1 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl3_pl)
    bl4_conv2 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl4_conv1)

    # bl4_conv3 = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(bl3_pl)
    # bl4_add = Add()([bl3_pl, bl4_conv2])
    bl4_concat = concatenate([bl3_pl, bl4_conv2], axis=3)
    bl4_pl = layers.MaxPooling2D((5, 5))(bl4_concat)

    encoded = layers.Conv2D(1, (4, 4), activation=af, strides=1, padding="same")(bl4_pl)

    bl5_conv1 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(encoded)
    bl5_conv2 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl5_conv1)

    # bl5_conv3 = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(encoded)
    # bl5_add = Add()([encoded, bl5_conv2])
    bl5_concat = concatenate([encoded, bl5_conv2], axis=3)
    bl5_pl = layers.UpSampling2D((5, 5))(bl5_concat)

    bl6_conv1 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl5_pl)
    bl6_conv2 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl6_conv1)

    # bl6_conv3 = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(bl5_pl)
    # bl6_add = Add()([bl5_pl, bl6_conv2])
    bl6_concat = concatenate([bl5_pl, bl6_conv2], axis=3)
    bl6_pl = layers.UpSampling2D((2, 2))(bl6_concat)

    bl7_conv1 = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(bl6_pl)
    bl7_conv2 = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(bl7_conv1)

    # bl7_conv3 = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(bl6_pl)
    # bl7_add = Add()([bl6_pl, bl7_conv2])
    bl7_concat = concatenate([bl6_pl, bl7_conv2], axis=3)
    bl7_pl = layers.UpSampling2D((2, 2))(bl7_concat)

    bl8_conv1 = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(bl7_pl)
    bl8_conv2 = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(bl8_conv1)

    # bl8_conv3 = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(bl7_pl)
    # bl8_add = Add()([bl7_pl, bl8_conv2])
    bl8_concat = concatenate([bl7_pl, bl8_conv2], axis=3)
    bl8_pl = layers.UpSampling2D((2, 2))(bl8_concat)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(bl8_pl)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def model_design_Unet2(input_side, n_channels=1, first_down_sampling=5):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(input_img)
    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((first_down_sampling, first_down_sampling))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.MaxPooling2D((5, 5))(x)

    x = layers.Conv2D(1, (2, 2), activation='relu', strides=1, padding="same")(x)

    # encoded = x

    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(encoded)
    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.UpSampling2D((5, 5))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((first_down_sampling, first_down_sampling))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def model_design_Unet3(input_side, n_channels):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(input_img)
    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((5, 5))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.MaxPooling2D((2, 2))(x)

    # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.MaxPooling2D((5, 5))(x)

    x = layers.Conv2D(1, (2, 2), activation='relu', strides=1, padding="same")(x)

    # encoded = x

    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(encoded)
    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.UpSampling2D((5, 5))(x)

    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((5, 5))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def basic_unet(input_side):
    inputs = Input((input_side, input_side, 1))
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    # conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    # conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    # conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    # conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    # conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    # conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool3)
    # conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    # conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(5, 5))(conv4)

    conv5 = Conv2D(1, (4, 4), activation='relu', padding='same')(pool4)
    # conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1, (4, 4), activation='relu', padding='same')(conv5)
    # conv5 = BatchNormalization()(conv5)

    up6 = concatenate([UpSampling2D((5, 5))(conv5), conv4], axis=3)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
    # conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
    # conv6 = BatchNormalization()(conv6)

    up7 = concatenate([UpSampling2D((2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    # conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)
    # conv7 = BatchNormalization()(conv7)

    up8 = concatenate([UpSampling2D((2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
    # conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)
    # conv8 = BatchNormalization()(conv8)

    up9 = concatenate([UpSampling2D((2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
    # conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)
    # conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model


def model_design_Unet_NF(input_side, z_size, flow_label=None, n_flows=5):
    if flow_label is None:
        flow_func = None
    elif flow_label == 'planar':
        flow_func = flow.Flow.uniform(n_flows, lambda i: Planar())
    elif flow_label == 'sylvester':
        flow_func = flow.Flow.uniform(n_flows, lambda i: TriangularSylvester(flip_z=i % 2 != 0))

    input_img = keras.Input(shape=(input_side, input_side, 1))

    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(input_img)
    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((5, 5))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.MaxPooling2D((5, 5))(x)

    h = layers.Conv2D(1, (4, 4), strides=1, padding="same")(x)

    z_mu = layers.Dense(z_size, activation='linear')(Flatten()(h))
    z_log_var = layers.Dense(z_size, activation='linear')(Flatten()(h))
    outputs = [z_mu, z_log_var]
    if flow is not None:
        z_shape = tf.TensorShape((None, z_size))
        params = layers.Dense(flow_func.param_count(z_shape), activation='linear')(Flatten()(h))
        outputs += [params]
        flow_func.initialize(z_shape)

    # z_mu = Input(shape=(z_size,))
    # z_log_var = Input(shape=(z_size,))
    # inputs = [z_mu, z_log_var]
    # if flow is not None:
    #     z_shape = tf.TensorShape((None, z_size))
    #     params = Input(shape=(flow.param_count(z_shape),))
    #     inputs += [params]
    #     flow.initialize(z_shape)
    flow_layer = FlowLayer(flow_func, min_beta=1.0E-3)
    zs, ldj, kld = flow_layer(outputs)
    z_k = zs[-1]
    s = 20  # the scale of dimension reduction from input to the bottleneck
    h_k = layers.Dense(input_side * input_side // s ** 2, activation='linear')(z_k)
    h_k = Reshape((input_side // s, input_side // s, 1))(h_k)

    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(h_k)
    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.UpSampling2D((5, 5))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(h_k)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((5, 5))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def vae_encoder(latent_dim, input_side, n_channels, af):
    encoder_inputs = keras.Input(shape=(input_side, input_side, n_channels))

    bl1_conv1 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(encoder_inputs)
    bl1_conv2 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(bl1_conv1)

    bl1_conv3 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(encoder_inputs)
    bl1_add = Add()([bl1_conv3, bl1_conv2])
    bl1_pl = layers.MaxPooling2D((2, 2))(bl1_add)

    bl2_conv1 = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(bl1_pl)
    bl2_conv2 = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(bl2_conv1)

    bl2_conv3 = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(bl1_pl)
    bl2_add = Add()([bl2_conv3, bl2_conv2])
    bl2_pl = layers.MaxPooling2D((2, 2))(bl2_add)

    bl3_conv1 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl2_pl)
    bl3_conv2 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl3_conv1)

    bl3_conv3 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl2_pl)
    bl3_add = Add()([bl3_conv3, bl3_conv2])
    bl3_pl = layers.MaxPooling2D((2, 2))(bl3_add)

    bl4_conv1 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl3_pl)
    bl4_conv2 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl4_conv1)

    bl4_conv3 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl3_pl)
    bl4_add = Add()([bl4_conv3, bl4_conv2])
    bl4_pl = layers.MaxPooling2D((5, 5))(bl4_add)

    encoded = layers.Conv2D(1, (4, 4), activation=af, strides=1, padding="same")(bl4_pl)

    x = layers.Flatten()(encoded)
    x = layers.Dense(latent_dim, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


def vae_encoder2(latent_dim, input_side, n_channels, af, flow=None):
    encoder_inputs = keras.Input(shape=(input_side, input_side, n_channels))

    bl1_conv1 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(encoder_inputs)
    # bl1_conv2 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(bl1_conv1)

    bl1_conv3 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(encoder_inputs)
    bl1_add = Add()([bl1_conv3, bl1_conv1])
    bl1_pl = layers.MaxPooling2D((5, 5))(bl1_add)

    bl2_conv1 = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(bl1_pl)
    # bl2_conv2 = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(bl2_conv1)

    bl2_conv3 = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(bl1_pl)
    bl2_add = Add()([bl2_conv3, bl2_conv1])
    bl2_pl = layers.MaxPooling2D((2, 2))(bl2_add)

    bl3_conv1 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl2_pl)
    # bl3_conv2 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl3_conv1)

    bl3_conv3 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl2_pl)
    bl3_add = Add()([bl3_conv3, bl3_conv1])
    bl3_pl = layers.MaxPooling2D((2, 2))(bl3_add)

    # bl4_conv1 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl3_pl)
    # # bl4_conv2 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl4_conv1)

    # bl4_conv3 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl3_pl)
    # bl4_add = Add()([bl4_conv3, bl4_conv1])
    # bl4_pl = layers.MaxPooling2D((5, 5))(bl4_add)

    encoded = layers.Conv2D(1, (2, 2), activation=af, strides=1, padding="same")(bl3_pl)

    x = layers.Flatten()(encoded)
    x = layers.Dense(latent_dim, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    # z = Sampling()([z_mean, z_log_var])
    outputs = [z_mean, z_log_var]
    if flow is not None:
        z_shape = tf.TensorShape((None, latent_dim))
        params = layers.Dense(flow.param_count(z_shape), activation='linear')(x)
        outputs += [params]
        flow.initialize(z_shape)

    flow_layer = FlowLayer(flow, min_beta=1.0E-3)
    zs, ldj, kld = flow_layer(outputs)
    z_k = zs[-1]

    encoder = keras.Model(encoder_inputs, [z_k, z_mean, z_log_var, ldj], name="encoder")
    return encoder


def vae_encoder_3params(latent_dim, input_side1, input_side2, n_channels, af):
    encoder_inputs1 = keras.Input(shape=(input_side1, input_side1, n_channels))
    encoder_inputs2 = keras.Input(shape=(input_side2,))

    dense_1 = layers.Dense(input_side1, activation="relu")(encoder_inputs2)
    dense_2 = layers.Dense(input_side1 ** 2, activation="relu")(dense_1)
    input_2 = layers.Reshape((input_side1, input_side1, 1))(dense_2)

    bl1_conv1 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(encoder_inputs1)
    bl1_conv2 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(bl1_conv1)

    bl1_conv3 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(encoder_inputs1)
    bl1_add = Add()([bl1_conv3, bl1_conv2, input_2])
    bl1_pl = layers.MaxPooling2D((2, 2))(bl1_add)

    bl2_conv1 = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(bl1_pl)
    bl2_conv2 = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(bl2_conv1)

    bl2_conv3 = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(bl1_pl)
    bl2_add = Add()([bl2_conv3, bl2_conv2])
    bl2_pl = layers.MaxPooling2D((2, 2))(bl2_add)

    bl3_conv1 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl2_pl)
    bl3_conv2 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl3_conv1)

    bl3_conv3 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl2_pl)
    bl3_add = Add()([bl3_conv3, bl3_conv2])
    bl3_pl = layers.MaxPooling2D((2, 2))(bl3_add)

    bl4_conv1 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl3_pl)
    bl4_conv2 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl4_conv1)

    bl4_conv3 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(bl3_pl)
    bl4_add = Add()([bl4_conv3, bl4_conv2])
    bl4_pl = layers.MaxPooling2D((5, 5))(bl4_add)

    encoded = layers.Conv2D(1, (4, 4), activation=af, strides=1, padding="same")(bl4_pl)

    x = layers.Flatten()(encoded)
    x = layers.Dense(latent_dim, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(inputs=[encoder_inputs1, encoder_inputs2],
                          outputs=[z_mean, z_log_var, z], name="encoder")
    return encoder


def vae_decoder(latent_dim, af, z_dim):
    latent_inputs = keras.Input(shape=(latent_dim,))

    x = layers.Dense(z_dim ** 2, activation="relu")(latent_inputs)
    x = layers.Reshape((z_dim, z_dim, 1))(x)

    bl5_conv1 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    bl5_conv2 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl5_conv1)

    bl5_conv3 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    bl5_add = Add()([bl5_conv3, bl5_conv2])
    bl5_pl = layers.UpSampling2D((5, 5))(bl5_add)

    bl6_conv1 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl5_pl)
    bl6_conv2 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl6_conv1)

    bl6_conv3 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl5_pl)
    bl6_add = Add()([bl6_conv3, bl6_conv2])
    bl6_pl = layers.UpSampling2D((2, 2))(bl6_add)

    bl7_conv1 = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(bl6_pl)
    bl7_conv2 = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(bl7_conv1)

    bl7_conv3 = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(bl6_pl)
    bl7_add = Add()([bl7_conv3, bl7_conv2])
    bl7_pl = layers.UpSampling2D((2, 2))(bl7_add)

    bl8_conv1 = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(bl7_pl)
    bl8_conv2 = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(bl8_conv1)

    bl8_conv3 = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(bl7_pl)
    bl8_add = Add()([bl8_conv3, bl8_conv2])
    bl8_pl = layers.UpSampling2D((2, 2))(bl8_add)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(bl8_pl)
    decoder = keras.Model(latent_inputs, decoded, name="decoder")
    return decoder


def vae_decoder2(latent_dim, af):
    latent_inputs = keras.Input(shape=(latent_dim,))

    x = layers.Dense(latent_dim, activation="relu")(latent_inputs)
    x = layers.Dense(2500, activation="relu")(x)
    x = layers.Reshape((50, 50, 1))(x)

    # bl5_conv1 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # # bl5_conv2 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl5_conv1)

    # bl5_conv3 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # bl5_add = Add()([bl5_conv3, bl5_conv1])
    # bl5_pl = layers.UpSampling2D((5, 5))(bl5_add)

    bl6_conv1 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # bl6_conv2 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl6_conv1)

    bl6_conv3 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    bl6_add = Add()([bl6_conv3, bl6_conv1])
    bl6_pl = layers.UpSampling2D((2, 2))(bl6_add)

    bl7_conv1 = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(bl6_pl)
    # bl7_conv2 = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(bl7_conv1)

    bl7_conv3 = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(bl6_pl)
    bl7_add = Add()([bl7_conv3, bl7_conv1])
    bl7_pl = layers.UpSampling2D((2, 2))(bl7_add)

    bl8_conv1 = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(bl7_pl)
    # bl8_conv2 = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(bl8_conv1)

    bl8_conv3 = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(bl7_pl)
    bl8_add = Add()([bl8_conv3, bl8_conv1])
    bl8_pl = layers.UpSampling2D((5, 5))(bl8_add)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(bl8_pl)
    decoder = keras.Model(latent_inputs, decoded, name="decoder")
    return decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder,
                 design='vae',
                 input_shape=(1000, 1000, 1),
                 **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.design = design
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def call(self, inputs, training=True):
        """
      This method is called when the model is being evaluated or used for inference.

      Args:
        inputs: The input tensor to the model.
        training: A boolean flag indicating whether or not the model is being trained.

      Returns:
        The output tensor from the model.
      """

        # The `call()` method should first check to see if the model is being trained.
        if training:
            # If the model is being trained, then the `call()` method should call the
            # `train_step()` method.
            return self.train_step(inputs)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_k, z_mean, z_log_var, ldj = self.encoder(data)
            if self.design == 'vae_3params':
                target = data[0]
            else:
                target = data
            reconstruction = self.decoder(z_k)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(target, reconstruction)) / 1000000
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def summary(self):
        x = tf.keras.Input(shape=self.input_shape, name="input_layer")
        self.model = tf.keras.Model(inputs=[x], outputs=self.call(x))
        return self.model.summary()

    def save(self, filepath):
        tf.saved_model.save(self.model, filepath)


def vae_NF(input_side, learning_rate=1E-4, flow_label=None, z_size=625, n_flows=4,
           loss=keras.losses.BinaryCrossentropy(from_logits=True)):
    """

    :type input_side: int
    """
    if flow_label is None:
        flow_func = None
    elif flow_label == 'planar':
        flow_func = flow.Flow.uniform(n_flows, lambda i: Planar())
    elif flow_label == 'sylvester':
        flow_func = flow.Flow.uniform(n_flows, lambda i: TriangularSylvester(flip_z=i % 2 != 0))

    beta_update = lambda i, beta: 1.0E-1 * (i + 1)
    vae = GatedConvVAE(input_side, input_side, flow_func, z_size=z_size,
                       loss=loss, learning_rate=learning_rate,
                       beta_update_fn=beta_update)
    return vae.model


def display_model(model):
    model.summary()


def read_saved_model(path):
    autoencoder_model = keras.models.load_model(path)
    return autoencoder_model


def scheduler(epoch, lr):
    step = epoch // 50
    if step == 0:
        return lr
    elif step == 1:
        return lr / 10
    elif step == 2:
        return lr / 100
    elif step == 3:
        return lr / 1000
    elif step > 3:
        return lr / 10000


def model_compile_change_lc_loss(model, learning_rate,
                                 lc_loss_metric,
                                 num_epochs,
                                 training_generator, validation_generator,
                                 output_dir,
                                 optimizer_=keras.optimizers.Adam,
                                 second_coeff=10,
                                 early_callback_=False, early_callback_type='early_stop'):
    loss1 = lc_loss_func(lc_loss_metric, 1)
    loss2 = lc_loss_func(lc_loss_metric, second_coeff)
    n_epochs1 = num_epochs // 2
    n_epochs2 = num_epochs - n_epochs1
    model.compile(optimizer=optimizer_(learning_rate=learning_rate), loss=loss1)
    # Train the model
    history1 = model_fit2(model, n_epochs1,
                          training_generator, validation_generator,
                          filepath=output_dir,
                          early_callback_=early_callback_,
                          early_callback__type=early_callback_type)

    model.compile(optimizer=optimizer_(learning_rate=learning_rate), loss=loss2)
    # Train the model
    history1 = model_fit2(model, n_epochs2,
                          training_generator, validation_generator,
                          filepath=output_dir,
                          early_callback_=early_callback_,
                          early_callback_type=early_callback_type)

    return history1


def model_compile(model, learning_rate, design, optimizer_=keras.optimizers.Adam,
                  loss=keras.losses.BinaryCrossentropy(from_logits=True)):
    if design == 'VAE_Unet_Resnet':
        model.compile(optimizer=optimizer_(learning_rate=learning_rate))
    else:
        model.compile(optimizer=optimizer_(learning_rate=learning_rate),
                      loss=loss)


def model_fit(model, epochs, training_generator, validation_generator,
              callback=[]):
    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=epochs,
                                  callbacks=callback,
                                  use_multiprocessing=True)
    return history


def model_fit2(model, epochs, training_generator, validation_generator, filepath=None,
               early_callback_=False, early_callback_type='early_stop'):
    ec = []
    if early_callback_:

        if early_callback_type == 'early_stop':
            ec.append(EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, min_delta=0.00001))
        elif early_callback_type == 'model_checkpoint':
            ec.append(ModelCheckpoint(
                filepath=filepath,
                save_freq='epoch'))

    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=epochs,
                                  callbacks=ec,
                                  use_multiprocessing=True)
    return history


def fig_loss(model_history):
    fig = plt.figure(figsize=(8, 8))
    plt.plot(np.array(model_history['val_loss']), label='Validation loss')
    plt.plot(np.array(model_history['loss']), label='loss')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend()
    return fig


def compareinout2D(outim, testimg, dim):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(131)
    ax.imshow(testimg.reshape((dim, dim)), cmap='gray', norm=LogNorm())
    #     pl.axis('off')
    #     ax.imshow(testimg.reshape(initialshape) , cmap="bone")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = fig.add_subplot(132)
    #     ax.imshow(outim.reshape(initialshape) , cmap="bone")
    ax.imshow(outim.reshape((dim, dim)), cmap='gray', norm=LogNorm())
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # l2 = np.sum((testimg - outim) ** 2)
    # print('l2 value = ', l2)

    return fig

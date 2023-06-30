"""Some code from:

https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
https://github.com/bgroenks96/normalizing-flows
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
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Add
import sys

from convolver import *

sys.path += ['..']
# path_to_module = '~/codes/normalizing_flows/normalizing_flows/'
# sys.path.append(path_to_module)
from codes.normalizing_flows.normalizing_flows.layers import FlowLayer
from codes.normalizing_flows.normalizing_flows.flows.affine import Planar, Radial
from codes.normalizing_flows.normalizing_flows.flows import flow
from codes.normalizing_flows.normalizing_flows.flows.sylvester import TriangularSylvester
from codes.normalizing_flows.normalizing_flows.models.vae2 import GatedConvVAE

from keras.callbacks import EarlyStopping, ModelCheckpoint


def tweedie_loss_func(p):
    def tweedie_loglikelihood(y, y_hat):
        # # print(np.mean(y.eval(session=tf.compat.v1.Session())),
        # #       np.mean(y_hat.eval(session=tf.compat.v1.Session())), "Inside loss function")
        # init_op = tf.initialize_all_variables()
        # with tf.Session() as sess:
        #     sess.run(init_op)  # execute init_op
        #     # print the random values that we sample
        #     print(sess.run(y_hat))
        tf.print("y_hat:", y_hat, output_stream=sys.stdout)
        loss = - y * tf.pow(y_hat, 1 - p) / (1 - p) + \
               tf.pow(y_hat, 2 - p) / (2 - p)
        return tf.reduce_mean(loss)

    return tweedie_loglikelihood


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, list_IDs, batch_size=8, dim=10000, n_channels=1,
                 res_scale=10, path='./../../../fred/oz108/GERLUMPH_project/DATABASES/gerlumph_db/',
                 shuffle=True, conv_const='./../data/all_maps_meta.csv',
                 convolve=False, rsrc=1):
        """Initialization"""

        self.dim = (int(dim / res_scale), int(dim / res_scale))
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.res_scale = res_scale
        self.path = path
        self.indexes = np.arange(len(self.list_IDs), dtype=int)
        self.on_epoch_end()
        self.conv_const_path = conv_const
        self.convolve = convolve
        if self.convolve:
            self.cv = convolver(rsrc=rsrc)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        self.list_IDs_temp = [self.list_IDs[k] for k in indexes]
        meta = pd.read_csv(self.conv_const_path)
        self.conv_const = meta.loc[meta['ID'].isin(self.list_IDs_temp)]['const'].values

        # Generate data
        X, y = self.__data_generation()

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self):
        """Generates data containing batch_size samples"""  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))
        if self.convolve:
            Y = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels))

        # Generate data
        for i, ID in enumerate(self.list_IDs_temp):
            # Store sample
            # try:
            f = open(self.path + str(ID) + "/map.bin", "rb")
            map_tmp = np.fromfile(f, 'i', -1, "")
            maps = (np.reshape(map_tmp, (-1, 10000)))
            tmp = maps[0::self.res_scale, 0::self.res_scale].reshape((self.dim[0], self.dim[1], self.n_channels))
            # print(np.mean(tmp))
            tmp = tmp * self.conv_const[i]
            # print(np.mean(tmp))
            if self.convolve:
                Y[i, :, :, int(self.n_channels - 1)] = self.convolve_maps(tmp.reshape((self.dim[0], self.dim[1])))
            # print(self.convolve)
            tmp = np.log10(tmp + 0.004)
            X[i,] = NormalizeData(tmp)
            # print(np.mean(NormalizeData(tmp)))

            # except ValueError:
            #     pass
        if self.convolve:
            return X, Y
        else:
            return X, X

    def convolve_maps(self, mag):
        self.cv.conv_map(mag)
        conv_map_tmp = self.cv.magcon
        conv_map_tmp = np.log10(conv_map_tmp)
        return NormalizeData(conv_map_tmp)

    def get_generator_indexes(self):
        return self.indexes

    def map_reader(self, list_IDs_temp, output='map'):
        '''
        This function is written to help see if the pipeline for reading the maps is working fine.
        For a given set of IDs, It reads the map, and takes log10, and normalized them to be between -3 and 6.
        It also does the convolution with source profile and returns that if the option is chosen.

        :param list_IDs_temp: A list of IDs of maps that you want to read
        :param output: What the output should be. The options are:
                "map": If you want the maps in units of magnification
                "map_norm": If you want the maps normalized.
                "map_conv": If you want the convolved maps with the source profile in magnification.
                "map_conv_norm": If you want the normalized convolved maps.
        :return:
        '''
        X = np.empty((len(list_IDs_temp), self.dim[0], self.dim[1], self.n_channels))
        meta = pd.read_csv(self.conv_const_path)
        conv_const = meta.loc[meta['ID'].isin(list_IDs_temp)]['const'].values
        # macro_mag = meta.loc[meta['ID'].isin(list_IDs_temp)]['const'].values

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # try:
            f = open(self.path + str(ID) + "/map.bin", "rb")
            map_tmp = np.fromfile(f, 'i', -1, "")
            maps = (np.reshape(map_tmp, (-1, 10000)))
            tmp = maps[0::self.res_scale, 0::self.res_scale].reshape((self.dim[0], self.dim[1], self.n_channels))
            tmp = tmp * conv_const[i]
            if output == 'map':
                X[i, :, :, 0] = tmp.reshape((self.dim[0], self.dim[1]))
            elif output == 'map_norm':
                tmp = np.log10(tmp + 0.004)
                X[i, :, :, 0] = NormalizeData(tmp).reshape((self.dim[0], self.dim[1]))
            elif output == 'map_conv':
                self.convolve_maps(tmp.reshape((self.dim[0], self.dim[1])))
                X[i, :, :, 0] = self.cv.magcon
            elif output == 'map_conv_norm':
                self.convolve_maps(tmp.reshape((self.dim[0], self.dim[1])))
                tmp_conv = np.log10(self.cv.magcon)
                X[i, :, :, 0] = NormalizeData(tmp_conv).reshape((self.dim[0], self.dim[1]))

        return X


def NormalizeData(data):
    data_max = 6
    data_min = -3
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


def model_design_Unet(input_side):
    input_img = keras.Input(shape=(input_side, input_side, 1))

    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(input_img)
    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

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


def model_design_Unet2(input_side):
    input_img = keras.Input(shape=(input_side, input_side, 1))

    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(input_img)
    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (2, 2), activation='relu', strides=1, padding="valid")(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (2, 2), activation='relu', strides=1, padding="valid")(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(1, (4, 4), activation='relu', strides=1, padding="same")(x)

    encoded = x

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(encoded)
    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (2, 2), activation='relu', strides=1, padding="valid")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (2, 2), activation='relu', strides=1, padding="valid")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (2, 2), activation='relu', strides=1, padding="valid")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def model_design_Unet3(input_side):
    input_img = keras.Input(shape=(input_side, input_side, 1))

    x = layers.Conv2D(32, (2, 2), activation='relu', strides=1, padding="same")(input_img)
    x = layers.Conv2D(32, (2, 2), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(32, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(32, (2, 2), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(64, (2, 2), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (2, 2), activation='relu', strides=1, padding="valid")(x)
    x = layers.Conv2D(64, (2, 2), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2D(128, (2, 2), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (2, 2), activation='relu', strides=1, padding="valid")(x)
    x = layers.Conv2D(128, (2, 2), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(1, (4, 4), activation='relu', strides=1, padding="same")(x)

    encoded = x

    x = layers.Conv2DTranspose(128, (2, 2), activation='relu', strides=1, padding="same")(encoded)
    x = layers.Conv2DTranspose(128, (2, 2), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(128, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (2, 2), activation='relu', strides=1, padding="valid")(x)
    # x = layers.Conv2DTranspose(128, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (2, 2), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (2, 2), activation='relu', strides=1, padding="valid")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (2, 2), activation='relu', strides=1, padding="valid")(x)
    # x = layers.Conv2DTranspose(64, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (2, 2), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (2, 2), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (2, 2), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

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


def model_design_Unet_NF(input_side, z_size):
    input_img = keras.Input(shape=(input_side, input_side, 1))

    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(input_img)
    x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

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

    h = layers.Conv2D(1, (4, 4), activation='relu', strides=1, padding="same")(x)

    z_mu = layers.Dense(z_size, activation='linear')(Flatten()(h))
    z_log_var = layers.Dense(z_size, activation='linear')(Flatten()(h))
    outputs = [z_mu, z_log_var]
    if flow is not None:
        z_shape = tf.TensorShape((None, z_size))
        params = layers.Dense(flow.param_count(z_shape), activation='linear')(Flatten()(h))
        outputs += [params]

    z_mu = Input(shape=(z_size,))
    z_log_var = Input(shape=(z_size,))
    inputs = [z_mu, z_log_var]
    if flow is not None:
        z_shape = tf.TensorShape((None, z_size))
        params = Input(shape=(flow.param_count(z_shape),))
        inputs += [params]
        flow.initialize(z_shape)
    flow_layer = FlowLayer(flow, min_beta=1.0E-3)
    zs, ldj, kld = flow_layer(inputs)
    z_k = zs[-1]
    s = 40  # the scale of dimension reduction from input to the bottleneck
    h_k = layers.Dense(input_side * input_side // s ** 2, activation='linear')(z_k)
    h_k = Reshape((input_side // s, input_side // s, 1))(h_k)

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


def model_compile(model, learning_rate, optimizer_=keras.optimizers.Adam,
                  loss=keras.losses.BinaryCrossentropy(from_logits=True)):
    model.compile(optimizer=optimizer_(learning_rate=learning_rate),
                  loss=loss)


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
                                  callbacks=ec)
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

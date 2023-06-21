"""Code from:

https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

"""
from re import X
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D,\
                         Dense, LeakyReLU, Reshape, Flatten, Add, InputLayer
import sys
sys.path += ['..']
# path_to_module = '~/codes/normalizing_flows/normalizing_flows/'
# sys.path.append(path_to_module)

from normalizing_flows.normalizing_flows.flows.affine import Planar, Radial
from normalizing_flows.normalizing_flows.flows import flow
from normalizing_flows.normalizing_flows.flows.sylvester import TriangularSylvester
from normalizing_flows.normalizing_flows.models.vae2 import GatedConvVAE


def tweedie_loss_func(p):
    def tweedie_loglikelihood(y, y_hat):
        loss = - y * tf.pow(y_hat, 1 - p) / (1 - p) + \
               tf.pow(y_hat, 2 - p) / (2 - p)
        return tf.reduce_mean(loss)

    return tweedie_loglikelihood


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""

    def __init__(self, IDs_params, data, batch_size=8, dim=10000, n_channels=1,
                 res_scale=10,
                 shuffle=True):
        """Initialization"""

        self.dim = (int(dim / res_scale), int(dim / res_scale))
        self.data = data
        self.batch_size = batch_size
        self.list_IDs = IDs_params[:,0]
        self.conv_const = IDs_params[:,-1]
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.res_scale = res_scale
        self.indexes = np.arange(len(self.list_IDs), dtype=int)
        self.on_epoch_end()
        

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        self.list_IDs_temp = [self.list_IDs[k] for k in indexes]
        self.conv_const_temp = [self.conv_const[k] for k in indexes]
        self.data_tmp = np.asarray([self.data[k, :, :, :] for k in indexes])
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
        Y = np.empty((self.batch_size, self.dim[0], self.dim[1], 1))

        # Generate data
        for i, ID in enumerate(self.list_IDs_temp):
              X[i, :, :, :] = self.data_tmp[i].reshape((self.dim[0], self.dim[1], self.n_channels))
              Y[i, ] = self.data_tmp[i, :, :, 0].reshape((self.dim[0], self.dim[1], 1))
        return X, Y

    def get_generator_indexes(self):
        return self.indexes

    # def map_reader(self, list_IDs_temp):
    #     X = np.empty((len(list_IDs_temp), self.dim[0], self.dim[1], self.n_channels))
    #     self.data_tmp = np.asarray([self.data[k, :, :, 0] for k in indexes])
    #     # Generate data
    #     for i, ID in enumerate(list_IDs_temp):
    #         X[i,] = self.data[i]

    #     return X


def NormalizeData(data, max_, min_):
    data_new = (data - min_) / (max_ - min_)
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


def model_design_Unet(input_side, n_channels = 1, af = 'relu'):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(input_img)
    x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((5, 5))(x)

    x = layers.Conv2D(1, (4, 4), activation= af, strides=1, padding="same")(x)

    encoded = x

    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(encoded)
    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((5, 5))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model

def model_design_Unet_resnet(input_side, n_channels = 1, af = 'relu'):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    bl1_conv1 = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(input_img)
    bl1_conv2 = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(bl1_conv1)

    bl1_conv3 = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(input_img)
    bl1_add = Add()([bl1_conv3, bl1_conv2])
    bl1_pl = layers.MaxPooling2D((2, 2))(bl1_add)

    bl2_conv1 = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(bl1_pl)
    bl2_conv2 = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(bl2_conv1)

    bl2_conv3 = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(bl1_pl)
    bl2_add = Add()([bl2_conv3, bl2_conv2])
    bl2_pl = layers.MaxPooling2D((2, 2))(bl2_add)

    bl3_conv1 = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(bl2_pl)
    bl3_conv2 = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(bl3_conv1)

    bl3_conv3 = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(bl2_pl)
    bl3_add = Add()([bl3_conv3, bl3_conv2])
    bl3_pl = layers.MaxPooling2D((2, 2))(bl3_add)

    bl4_conv1 = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(bl3_pl)
    bl4_conv2 = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(bl4_conv1)

    bl4_conv3 = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(bl3_pl)
    bl4_add = Add()([bl4_conv3, bl4_conv2])
    bl4_pl = layers.MaxPooling2D((5, 5))(bl4_add)

    encoded = layers.Conv2D(1, (4, 4), activation= af, strides=1, padding="same")(bl4_pl)

    bl5_conv1 = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(encoded)
    bl5_conv2 = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(bl5_conv1)

    bl5_conv3 = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(encoded)
    bl5_add = Add()([bl5_conv3, bl5_conv2])
    bl5_pl = layers.UpSampling2D((5, 5))(bl5_add)

    bl6_conv1 = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(bl5_pl)
    bl6_conv2 = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(bl6_conv1)

    bl6_conv3 = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(bl5_pl)
    bl6_add = Add()([bl6_conv3, bl6_conv2])
    bl6_pl = layers.UpSampling2D((2, 2))(bl6_add)

    bl7_conv1 = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(bl6_pl)
    bl7_conv2 = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(bl7_conv1)

    bl7_conv3 = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(bl6_pl)
    bl7_add = Add()([bl7_conv3, bl7_conv2])
    bl7_pl = layers.UpSampling2D((2, 2))(bl7_add)

    bl8_conv1 = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(bl7_pl)
    bl8_conv2 = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(bl8_conv1)

    bl8_conv3 = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(bl7_pl)
    bl8_add = Add()([bl8_conv3, bl8_conv2])
    bl8_pl = layers.UpSampling2D((2, 2))(bl8_add)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(bl8_pl)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model

def model_design_Unet_resnet2(input_side, n_channels = 1, af = 'relu'):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    bl1_conv1 = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(input_img)
    bl1_conv2 = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(bl1_conv1)

    # bl1_conv3 = layers.InputLayer(input_shape=(input_side, input_side, n_channels))
    # bl1_add = Add()([bl1_conv1, bl1_conv2])

    bl1_pl = layers.MaxPooling2D((2, 2))(bl1_conv2)

    bl2_conv1 = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(bl1_pl)
    bl2_conv2 = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(bl2_conv1)

    # bl2_conv3 = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(bl1_pl)
    # bl2_add = Add()([bl1_pl, bl2_conv2])
    bl2_concat = concatenate([bl1_pl, bl2_conv2], axis=3)
    bl2_pl = layers.MaxPooling2D((2, 2))(bl2_concat)

    bl3_conv1 = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(bl2_pl)
    bl3_conv2 = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(bl3_conv1)

    # bl3_conv3 = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(bl2_pl)
    # bl3_add = Add()([bl2_pl, bl3_conv2])
    bl3_concat = concatenate([bl2_pl, bl3_conv2], axis=3)
    bl3_pl = layers.MaxPooling2D((2, 2))(bl3_concat)

    bl4_conv1 = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(bl3_pl)
    bl4_conv2 = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(bl4_conv1)

    # bl4_conv3 = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(bl3_pl)
    # bl4_add = Add()([bl3_pl, bl4_conv2])
    bl4_concat = concatenate([bl3_pl, bl4_conv2], axis=3)
    bl4_pl = layers.MaxPooling2D((5, 5))(bl4_concat)

    encoded = layers.Conv2D(1, (4, 4), activation= af, strides=1, padding="same")(bl4_pl)

    bl5_conv1 = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(encoded)
    bl5_conv2 = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(bl5_conv1)

    # bl5_conv3 = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(encoded)
    # bl5_add = Add()([encoded, bl5_conv2])
    bl5_concat = concatenate([encoded, bl5_conv2], axis=3)
    bl5_pl = layers.UpSampling2D((5, 5))(bl5_concat)

    bl6_conv1 = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(bl5_pl)
    bl6_conv2 = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(bl6_conv1)

    # bl6_conv3 = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(bl5_pl)
    # bl6_add = Add()([bl5_pl, bl6_conv2])
    bl6_concat = concatenate([bl5_pl, bl6_conv2], axis=3)
    bl6_pl = layers.UpSampling2D((2, 2))(bl6_concat)

    bl7_conv1 = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(bl6_pl)
    bl7_conv2 = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(bl7_conv1)

    # bl7_conv3 = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(bl6_pl)
    # bl7_add = Add()([bl6_pl, bl7_conv2])
    bl7_concat = concatenate([bl6_pl, bl7_conv2], axis=3)
    bl7_pl = layers.UpSampling2D((2, 2))(bl7_concat)

    bl8_conv1 = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(bl7_pl)
    bl8_conv2 = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(bl8_conv1)

    # bl8_conv3 = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(bl7_pl)
    # bl8_add = Add()([bl7_pl, bl8_conv2])
    bl8_concat = concatenate([bl7_pl, bl8_conv2], axis=3)
    bl8_pl = layers.UpSampling2D((2, 2))(bl8_concat)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(bl8_pl)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def model_design_Unet_flatten(input_side, n_channels = 1, af = 'relu', z_flatten_dim = 625):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(input_img)
    x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((5, 5))(x)

    x = layers.Conv2D(1, (4, 4), activation= af, strides=1, padding="same")(x)

    encoded = Flatten()(x)

    x = Dense(z_flatten_dim)(x)
    x = LeakyReLU(0.1)(x)

    encoder_output = Dense(300)(x)

    x = Dense(z_flatten_dim, activation= af)(encoder_output)
    x = Reshape((25, 25, 1))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(encoded)
    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((5, 5))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def model_design_Unet2(input_side, n_channels = 1, af = 'relu'):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(input_img)
    x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # x = layers.Conv2D(64, (2, 2), activation= af, strides=1, padding="valid")(x)
    # x = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((5, 5))(x)

    x = layers.Conv2D(1, (4, 4), activation= af, strides=1, padding="same")(x)

    encoded = x

    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(encoded)
    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((5, 5))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    # x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="valid")(x)
    # x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    # x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def model_design_Unet3(input_side, n_channels = 1, af = 'relu'):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(input_img)
    x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(x)

    x = layers.MaxPooling2D((2, 2))(x)

    # x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(x)    
    x = layers.MaxPooling2D((2, 2))(x)

    # x = layers.Conv2D(64, (2, 2), activation= af, strides=1, padding="valid")(x)
    # x = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((5, 5))(x)

    x = layers.Conv2D(1, (4, 4), activation= af, strides=1, padding="same")(x)

    encoded = x

    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(encoded)
    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((5, 5))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    # x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="valid")(x)
    # x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation= af, strides=1, padding="same")(x)

    x = layers.UpSampling2D((2, 2))(x)

    # x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    # x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation= af, strides=1, padding="same")(x)

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

def basic_unet2(input_side):
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

    up6 = UpSampling2D((5, 5))(conv5)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(up6)
    # conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)
    # conv6 = BatchNormalization()(conv6)

    up7 = UpSampling2D((2, 2))(conv6)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(up7)
    # conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv7)
    # conv7 = BatchNormalization()(conv7)

    up8 = UpSampling2D((2, 2))(conv7)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
    # conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)
    # conv8 = BatchNormalization()(conv8)

    up9 = UpSampling2D((2, 2))(conv8)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
    # conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)
    # conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model


def vae_NF(input_side, learning_rate = 1E-4, flow_label=None, z_size=625, n_flows=4,
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

def model_fit1(model, epochs, batch_size,
               training):
    history = model.fit(training, training,
                        validation_split = 0.2,
                        epochs=epochs,
                        batch_size=batch_size,
                        )

    return history

def model_fit2(model, epochs, training_generator,
               validation_generator, model_design = 'Unet',
               es = []):
            
    if model_design == 'Unet_NF':
        history = model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                            epochs=epochs, callbacks=[es])
    else:
        history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=epochs,
                                  use_multiprocessing=True,
                                  workers=4,
                                  callbacks=[es]
                                  )

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

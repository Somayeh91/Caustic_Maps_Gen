
from tensorflow import keras
from keras import layers
from tensorflow.keras.layers import Flatten, Reshape, Dense, Concatenate
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Add, \
    BatchNormalization
import sys
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from keras.saving import saving_api


sys.path += ['..']
path_to_module = '~/codes/normalizing_flows/normalizing_flows/'
sys.path.append(path_to_module)
from codes.normalizing_flows.normalizing_flows.layers import FlowLayer
# from codes.normalizing_flows.normalizing_flows.flows.affine import Planar, Radial
# from codes.normalizing_flows.normalizing_flows.flows import flow
# from codes.normalizing_flows.normalizing_flows.flows.sylvester import TriangularSylvester
# from codes.normalizing_flows.normalizing_flows.models.vae2 import GatedConvVAE


# Design model
def model_design_2l(input_side, input2=None, n_channels=1, z_size=625, flow_label=None, n_flows=5, first_down_sampling=5, af='relu'):
    input_img = keras.Input(shape=(input_side, input_side, 1))

    x = layers.Conv2D(128, (5, 5), activation=af,
                      strides=(4, 4), padding="same")(input_img)
    x = layers.Conv2D(128, (6, 6), activation=af)(x)
    x = layers.MaxPooling2D((first_down_sampling, first_down_sampling))(x)
    x = layers.Conv2D(1, (5, 5), activation=af)(x)
    encoded = layers.MaxPooling2D((5, 5))(x)

    x = layers.UpSampling2D((5, 5))(encoded)
    x = layers.Conv2DTranspose(128, (5, 5), activation=af)(x)
    x = layers.UpSampling2D((first_down_sampling, first_down_sampling))(x)
    x = layers.Conv2DTranspose(128, (6, 6), activation=af)(x)
    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid',
                                     strides=(4, 4), padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def model_design_3l(input_side, input2=None, n_channels=1, z_size=625, flow_label=None, n_flows=5, first_down_sampling=5, af='relu'):
    input_img = keras.Input(shape=(input_side, input_side, 1))

    x = layers.Conv2D(32, (4, 4), activation=af,
                      strides=1, padding="same")(input_img)
    x = layers.MaxPooling2D((first_down_sampling, first_down_sampling))(x)
    x = layers.Conv2D(128, (4, 4), activation=af,
                      strides=1, padding="same")(x)
    x = layers.MaxPooling2D((5, 5))(x)
    x = layers.Conv2D(256, (4, 4), activation=af,
                      strides=1, padding="same")(x)
    x = layers.MaxPooling2D((5, 5))(x)
    x = layers.Conv2D(1, (4, 4), activation=af,
                      strides=1, padding="same")(x)
    encoded = x  # layers.MaxPooling2D((5, 5))(x)

    # x = layers.UpSampling2D((5, 5))(encoded)
    x = layers.Conv2DTranspose(256, (4, 4), activation=af,
                               strides=1, padding="same")(encoded)
    x = layers.UpSampling2D((5, 5))(x)
    x = layers.Conv2DTranspose(128, (4, 4), activation=af,
                               strides=1, padding="same")(x)
    x = layers.UpSampling2D((5, 5))(x)
    x = layers.Conv2DTranspose(32, (4, 4), activation=af,
                               strides=1, padding="same")(x)
    x = layers.UpSampling2D((first_down_sampling, first_down_sampling))(x)
    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid',
                                     padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def model_design_Unet(input_side, input2=None, n_channels=1, z_size=625, flow_label=None, n_flows=5, first_down_sampling=2, af='relu'):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img)
    x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((first_down_sampling, first_down_sampling))(x)

    x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((5, 5))(x)

    x = layers.Conv2D(1, (4, 4), activation=af, strides=1, padding="same")(x)

    encoded = x

    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(encoded)
    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((5, 5))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((first_down_sampling, first_down_sampling))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def Unet_lens_pos(input_side1, input2=None, n_channels=1, z_size=625, flow_label=None, n_flows=5, first_down_sampling=5, af='relu'):
    input_img1 = keras.Input(shape=(input_side1, input_side1, n_channels))
    input_img2 = keras.Input(shape=(input2, input2, n_channels))

    x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img1)
    # x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation= af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((first_down_sampling, first_down_sampling))(x)

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
    x = layers.UpSampling2D((first_down_sampling, first_down_sampling))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model([input_img1, input_img2], decoded)

    return autoencoder_model


def Unet_take_two_channels_separately(input_side, input2=None, n_channels=1, z_size=625, flow_label=None, n_flows=5, first_down_sampling=2, af='relu'):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))
    input_img2 = keras.Input(shape=(input_side, input_side, n_channels))

    x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img)
    x2 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img2)
    x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x2 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(x2)
    x3 = layers.Add()([x, x2])
    x = layers.MaxPooling2D((2, 2))(x3)

    x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((5, 5))(x)

    x = layers.Conv2D(1, (4, 4), activation=af, strides=1, padding="same")(x)

    encoded = x

    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(encoded)
    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((5, 5))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((first_down_sampling, first_down_sampling))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def Unet_sobel_edges1(input_side, input2=None, n_channels=1, z_size=625, flow_label=None, n_flows=5, first_down_sampling=2, af='relu'):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    grad_components = tf.image.sobel_edges(input_img)
    grad_mag_components = grad_components ** 2
    grad_mag_square = tf.math.reduce_sum(grad_mag_components, axis=-1)  # sum all magnitude components
    grad_mag_img = tf.sqrt(grad_mag_square)
    # grad_mag_img = BatchNormalization()(grad_mag_img)
    mask = tf.greater(grad_mag_img, 0.0001 * tf.ones_like(grad_mag_img))
    grad_mag_img = tf.multiply(grad_mag_img, tf.cast(mask, tf.float32))

    x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img)
    x2 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(grad_mag_img)
    x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = Add()([x, x2])
    x = layers.MaxPooling2D((first_down_sampling, first_down_sampling))(x)

    x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((5, 5))(x)

    x = layers.Conv2D(1, (4, 4), activation=af, strides=1, padding="same")(x)

    encoded = x

    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(encoded)
    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((5, 5))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=1, padding="same")(x)
    x = layers.UpSampling2D((first_down_sampling, first_down_sampling))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def Unet_sobel_edges2(input_side, input2=None, n_channels=1, z_size=625, flow_label=None, n_flows=5, first_down_sampling=2, af='relu'):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    grad_components = tf.image.sobel_edges(input_img)
    grad_mag_components = grad_components ** 2
    grad_mag_square = tf.math.reduce_sum(grad_mag_components, axis=-1)  # sum all magnitude components
    grad_mag_img = tf.sqrt(grad_mag_square)
    # grad_mag_img = BatchNormalization()(grad_mag_img)
    mask = tf.greater(grad_mag_img, 0.0001 * tf.ones_like(grad_mag_img))
    grad_mag_img = tf.multiply(grad_mag_img, tf.cast(mask, tf.float32))

    x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img)
    x2 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(grad_mag_img)
    x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = Add()([x, x2])
    x = layers.MaxPooling2D((first_down_sampling, first_down_sampling))(x)
    x2 = layers.MaxPooling2D((first_down_sampling, first_down_sampling))(x2)

    x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x3 = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x2)
    x = Add()([x, x3])
    x = layers.MaxPooling2D((2, 2))(x)
    x3 = layers.MaxPooling2D((2, 2))(x3)

    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x4 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x3)
    x = Add()([x, x4])
    x = layers.MaxPooling2D((2, 2))(x)
    x4 = layers.MaxPooling2D((2, 2))(x4)

    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x5 = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x4)
    x = Add()([x, x5])
    x = layers.MaxPooling2D((5, 5))(x)
    x5 = layers.MaxPooling2D((2, 2))(x5)

    x = layers.Conv2D(1, (4, 4), activation=af, strides=1, padding="same")(x)

    encoded = x

    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(encoded)
    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((5, 5))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((first_down_sampling, first_down_sampling))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def model_design_Unet_resnet(input_side, input2=None, n_channels=1, z_size=625, flow_label=None, n_flows=5, first_down_sampling=2, af='relu'):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    bl1_conv1 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img)
    bl1_conv2 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(bl1_conv1)

    bl1_conv3 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img)
    bl1_add = Add()([bl1_conv3, bl1_conv2])
    bl1_pl = layers.MaxPooling2D((first_down_sampling, first_down_sampling))(bl1_add)

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
    bl8_pl = layers.UpSampling2D((first_down_sampling, first_down_sampling))(bl8_add)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(bl8_pl)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def Unet_resnet_3param(input_side1, input2, n_channels=1, z_size=625, flow_label=None, n_flows=5, first_down_sampling=2, af='relu'):
    input_img1 = keras.Input(shape=(input_side1, input_side1, n_channels))
    input_img2 = keras.Input(shape=(input2,))

    dense_1 = layers.Dense(input_side1, activation="relu")(input_img2)
    dense_2 = layers.Dense(input_side1 ** 2, activation="relu")(dense_1)
    input_2 = layers.Reshape((input_side1, input_side1, 1))(dense_2)

    bl1_conv1 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img1)
    bl1_conv2 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(bl1_conv1)

    bl1_conv3 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img1)
    bl1_add = Add()([bl1_conv3, bl1_conv2, input_2])
    bl1_pl = layers.MaxPooling2D((first_down_sampling, first_down_sampling))(bl1_add)

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
    bl8_pl = layers.UpSampling2D((first_down_sampling, first_down_sampling))(bl8_add)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(bl8_pl)

    autoencoder_model = keras.Model(inputs=([input_img1, input_img2]), outputs=decoded)

    return autoencoder_model


def model_design_Unet_resnet2(input_side, input2=None, n_channels=1, z_size=None, flow_label=None, n_flows=None, first_down_sampling=2, af='relu'):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    bl1_conv1 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img)
    bl1_conv2 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(bl1_conv1)

    # bl1_conv3 = layers.InputLayer(input_shape=(input_side, input_side, n_channels))
    # bl1_add = Add()([bl1_conv1, bl1_conv2])

    bl1_pl = layers.MaxPooling2D((first_down_sampling, first_down_sampling))(bl1_conv2)

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
    bl8_pl = layers.UpSampling2D((first_down_sampling, first_down_sampling))(bl8_concat)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(bl8_pl)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def model_design_Unet2(input_side, input2=None, n_channels=1, z_size=None, flow_label=None, n_flows=None, first_down_sampling=5, af='relu'):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img)
    x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((first_down_sampling, first_down_sampling))(x)

    x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # # x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.MaxPooling2D((5, 5))(x)

    x = layers.Conv2D(1, (2, 2), activation=af, strides=1, padding="same")(x)

    # encoded = x

    # x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(encoded)
    # x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # # x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.UpSampling2D((5, 5))(x)

    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((first_down_sampling, first_down_sampling))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def model_design_Unet3(input_side, input2=None, n_channels=1, z_size=None, flow_label=None, n_flows=None, first_down_sampling=5, af='relu'):
    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img)
    x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((first_down_sampling, first_down_sampling))(x)

    x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(1, (2, 2), activation=af, strides=1, padding="same")(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((first_down_sampling, first_down_sampling))(x)

    decoded = layers.Conv2DTranspose(1, (1, 1), activation='sigmoid', padding="same")(x)

    autoencoder_model = keras.Model(input_img, decoded)

    return autoencoder_model


def basic_unet(input_side, input2=None, n_channels=1, z_size=None, flow_label=None, n_flows=None, first_down_sampling=2, af='relu'):
    inputs = Input((input_side, input_side, n_channels))
    conv1 = Conv2D(16, (3, 3), activation=af, padding='same')(inputs)
    # conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(16, (3, 3), activation=af, padding='same')(conv1)
    # conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(first_down_sampling, first_down_sampling))(conv1)

    conv2 = Conv2D(32, (3, 3), activation=af, padding='same')(pool1)
    # conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(32, (3, 3), activation=af, padding='same')(conv2)
    # conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation=af, padding='same')(pool2)
    # conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(64, (3, 3), activation=af, padding='same')(conv3)
    # conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation=af, padding='same')(pool3)
    # conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(128, (3, 3), activation=af, padding='same')(conv4)
    # conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(5, 5))(conv4)

    conv5 = Conv2D(1, (4, 4), activation=af, padding='same')(pool4)
    # conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(1, (4, 4), activation=af, padding='same')(conv5)
    # conv5 = BatchNormalization()(conv5)

    up6 = concatenate([UpSampling2D((5, 5))(conv5), conv4], axis=3)
    conv6 = Conv2D(128, (3, 3), activation=af, padding='same')(up6)
    # conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(128, (3, 3), activation=af, padding='same')(conv6)
    # conv6 = BatchNormalization()(conv6)

    up7 = concatenate([UpSampling2D((2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(64, (3, 3), activation=af, padding='same')(up7)
    # conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(64, (3, 3), activation=af, padding='same')(conv7)
    # conv7 = BatchNormalization()(conv7)

    up8 = concatenate([UpSampling2D((2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(32, (3, 3), activation=af, padding='same')(up8)
    # conv8 = BatchNormalization()(conv8)
    conv8 = Conv2D(32, (3, 3), activation=af, padding='same')(conv8)
    # conv8 = BatchNormalization()(conv8)

    up9 = concatenate([UpSampling2D((first_down_sampling, first_down_sampling))(conv8), conv1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation=af, padding='same')(up9)
    # conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(16, (3, 3), activation=af, padding='same')(conv9)
    # conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model


def model_design_Unet_NF(input_side, input2=None, n_channels=1, z_size=625, flow_label=None, n_flows=5, first_down_sampling=5, af='relu'):
    if flow_label is None:
        flow_func = None
    elif flow_label == 'planar':
        flow_func = flow.Flow.uniform(n_flows, lambda i: Planar())
    elif flow_label == 'sylvester':
        flow_func = flow.Flow.uniform(n_flows, lambda i: TriangularSylvester(flip_z=i % 2 != 0))

    input_img = keras.Input(shape=(input_side, input_side, n_channels))

    x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(input_img)
    x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((first_down_sampling, first_down_sampling))(x)

    x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2D(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

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

    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(h_k)
    x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(64, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)

    x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    # x = layers.Conv2DTranspose(32, (3, 3), activation=af, strides=1, padding="same")(x)
    x = layers.UpSampling2D((first_down_sampling, first_down_sampling))(x)

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


def vae_encoder(input_side, input2=None, n_channels=1, z_size=625, flow_label=None, n_flows=5, first_down_sampling=2, af='relu'):
    encoder_inputs = keras.Input(shape=(input_side, input_side, n_channels))

    bl1_conv1 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(encoder_inputs)
    bl1_conv2 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(bl1_conv1)

    bl1_conv3 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(encoder_inputs)
    bl1_add = Add()([bl1_conv3, bl1_conv2])
    bl1_pl = layers.MaxPooling2D((first_down_sampling, first_down_sampling))(bl1_add)

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
    x = layers.Dense(z_size, activation="relu")(x)
    z_mean = layers.Dense(z_size, name="z_mean")(x)
    z_log_var = layers.Dense(z_size, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    return encoder


def vae_encoder2(input_side, input2=None, n_channels=1, z_size=625, flow_label=None, n_flows=5, first_down_sampling=5, af='relu'):
    encoder_inputs = keras.Input(shape=(input_side, input_side, n_channels))

    bl1_conv1 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(encoder_inputs)
    # bl1_conv2 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(bl1_conv1)

    bl1_conv3 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(encoder_inputs)
    bl1_add = Add()([bl1_conv3, bl1_conv1])
    bl1_pl = layers.MaxPooling2D((first_down_sampling, first_down_sampling))(bl1_add)

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

    encoded = layers.Conv2D(1, (2, 2), activation=af, strides=1, padding="same")(bl3_pl)

    x = layers.Flatten()(encoded)
    x = layers.Dense(z_size, activation="relu")(x)
    z_mean = layers.Dense(z_size, name="z_mean")(x)
    z_log_var = layers.Dense(z_size, name="z_log_var")(x)
    # z = Sampling()([z_mean, z_log_var])
    outputs = [z_mean, z_log_var]
    if flow_label is not None:
        z_shape = tf.TensorShape((None, z_size))
        params = layers.Dense(flow_label.param_count(z_shape), activation='linear')(x)
        outputs += [params]
        flow_label.initialize(z_shape)

    flow_layer = FlowLayer(flow_label, min_beta=1.0E-3)
    zs, ldj, kld = flow_layer(outputs)
    z_k = zs[-1]

    encoder = keras.Model(encoder_inputs, [z_k, z_mean, z_log_var, ldj], name="encoder")
    return encoder


def vae_encoder_3params(input_side, input2=3, n_channels=1, z_size=625, flow_label=None, n_flows=5, first_down_sampling=2, af='relu'):
    encoder_inputs1 = keras.Input(shape=(input_side, input_side, n_channels))
    encoder_inputs2 = keras.Input(shape=(input2,))

    dense_1 = layers.Dense(input_side, activation="relu")(encoder_inputs2)
    dense_2 = layers.Dense(input_side ** 2, activation="relu")(dense_1)
    input_2 = layers.Reshape((input_side, input_side, 1))(dense_2)

    bl1_conv1 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(encoder_inputs1)
    bl1_conv2 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(bl1_conv1)

    bl1_conv3 = layers.Conv2D(32, (3, 3), activation=af, strides=1, padding="same")(encoder_inputs1)
    bl1_add = Add()([bl1_conv3, bl1_conv2, input_2])
    bl1_pl = layers.MaxPooling2D((first_down_sampling, first_down_sampling))(bl1_add)

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
    x = layers.Dense(z_size, activation="relu")(x)
    z_mean = layers.Dense(z_size, name="z_mean")(x)
    z_log_var = layers.Dense(z_size, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(inputs=[encoder_inputs1, encoder_inputs2],
                          outputs=[z_mean, z_log_var, z], name="encoder")
    return encoder


def vae_decoder(input_side=None, input2=None, n_channels=1, z_size=625, flow_label=None, n_flows=5, first_down_sampling=5, af='relu'):
    latent_inputs = keras.Input(shape=(z_size,))
    z_dim = np.sqrt(z_size)
    x = layers.Dense(z_dim ** 2, activation="relu")(latent_inputs)
    x = layers.Reshape((z_dim, z_dim, 1))(x)

    bl5_conv1 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    bl5_conv2 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl5_conv1)

    bl5_conv3 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    bl5_add = Add()([bl5_conv3, bl5_conv2])
    bl5_pl = layers.UpSampling2D((first_down_sampling, first_down_sampling))(bl5_add)

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


def vae_decoder2(input_side=None, input2=None, n_channels=1, z_size=625, flow_label=None, n_flows=5, first_down_sampling=2, af='relu'):
    latent_inputs = keras.Input(shape=(z_size,))

    x = layers.Dense(z_size, activation="relu")(latent_inputs)
    x = layers.Dense(2500, activation="relu")(x)
    x = layers.Reshape((50, 50, 1))(x)


    bl6_conv1 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    # bl6_conv2 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(bl6_conv1)

    bl6_conv3 = layers.Conv2DTranspose(128, (3, 3), activation=af, strides=1, padding="same")(x)
    bl6_add = Add()([bl6_conv3, bl6_conv1])
    bl6_pl = layers.UpSampling2D((first_down_sampling, first_down_sampling))(bl6_add)

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






def kgs_to_bt(input_shape, input2_shape=None):
    input_ = keras.Input(shape=input_shape)
    x1 = Dense(3, activation='relu')(input_)
    # x2 = Dense(25, activation=  'relu')(x1)
    # x3 = Dense(1024, activation=  'relu')(x1)
    # x4 = Dense(25 * 25 * 128, activation='relu')(x1)
    # x4 = layers.BatchNormalization(synchronized=True)(x4)
    # x5 = Reshape((5, 5, 128))(x4)
    x6 = Dense(50 * 50 * 32 * 32, activation='relu')(x1)
    x66 = Reshape((50, 50, 32 * 32))(x6)
    # x7 = UpSampling2D((5, 5))(x5)
    # x77 = Add()([x66, x7])
    x8 = Conv2DTranspose(128, (2, 2), activation='relu', strides=1, padding="same")(x66)
    x9 = Conv2DTranspose(64, (2, 2), activation='relu', strides=1, padding="same")(x8)
    x10 = Conv2DTranspose(32, (2, 2), activation='relu', strides=1, padding="same")(x9)
    # x10 = layers.BatchNormalization(synchronized=True)(x10)
    x11 = Conv2DTranspose(32, (2, 2), activation='relu', strides=1, padding="same")(x66)
    # x11 = layers.BatchNormalization(synchronized=True)(x11)
    x12 = Add()([x11, x10])
    x13 = Conv2DTranspose(1, (2, 2), activation='sigmoid', strides=1, padding="same")(x12)
    # return x13
    return keras.Model(input_, x13)


def kgs_lens_pos_to_bt(input1_shape, input2_shape):
    input1 = keras.Input(shape=(input1_shape,))
    input2 = keras.Input(shape=(input2_shape, input2_shape, 1))
    x1 = Dense(3, activation='relu')(input1)
    x5 = Reshape((400,))(input2)
    x2 = Dense(50 * 50 * 128, activation='relu')(x5)
    x7 = Reshape((50, 50, 128))(x2)
    x6 = Dense(50 * 50 * 128, activation='relu')(x1)
    x66 = Reshape((50, 50, 128))(x6)
    x8 = Conv2DTranspose(128, (2, 2), activation='relu', strides=1, padding="same")(x66)
    x9 = Conv2DTranspose(64, (2, 2), activation='relu', strides=1, padding="same")(x8)
    x10 = Conv2DTranspose(64, (2, 2), activation='relu', strides=1, padding="same")(x7)
    x12 = Concatenate()([x9, x10])
    x13 = Conv2DTranspose(1, (2, 2), activation='sigmoid', strides=1, padding="same")(x12)
    return keras.Model([input1, input2], x13)


def bt_to_kgs(input_shape, input2_shape=None):
    input_img = keras.Input(shape=input_shape)

    x1 = Conv2D(128, (2, 2), activation='relu', strides=1, padding="same")(input_img)
    x11 = MaxPooling2D((5, 5))(x1)
    x2 = Conv2D(32, (2, 2), activation='relu', strides=1, padding="same")(input_img)
    x3 = Conv2D(64, (2, 2), activation='relu', strides=1, padding="same")(x2)
    x4 = Conv2D(128, (2, 2), activation='relu', strides=1, padding="same")(x3)
    x5 = MaxPooling2D((5, 5))(x4)
    x55 = Add()([x11, x5])
    x6 = Flatten()(x55)
    x7 = Dense(1024, activation='relu')(x6)

    x10 = Dense(3, activation='sigmoid')(x7)
    return keras.Model([input_img], x10)
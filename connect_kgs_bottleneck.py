import argparse
import json
import os
import pickle as pkl
import sys

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Conv2DTranspose, Add, Reshape, Flatten, Conv2D, \
    MaxPooling2D, Concatenate

from my_classes import tweedie_loss_func, lc_loss_func, custom_loss_func

physical_devices = tf.config.experimental.list_physical_devices('GPU')
# physical_devices = tf.config.experimental.list_physical_devices('CPU')
print("physical_devices-------------", len(physical_devices))
# tf.debugging.set_log_device_placement(True)

if len(physical_devices) == 0:
    print('No GPUs are accessible!')
    sys.exit()


# tf.config.experimental.set_memory_growth(physical_devices[0], physical_devices[0], True)
# mirrored_strategy = tf.distribute.MirroredStrategy()

# Parameters
def parse_options():
    """Function to handle options speficied at command line
    """
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('-dim', action='store', default=10000,
                        help='What is the dimension of the maps.')
    parser.add_argument('-directory', action='store',
                        default='./../../../fred/oz108/GERLUMPH_project/DATABASES/gerlumph_db/',
                        help='Specify the directory where the maps are stored on the supercomputer.')
    parser.add_argument('-directory_bulk', action='store',
                        default='./../../../fred/oz108/skhakpas/',
                        help='Specify the directory where the maps are stored on the supercomputer.')
    parser.add_argument('-batch_size', action='store', default=32, help='Batch size for the autoencoder.')
    parser.add_argument('-n_epochs', action='store', default=50, help='Number of epochs.')
    parser.add_argument('-n_channels', action='store', default=1, help='Number of channels.')
    parser.add_argument('-lr_rate', action='store', default=0.00001, help='Learning rate.')
    parser.add_argument('-res_scale', action='store', default=10,
                        help='With what scale should the original maps resoultion be reduced.')
    parser.add_argument('-crop_scale', action='store', default=1,
                        help='With what scale should the original maps be cropped and multipled by 4.')
    parser.add_argument('-date', action='store',
                        default='',
                        help='Specify the directory where the results should be stored.')
    parser.add_argument('-list_IDs_directory', action='store',
                        default='./../data/GD1_ids_list.txt',
                        help='Specify the directory where the list of map IDs is stored.')
    parser.add_argument('-conversion_file_directory', action='store',
                        default='./../data/all_maps_meta_kgs.csv',
                        help='Specify the directory where the conversion parameters.')
    parser.add_argument('-lens_pos_directory', action='store',
                        default='./../data/all_maps_lens_pos.pkl',
                        help='Specify the directory where the saved lens posistions are.')
    parser.add_argument('-test_set_size', action='store',
                        default=10,
                        help='Size of the test set.'),
    parser.add_argument('-sample_size', action='store',
                        default=3828,
                        help='Size of the sample data set.'
                        )
    parser.add_argument('-saved_model', action='store',
                        default=False,
                        help='Whether or not you want to train an already-saved model.'
                        )
    parser.add_argument('-model_design', action='store',
                        default='kgs_to_bt',
                        help='Which model design you want.'
                        )
    parser.add_argument('-saved_bottleneck_path', action='store',
                        default='./../../../fred/oz108/skhakpas/results/23-10-14-11-00-03/model_10000_8_1e-05',
                        help='Whether or not you want to train an already-saved model.'
                        )
    parser.add_argument('-loss_function', action='store',
                        default='binary_crossentropy',
                        help='What is the loss function?'
                        )
    parser.add_argument('-lc_loss_function_metric', action='store',
                        default='mse',
                        help='What is the metric to calculate statistics in the lc loss function?'
                        )
    parser.add_argument('-lc_loss_function_coeff', action='store',
                        default=1,
                        help='What is the metric to calculate statistics in the lc loss function?'
                        )
    parser.add_argument('-optimizer', action='store',
                        default='adam',
                        help='What is the optimizer?'
                        )
    parser.add_argument('-flow_label', action='store',
                        default=None,
                        help='What flow model you want to use. Use this with Unet_NF model design ONLY.'
                        )
    parser.add_argument('-n_flows', action='store',
                        default=4,
                        help='Number of flows in the Unet_NF model design.'
                        )
    parser.add_argument('-bottleneck_size', action='store',
                        default=625,
                        help='Size of the bottleneck when NF is added. ONLY with Unet_NF model design.'
                        )
    parser.add_argument('-test_set_selection', action='store',
                        default='random',
                        help='Do you want to choose your test set randomly or are you looking for a chosen set of IDs')
    parser.add_argument('-early_callback', action='store',
                        default='False',
                        help='Do you want to add early calback when fitting the model?')
    parser.add_argument('-early_callback_type', action='store',
                        default=1,
                        help='What type of early callback you want?')
    parser.add_argument('-add_params', action='store',
                        default=False,
                        help='Do you want to add the 3 map params as new channels?')
    parser.add_argument('-add_lens_pos', action='store',
                        default=False,
                        help='Do you want to add the lens positions?')
    parser.add_argument('-add_map_units', action='store',
                        default=False,
                        help='Do you want to add the map units in RE? That is more useful when lens positions are '
                             'included')
    parser.add_argument('-model_input_format', action='store',
                        default='xx',
                        help='Does the model get x as data and y as targets? xy, other options are xx, x')
    parser.add_argument('-ngpu', action='store',
                        default=1,
                        help='Number of GPUs you are selecting.')
    parser.add_argument('-training_plans', action='store',
                        default='kgs_to_bt',
                        help='If you want to change the optimizer during the training.')
    parser.add_argument('-trained_model', action='store',
                        default='model_batch128_lr0.01_loss_custom',
                        help='If you want to change the optimizer during the training.')
    parser.add_argument('-saved_date', action='store',
                        default='23-12-15-00-23-29',
                        help='If you want to change the optimizer during the training.')
    # Parses through the arguments and saves them within the keyword args
    arguments = parser.parse_args()
    return arguments


args = parse_options()
model_design_key = args.model_design
loss_function_key = args.loss_function
optimizer_key = args.optimizer
training_plan = args.training_plans
lens_pos_path = args.lens_pos_directory

flow_label = args.flow_label
n_flows = int(args.n_flows)
z_size = args.bottleneck_size
n_test_set = int(args.test_set_size)
test_set_selection = args.test_set_selection
early_callback = args.early_callback
early_callback_type = args.early_callback_type
add_params = args.add_params
model_input_format = args.model_input_format
trained_model = args.trained_model
saved_date = args.saved_date
ngpu = int(args.ngpu)
directory_bulk = args.directory_bulk

saved_model = args.saved_model

if saved_model:
    date = saved_date
else:
    date = args.date
    os.system("mkdir ./../../../fred/oz108/skhakpas/results/kgs_bt/" + str(date))


saved_bottleneck_path = args.saved_bottleneck_path
lr_rate = float(args.lr_rate)
crop_scale = float(args.crop_scale)
n_epochs = int(args.n_epochs)
dim = int(args.dim)
lc_loss_function_metric = args.lc_loss_function_metric
lc_loss_function_coeff = int(args.lc_loss_function_coeff)
sample_size = int(args.sample_size)
include_lens_pos = args.add_lens_pos
include_map_units = args.add_map_units
output_dir = './../../../fred/oz108/skhakpas/results/kgs_bt/' + date + '/'


def kgs_to_bt(input_shape):
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


def bt_to_kgs(input_shape):
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


def NormalizeData(data, max_data, min_data):
    data_max = max_data
    data_min = min_data
    data_new = (data - data_min) / (data_max - data_min)
    return data_new




print('Setting up the network parameters...')
model_designs = {'kgs_to_bt': kgs_to_bt,
                 'bt_to_kgs': bt_to_kgs,
                 'kgs_lens_pos_bt': kgs_lens_pos_to_bt}
loss_functions = {'binary_crossentropy': keras.losses.BinaryCrossentropy(),
                  'poisson': keras.losses.Poisson(),
                  'tweedie': tweedie_loss_func(0.5),
                  'lc_loss': lc_loss_func(lc_loss_function_metric, lc_loss_function_coeff),
                  'huber': tf.keras.losses.Huber(delta=1),
                  'custom': custom_loss_func()}
optimizers = {'adam': keras.optimizers.Adam,
              'sgd': keras.optimizers.SGD,
              'adamax': keras.optimizers.Adamax,
              'adadelta': keras.optimizers.Adadelta,
              'rmsprop': keras.optimizers.RMSprop}

params = {'dim': dim,
          'batch_size': int(args.batch_size),
          'n_channels': int(args.n_channels),
          'res_scale': int(args.res_scale),
          'crop_scale': crop_scale,
          'path': args.directory,
          'shuffle': True,
          'conv_const': args.conversion_file_directory,
          'output_format': model_input_format,
          'include_lens_pos': include_lens_pos,
          'include_map_units': include_map_units}

print('Reading in data')
all_params_ = pd.read_csv(params['conv_const'])
bottleneck = np.load(saved_bottleneck_path.split("model")[0] + 'bottleneck.npy')
IDs_ = np.load(saved_bottleneck_path.split("model")[0] + 'bottleneck_labels.npy')

all_lens_pos = pkl.load(open(lens_pos_path, "rb"))

lens_pos = np.zeros((len(IDs_), 20, 20, 1))
for i, key in enumerate(IDs_):
    if all_lens_pos[key].shape[0] == 0:
        pass
    elif all_lens_pos[key].shape[0] != 0 and len(all_lens_pos[key].shape) == 1:
        lens_pos[i, :, :, 0] = np.histogram2d(np.asarray([all_lens_pos[key]])[:, 0],
                                              np.asarray([all_lens_pos[key]])[:, 1],
                                              bins=20)[0]
    else:
        lens_pos[i, :, :, 0] = np.histogram2d(all_lens_pos[key][:, 0],
                                              all_lens_pos[key][:, 1],
                                              bins=20)[0]

all_params = all_params_[['ID', 'k', 'g', 's']].values
x_params = np.asarray([all_params[:, 1:][all_params[:, 0] == ID][0] for ID in IDs_])

# x_params[:, 0] = NormalizeData(x_params[:, 0], np.max(x_params[:, 0]), np.min(x_params[:, 0]))
# x_params[:, 1] = NormalizeData(x_params[:, 1], np.max(x_params[:, 1]), np.min(x_params[:, 1]))

bottleneck = bottleneck / np.max(bottleneck)
n_lc = len(x_params)
bt_size = bottleneck.shape[1]
bottleneck_10 = np.zeros((n_lc * 10, bt_size, bt_size, 1))
x_params_10 = np.zeros((n_lc * 10, 3))
IDs_10 = np.zeros((n_lc * 10))
lens_pos_10 = np.zeros((n_lc * 10, 20, 20, 1))

for i in range(n_lc):
    for k in range(10):
        bottleneck_10[10 * i + k] = bottleneck[i]
        x_params_10[10 * i + k] = x_params[i]
        IDs_10[10 * i + k] = IDs_[i]
        lens_pos_10[10 * i + k] = lens_pos[i]

print('Defining train and test datasets...')
n = len(x_params_10)
train_n = int(0.9 * n)
test_n = int(0.1 * n)

shuffler = np.random.permutation(n)
bottleneck_10 = bottleneck_10[shuffler]
x_params_10 = x_params_10[shuffler]
lens_pos_10 = lens_pos_10[shuffler]
IDs_10 = IDs_10[shuffler]

y_train_ = x_params_10[:train_n]
y_train_2 = lens_pos_10[:train_n]
x_train = bottleneck_10[:train_n]
y_test_ = x_params_10[train_n:]
y_test_2 = lens_pos_10[train_n:]
x_test = bottleneck_10[train_n:]

ids_train = IDs_10[:train_n]
ids_test = IDs_10[train_n:]

print('Max of the training set: ', np.max(y_train_2))
y_train_2 = y_train_2 / np.max(y_train_2)
print('Max of the test set: ', np.max(y_test_2))
y_test_2 = y_test_2 / np.max(y_test_2)

y_train = np.zeros(y_train_.shape)
y_test = np.zeros(y_test_.shape)

print('Min and Max for kappa: ', min(y_train_[:, 0]), max(y_train_[:, 0]))
print('Min and Max for gamma: ', min(y_train_[:, 1]), max(y_train_[:, 1]))

y_train[:, 0] = NormalizeData(y_train_[:, 0], np.max(y_train_[:, 0]), np.min(y_train_[:, 0]))
y_train[:, 1] = NormalizeData(y_train_[:, 1], np.max(y_train_[:, 1]), np.min(y_train_[:, 1]))
y_train[:, 2] = y_train_[:, 2]
y_test[:, 0] = NormalizeData(y_test_[:, 0], np.max(y_test_[:, 0]), np.min(y_test_[:, 0]))
y_test[:, 1] = NormalizeData(y_test_[:, 1], np.max(y_test_[:, 1]), np.min(y_test_[:, 1]))
y_test[:, 2] = y_test_[:, 2]

print('Model params are:')
params['epochs'] = n_epochs
params['lr_rate'] = lr_rate
params['date'] = date
params['saved_model'] = saved_model
params['sample_size'] = sample_size
params['loss_function'] = loss_function_key
params['loss_function_metric'] = lc_loss_function_metric
params['optimizer'] = optimizer_key
params['model_design'] = model_design_key
params['training_mode'] = training_plan

print('Defining the model...')

if training_plan == 'kgs_lens_pos_bt':
    model = model_designs[model_design_key](3, 20)
    data_training = [y_train, y_train_2]
    data_target = x_train
    data_testing = [y_test, y_test_2]
    data_target_test = x_test
elif training_plan == 'bt_to_kgs':
    model = model_designs[model_design_key]((bt_size, bt_size, 1))
    data_training = x_train
    data_target = y_train
    data_testing = x_test
    data_target_test = y_test
elif training_plan == 'retrain':
    try:
        model = keras.models.load_model(output_dir + trained_model)
    except:
        model = keras.models.load_model(output_dir + trained_model,
                                        custom_objects={'custom_loglikelihood': custom_loss_func()})
    if model_design_key == 'kgs_to_bt':
        data_training = y_train
        data_target = x_train
        data_testing = y_test
        data_target_test = x_test
    elif model_design_key == 'kgs_lens_pos_bt':
        data_training = [y_train, y_train_2]
        data_target = x_train
        data_testing = [y_test, y_test_2]
        data_target_test = x_test
    elif model_design_key == 'bt_to_kgs':
        data_training = x_train
        data_target = y_train
        data_testing = x_test
        data_target_test = y_test
else:
    #training_plan == 'kgs_to_bt'
    model = model_designs[model_design_key]((3,))
    data_training = y_train
    data_target = x_train
    data_testing = y_test
    data_target_test = x_test



print('Arguments are:', params)
print('Loss function: ', loss_function_key)

print(json.dumps(params, indent=len(params.keys())))

print('Compiling the model...')

model.compile(optimizer=optimizers[optimizer_key](learning_rate=lr_rate),
                  loss=loss_functions[loss_function_key])

es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                   min_delta=0.0000001,
                                   patience=10)
print('Fitting the model...')
history = model.fit(data_training, data_target,
                    epochs=params['epochs'],
                    batch_size=params['batch_size'],
                    validation_split=0.2,
                    callbacks=[es])


print('Saving the model...')
model.save(output_dir + 'model_batch{}_lr{}_loss_{}_V3'.format(params['batch_size'], lr_rate, loss_function_key))
print('Saving the model history...')
with open(output_dir + 'model_history_batch{}_lr{}_loss_{}_V3.pkl'.format(params['batch_size'], lr_rate, loss_function_key),
          'wb') as file_pi:
    pkl.dump(history.history, file_pi)

print('Predicting with the model...')
bt_predictions = model.predict(data_testing)

print('Saving the output...')
np.save(output_dir + 'input_predictions_V3', np.asarray(data_testing))
np.save(output_dir + 'output_predictions_V3', bt_predictions)
np.save(output_dir + 'output_targets_V3', data_target_test)
np.save(output_dir + 'ID_predictions_V3', ids_test)

print('Saving examples...')
for i in range(10):
    id_ = ids_test[i]

    fig, axs = plt.subplots(1, 3, figsize=(20, 20))
    axs[0].imshow(y_test[i].reshape((bt_size, bt_size)))
    axs[1].imshow(bt_predictions[i].reshape((bt_size, bt_size)))
    axs[2].hist(y_test[i].flatten(), bins=100)
    axs[2].hist(bt_predictions[i].flatten(), bins=100)

    axs[0].set_title('Autoencoder Bottleneck')
    axs[1].set_title('Generated Bottleneck from Parameters')
    axs[2].set_title('Histograms')

    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)

    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)

    plt.title(id_)
    fig.savefig(output_dir + 'exp_bottleneck_' +
                '_' + str(id_) + '.png')

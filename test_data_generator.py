import numpy as np
import argparse
import keras
import sys, os
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import pickle as pkl
import json
import math

# date = time.strftime("%y-%m-%d-%H-%M-%S")


import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
# physical_devices = tf.config.experimental.list_physical_devices('CPU')
print("physical_devices-------------", len(physical_devices))


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
    parser.add_argument('-batch_size', action='store', default=8, help='Batch size for the autoencoder.')
    parser.add_argument('-n_epochs', action='store', default=50, help='Number of epochs.')
    parser.add_argument('-lr_rate', action='store', default=0.00001, help='Learning rate.')
    parser.add_argument('-res_scale', action='store', default=10,
                        help='With what scale should the original maps resoultion be reduced.')
    parser.add_argument('-date', action='store',
                        default='',
                        help='Specify the directory where the results should be stored.')
    parser.add_argument('-list_IDs_directory', action='store',
                        default='./../data/ID_maps_selected_kappa_equal_gamma.dat',
                        help='Specify the directory where the list of map IDs is stored.')
    parser.add_argument('-conversion_file_directory', action='store',
                        default='./../data/maps_selected_kappa_equal_gamma.csv',
                        help='Specify the directory where the conversion parameters.')
    parser.add_argument('-test_set_size', action='store',
                        default=100,
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
                        default='2l',
                        help='Which model design you want.'
                        )
    parser.add_argument('-saved_model_path', action='store',
                        default='./../results/23-03-09-01-39-47/model_10000_8_1e-05',
                        help='Whether or not you want to train an already-saved model.'
                        )
    parser.add_argument('-loss_function', action='store',
                        default='binary_crossentropy',
                        help='What is the loss function?'
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
    parser.add_argument('-convolve', action='store',
                        default='False',
                        help='Do you want your target maps to be convolved maps or not.')
    parser.add_argument('-rsrc', action='store',
                        default=1,
                        help='Size of the source in units of lens Einstein radius.')

    # Parses through the arguments and saves them within the keyword args
    arguments = parser.parse_args()
    return arguments




args = parse_options()
model_design_key = args.model_design
loss_function_key = args.loss_function
optimizer_key = args.optimizer

flow_label = args.flow_label
n_flows = int(args.n_flows)
z_size = args.bottleneck_size
n_test_set = int(args.test_set_size)
test_set_selection = args.test_set_selection

date = args.date
os.system("mkdir ./../results/" + str(date))

saved_model = args.saved_model
saved_model_path = args.saved_model_path
lr_rate = float(args.lr_rate)
n_epochs = int(args.n_epochs)
dim = int(args.dim)
sample_size = int(args.sample_size)
convolve = args.convolve
rsrc = float(args.rsrc)
output_dir = './../results/' + args.date + '/'

if convolve:
    from my_classes_convolved import DataGenerator
else:
    from my_classes import DataGenerator

print('Setting up the parameters of the Data Generator ...')

params = {'dim': dim,
          'batch_size': int(args.batch_size),
          'n_channels': 1,
          'res_scale': int(args.res_scale),
          'path': args.directory,
          'shuffle': True,
          'conv_const': args.conversion_file_directory,
          'convolve': convolve,
          'rsrc': rsrc}
keys_old = params.keys()

# Datasets
partition = {}
ls_maps = np.loadtxt(args.list_IDs_directory, usecols=(0,), dtype=int)

random.seed(10)
shuffler = np.random.permutation(len(ls_maps))

shuffler = random.sample(list(shuffler), int(sample_size))
ls_maps = ls_maps[shuffler]

n_maps = len(ls_maps)
indx1 = np.arange(int(0.8 * n_maps), dtype=int)
indx2 = np.arange(int(0.8 * n_maps), int(0.9 * n_maps))
indx3 = np.arange(int(0.9 * n_maps), n_maps)


partition['train'] = ls_maps[indx1]
partition['validation'] = ls_maps[indx2]
partition['test'] = ls_maps[indx3]
f = open(output_dir + 'sample_set_indexes.pkl', 'wb')
pkl.dump(partition, f)

# Generators
training_generator = DataGenerator(partition['train'], **params)
validation_generator = DataGenerator(partition['validation'], **params)

print('Setting up the test set...')

if test_set_selection == 'random':
    test_set_index = np.asarray(random.sample(list(partition['test']), n_test_set))
elif test_set_selection == 'all_test':
    test_set_index = partition['test']
elif test_set_selection == 'all_train':
    test_set_index = partition['train']
elif test_set_selection == 'all_data':
    test_set_index = np.concatenate((partition['train'], partition['validation'], partition['test']))
    n_test_set = math.trunc(len(test_set_index) / params['batch_size']) * params['batch_size']
elif test_set_selection == 'sorted':
    test_set_index = np.sort(partition['test'])[:n_test_set]
else:
    test_set_index = np.loadtxt(test_set_selection)

test_generator = DataGenerator(test_set_index, **params)


print('Model params are:')
params['epochs'] = n_epochs
params['lr_rate'] = lr_rate
params['date'] = date
params['saved_model'] = saved_model
params['sample_size'] = sample_size

x_test = test_generator.map_reader(partition['test'], output='map')
generator_indexes = test_generator.get_generator_indexes()
x_test = x_test[generator_indexes]

x_test_norm = test_generator.map_reader(partition['test'], output='map_norm')
x_test_norm = x_test_norm[generator_indexes]

if convolve:
    x_test_conv = test_generator.map_reader(partition['test'], output='map_conv')
    x_test_conv = x_test_conv[generator_indexes]

    x_test_conv_norm = test_generator.map_reader(partition['test'], output='map_conv_norm')
    x_test_conv_norm = x_test_conv_norm[generator_indexes]


i = 0
print('Printing the Min, Max, Mean, and Median of the first map:')
map_mag = x_test[i].flatten()
print('Map in units of magnification: ', np.min(map_mag), np.max(map_mag), np.mean(map_mag), np.median(map_mag))

map_mag = x_test_norm[i].flatten()
print('Normalized map in units of magnification: ', np.min(map_mag), np.max(map_mag), np.mean(map_mag), np.median(map_mag))

if convolve:
    map_mag = x_test_conv[i].flatten()
    print('Convolved map in units of magnification: ', np.min(map_mag), np.max(map_mag), np.mean(map_mag), np.median(map_mag))

    map_mag = x_test_conv_norm[i].flatten()
    print('Normalized convolved map in units of magnification: ', np.min(map_mag), np.max(map_mag), np.mean(map_mag), np.median(map_mag))
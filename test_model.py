from my_classes import DataGenerator, model_design_2l, display_model, model_fit2, model_compile, \
    compareinout2D, fig_loss, tweedie_loss_func, basic_unet, model_design_Unet2, model_design_Unet3, vae_NF, \
    lc_loss_func
import numpy as np
import argparse
import keras
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import os, sys
import pickle as pkl
import pandas as pd
import math

# date = time.strftime("%y-%m-%d")
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
# physical_devices = tf.config.experimental.list_physical_devices('CPU')
print("physical_devices-------------", len(physical_devices))

if len(physical_devices) == 0:
    print('No GPUs are accessible!')
    sys.exit()


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
    parser.add_argument('-batch_size', action='store', default=16, help='Batch size for the autoencoder.')
    parser.add_argument('-n_epochs', action='store', default=20, help='Number of epochs.')
    parser.add_argument('-n_channels', action='store', default=1, help='Number of channels.')
    parser.add_argument('-lr_rate', action='store', default=0.00001, help='Learning rate.')
    parser.add_argument('-res_scale', action='store', default=10,
                        help='With what scale should the original maps resoultion be reduced.')
    parser.add_argument('-date', action='store',
                        default='',
                        help='Specify the directory where the results should be stored.')
    parser.add_argument('-list_IDs_directory', action='store',
                        default='./../data/GD1_ids_list.txt',
                        help='Specify the directory where the list of map IDs is stored.')
    parser.add_argument('-model_direc', action='store',
                        default='./../results/23-03-01-07-57-08/model_10000_8_1e-05',
                        help='Specify the directory where the saved AI model is stored.')
    parser.add_argument('-partition_direc', action='store',
                        default=None,
                        help='Specify the directory where the partition set is saved.')
    parser.add_argument('-output_dir', action='store',
                        default=None,
                        help='Specify the directory where the saved AI model is stored.'
                             'Make sure you put / at the end of it!')
    parser.add_argument('-test_set_size', action='store',
                        default=10,
                        help='Size of the test set.')
    parser.add_argument('-predict', action='store',
                        default=False,
                        help='Do you want the code to make more predictions of your saved model?')
    parser.add_argument('-save_bottleneck', action='store',
                        default=False,
                        help='Do you want the code to save outputs of individual layers?')
    parser.add_argument('-save_test_set', action='store',
                        default=False,
                        help='Do you want the code to save outputs of individual layers?')
    parser.add_argument('-model_design', action='store',
                        default='2l',
                        help='Which model design you want.'
                        )
    parser.add_argument('-conversion_file_directory', action='store',
                        default='./../data/all_maps_meta_kgs.csv',
                        help='Specify the directory where the conversion parameters.')
    parser.add_argument('-print_loss', action='store',
                        default=False,
                        help='Do you want the code to output the loss function plot?')
    parser.add_argument('-test_set_selection', action='store',
                        default='random',
                        help='Do you want to choose your test set randomly or are you looking for a chosen set of IDs')
    parser.add_argument('-loss_function', action='store',
                        default='binary_crossentropy',
                        help='Whether or not you want to train an already-saved model.'
                        )
    parser.add_argument('-lc_loss_function_metric', action='store',
                        default='mse',
                        help='What is the metric to calculate statistics in the lc loss function?'
                        )

    # Parses through the arguments and saves them within the keyword args
    arguments = parser.parse_args()
    return arguments


print('Reading in the arguments...')
args = parse_options()
n_test_set = int(args.test_set_size)
model_direc = args.model_direc
predict = args.predict
conversion_file_directory = args.conversion_file_directory
save_bottleneck = args.save_bottleneck
save_test_set = args.save_test_set
print_loss = args.print_loss
loss_func = args.loss_function
lc_loss_function_metric = args.lc_loss_function_metric
test_set_selection = args.test_set_selection
params = {'dim': int(args.dim),
          'batch_size': int(args.batch_size),
          'n_channels': int(args.n_channels),
          'res_scale': int(args.res_scale),
          'path': args.directory,
          'shuffle': True}
bottleneck_layer_name = {'2l': 'max_pooling2d_1',
                         '3l': 'conv2d_3',
                         'Unet': 'conv2d_8',
                         'Unet2': model_design_Unet2,
                         'Unet3': model_design_Unet3,
                         'basic_unet': basic_unet,
                         'Unet_NF': vae_NF}
loss_functions = {'binary_crossentropy': keras.losses.BinaryCrossentropy(from_logits=True),
                  'poisson': keras.losses.Poisson(),
                  'tweedie': tweedie_loss_func(0.5),
                  'lc_loss': lc_loss_func(lc_loss_function_metric)}
lr_rate = float(args.lr_rate)
n_epochs = int(args.n_epochs)
date = args.date
model_design = args.model_design

if args.output_dir is None:
    os.system("mkdir ./../results/" + str(date))
    output_dir = './../results/' + args.date + '/'
else:
    output_dir = args.output_dir

dim = int(params['dim'] / params['res_scale'])
# Datasets

if args.partition_direc is None:
    partition_direc = model_direc.split("model")[0] + 'sample_set_indexes.pkl'
else:
    partition_direc = args.partition_direc + 'sample_set_indexes.pkl'

print('Reading the partition file from ', partition_direc)
file_partition = open(partition_direc, 'rb')
partition = pkl.load(file_partition)

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

print('Selecting a test set based on ', test_set_selection)
print('Length of test set indexes is ', str(len(test_set_index)))
test_generator = DataGenerator(test_set_index, **params)
x_test = test_generator.map_reader(test_set_index, output='map_norm')
generator_indexes = test_generator.get_generator_indexes()
# print('initial test set IDs ',test_set_index)
x_test = x_test[generator_indexes]
# test_set_index = test_set_index[generator_indexes]

# print('The true order of the IDs: ', test_set_index)

test_set_index_argsort = np.asarray(test_set_index).argsort()
test_set_index_argsort_argsort = np.asarray(test_set_index).argsort().argsort()

# print('sorted test set IDs', test_set_index)
print('Loading in the saved model at ', model_direc)
tf.config.run_functions_eagerly(True)
if loss_func == 'lc_loss':
    autoencoder = keras.models.load_model(model_direc, custom_objects= \
        {'lc_loglikelihood': lc_loss_func(metric=lc_loss_function_metric)})
else:
    autoencoder = keras.models.load_model(model_direc, custom_objects={'loss': loss_functions[loss_func]})
autoencoder.summary()

if predict:
    print('Predicting on the test set via the saved model...')
    output_autoencoder = autoencoder.predict(test_generator)

    print('Size of the test set: ', len(x_test))
    print('Size of the test set prediction: ', len(output_autoencoder))

    print('Saving the predictions...')
    np.save(output_dir + 'x_test_predictions', output_autoencoder)

    print('Saving output plots at ', output_dir)
    for i in tqdm(range(len(output_autoencoder))):
        fig = compareinout2D(output_autoencoder[i], np.asarray(x_test)[i], dim)
        fig.savefig(output_dir + 'compare_' + str(test_set_index[i]) + '.png', bbox_inches='tight')
        print('True map (min,max,mean,median): ',
              np.min(np.asarray(x_test)[i]),
              np.max(np.asarray(x_test)[i]),
              np.mean(np.asarray(x_test)[i]),
              np.median(np.asarray(x_test)[i]))
        print('Predicted map (min,max,mean,median): ',
              np.min(output_autoencoder[i]),
              np.max(output_autoencoder[i]),
              np.mean(output_autoencoder[i]),
              np.median(output_autoencoder[i]))

#
if print_loss or save_bottleneck:
    history_direc = model_direc.split("model")[0] + 'model_history.pkl'
    file = open(history_direc, 'rb')
    autoencoder_history = pkl.load(file)

if print_loss:
    fig = fig_loss(autoencoder_history)
    print('Plotting the loss function at ', output_dir + 'loss.png')
    plt.savefig(output_dir + 'loss.png')

if save_bottleneck:
    bt_ly_name = bottleneck_layer_name[model_design]
    bottleneck_output = autoencoder.get_layer(bt_ly_name).output
    model_bottleneck = keras.models.Model(inputs=autoencoder.input, outputs=bottleneck_output)
    print('Generating the latent space representation for ' + str(len(x_test)) + ' test objects...')
    bottleneck_predictions = model_bottleneck.predict(test_generator)

    print('Saving the bottleneck and test set and labels at ', output_dir)
    np.save(output_dir + 'bottleneck', bottleneck_predictions)

if save_test_set:
    meta = pd.read_csv(conversion_file_directory)
    true_labels = meta.loc[meta['ID'].isin(test_set_index)][['ID', 'k', 'g', 's']].values

    # The order of the true_labels should be the same as the x_test and output_autoencoder
    true_labels_sorted = true_labels[true_labels[:, 0].argsort()]
    true_labels_in_test_set_index_order = true_labels_sorted[test_set_index_argsort_argsort]

    print('final true labels: ', true_labels_in_test_set_index_order)

    np.save(output_dir + 'x_test', x_test)
    np.save(output_dir + 'x_test_labels', true_labels_in_test_set_index_order)

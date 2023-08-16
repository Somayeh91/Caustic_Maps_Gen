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
from my_classes_convolved import DataGenerator, model_design_2l, display_model, model_fit2, model_compile, \
        compareinout2D, fig_loss, read_saved_model, model_design_3l, model_design_Unet, tweedie_loss_func, \
        basic_unet, model_design_Unet2, model_design_Unet3, vae_NF, model_design_Unet_resnet, model_design_Unet_resnet2,\
        lc_loss_func
# date = time.strftime("%y-%m-%d-%H-%M-%S")


import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
# physical_devices = tf.config.experimental.list_physical_devices('CPU')
print("physical_devices-------------", len(physical_devices))

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
    parser.add_argument('-lc_loss_function_metric', action='store',
                        default='mse',
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
    parser.add_argument('-convolve', action='store',
                        default='True',
                        help='Do you want your target maps to be convolved maps or not.')
    parser.add_argument('-rsrc', action='store',
                        default=1,
                        help='Size of the source in units of lens Einstein radius.')
    parser.add_argument('-early_callback', action='store',
                        default='False',
                        help='Do you want to add early calback when fitting the model?')
    parser.add_argument('-early_callback_type', action='store',
                        default=1,
                        help='What type of early callback you want?')

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
early_callback = args.early_callback
early_callback_type = args.early_callback_type
lc_loss_function_metric = args.lc_loss_function_metric

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

print('Arguments are:')
print('Loss function: ', )

print('Setting up the network parameters...')
model_designs = {'2l': model_design_2l,
                 '3l': model_design_3l,
                 'Unet': model_design_Unet,
                 'Unet2': model_design_Unet2,
                 'Unet3': model_design_Unet3,
                 'basic_unet': basic_unet,
                 'Unet_NF': vae_NF,
                 'Unet_resnet': model_design_Unet_resnet,
                 'Unet_resnet2': model_design_Unet_resnet2}
loss_functions = {'binary_crossentropy': keras.losses.BinaryCrossentropy(from_logits=True),
                  'poisson': keras.losses.Poisson(),
                  'tweedie': tweedie_loss_func(0.5),
                  'lc_loss': lc_loss_func(lc_loss_function_metric)
                  }
optimizers = {'adam': keras.optimizers.Adam,
              'sgd': keras.optimizers.SGD,
              'adamax': keras.optimizers.Adamax,
              'adadelta': keras.optimizers.Adadelta,
              'rmsprop': keras.optimizers.RMSprop}

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

if saved_model:
    partition_direc = saved_model_path.split("model")[0] + 'sample_set_indexes.pkl'
    file_partition = open(partition_direc, 'rb')
    partition = pkl.load(file_partition)

    file_params = open(saved_model_path.split('model')[0] + 'params.pkl', 'rb')
    params_saved = pkl.load(file_params)
    print('Model params are: ')
    print(json.dumps(params_saved, indent=len(params_saved.keys())))

    # Generators

    params_saved = {key: params_saved[key] for key in params.keys()}
    training_generator = DataGenerator(partition['train'], **params_saved)
    validation_generator = DataGenerator(partition['validation'], **params_saved)
    test_generator = DataGenerator(partition['test'], **params_saved)
    test_set_index = random.sample(list(partition['test']), n_test_set)
    generator_indexes = test_generator.get_generator_indexes()

    print('Loading a trained model...')

    autoencoder = read_saved_model(saved_model_path)

else:
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
    # x_test = test_generator.map_reader(test_set_index)
    # generator_indexes = test_generator.get_generator_indexes()
    # x_test = x_test[generator_indexes]

    print('Model params are:')
    params['epochs'] = n_epochs
    params['lr_rate'] = lr_rate
    params['date'] = date
    params['saved_model'] = saved_model
    params['sample_size'] = sample_size
    params['loss_function'] = loss_function_key
    params['optimizer'] = optimizer_key
    params['model_design'] = model_design_key

    print(json.dumps(params, indent=len(params.keys())))

    print('Saving the parameters of the run...')
    file = open(output_dir + 'params.pkl', 'wb')
    pkl.dump(params, file)

    # Design the model
    print('Desigining the network...')
    # with mirrored_strategy.scope():
    if model_design_key == 'Unet_NF':
        autoencoder = model_designs[model_design_key](int(dim / params['res_scale']),
                                                      learning_rate=lr_rate, flow_label=flow_label,
                                                      z_size=z_size, n_flows=n_flows,
                                                      loss=loss_functions[loss_function_key])
    else:
        autoencoder = model_designs[model_design_key](int(dim / params['res_scale']))
    display_model(autoencoder)
    model_compile(autoencoder, lr_rate, loss=loss_functions[loss_function_key],
                  optimizer_=optimizers[optimizer_key])

# Train the model
autoencoder_history = model_fit2(autoencoder, n_epochs,
                                 training_generator, validation_generator,
                                 filepath = output_dir,
                                 early_callback_ = early_callback,
                                 early_callback_type = early_callback_type)

print('Saving the model...')
autoencoder.save(output_dir + 'model_' + str(params['dim']) + '_' + str(params['batch_size']) + '_' +
                 str(lr_rate))
print('Saving the model history...')
with open(output_dir + 'model_history.pkl', 'wb') as file_pi:
    pkl.dump(autoencoder_history.history, file_pi)

print('Plotting the loss function...')
fig = fig_loss(autoencoder_history.history)
plt.savefig(output_dir + 'loss.png')

print('Making predictions on the test set...')
if convolve:
    test_output_option = 'map_conv_norm'
else:
    test_output_option = 'map'
x_test = test_generator.map_reader(test_set_index, output=test_output_option)
generator_indexes = test_generator.get_generator_indexes()
x_test = x_test[generator_indexes]
test_set_index = test_set_index[generator_indexes]
output_autoencoder = autoencoder.predict(test_generator)

print('Plotting the input/output compare figure...')
for i in tqdm(range(n_test_set)):
    fig = compareinout2D(output_autoencoder[i], np.asarray(x_test)[i], int(dim / params['res_scale']))
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

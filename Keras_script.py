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
from my_models import DataGenerator2, model_design_2l, display_model, model_fit2, model_compile, \
    compareinout2D, fig_loss, model_design_3l, model_design_Unet, tweedie_loss_func, \
    basic_unet, model_design_Unet2, model_design_Unet3, model_design_Unet_NF, model_design_Unet_resnet, \
    model_design_Unet_resnet2, lc_loss_func, Unet_sobel_edges1, Unet_resnet_3param, VAE, \
    Unet_sobel_edges2, model_compile_change_lc_loss, Unet_take_two_channels_separately, Unet_lens_pos, \
    custom_loss_func, model_fit, scheduler

import tensorflow as tf

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
    parser.add_argument('-batch_size', action='store', default=8, help='Batch size for the autoencoder.')
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
                        default='./../data/ID_maps_selected_kappa_equal_gamma.dat',
                        help='Specify the directory where the list of map IDs is stored.')
    parser.add_argument('-conversion_file_directory', action='store',
                        default='./../data/maps_selected_kappa_equal_gamma.csv',
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
                        default='2l',
                        help='Which model design you want.'
                        )
    parser.add_argument('-saved_model_path', action='store',
                        default='./../../../fred/oz108/skhakpas/results/23-03-09-01-39-47/model_10000_8_1e-05',
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
                        default='simple',
                        help='If you want to change the optimizer during the training.')
    # Parses through the arguments and saves them within the keyword args
    arguments = parser.parse_args()
    return arguments


args = parse_options()
model_design_key = args.model_design
loss_function_key = args.loss_function
optimizer_key = args.optimizer
training_plan = args.training_plans

flow_label = args.flow_label
n_flows = int(args.n_flows)
z_size = args.bottleneck_size
n_test_set = int(args.test_set_size)
test_set_selection = args.test_set_selection
early_callback = args.early_callback
early_callback_type = args.early_callback_type
add_params = args.add_params
model_input_format = args.model_input_format
ngpu = int(args.ngpu)

date = args.date
os.system("mkdir ./../../../fred/oz108/skhakpas/results/" + str(date))

saved_model = args.saved_model
saved_model_path = args.saved_model_path
lens_pos_path = args.lens_pos_directory
lr_rate = float(args.lr_rate)
crop_scale = float(args.crop_scale)
n_epochs = int(args.n_epochs)
dim = int(args.dim)
lc_loss_function_metric = args.lc_loss_function_metric
lc_loss_function_coeff = int(args.lc_loss_function_coeff)
sample_size = int(args.sample_size)
include_lens_pos = args.add_lens_pos
include_map_units = args.add_map_units
output_dir = './../../../fred/oz108/skhakpas/results/' + args.date + '/'

print('Arguments are:')
print('Loss function: ', )

print('Setting up the network parameters...')
model_designs = {'2l': model_design_2l,
                 '3l': model_design_3l,
                 'Unet': model_design_Unet,
                 'Unet2': model_design_Unet2,
                 'Unet3': model_design_Unet3,
                 'basic_unet': basic_unet,
                 'Unet_NF': model_design_Unet_NF,
                 'Unet_sobel1': Unet_sobel_edges1,
                 'Unet_sobel2': Unet_sobel_edges2,
                 'Unet_resnet': model_design_Unet_resnet,
                 'Unet_resnet2': model_design_Unet_resnet2,
                 'VAE_Unet_Resnet': VAE,
                 'Unet_resnet_3param': Unet_resnet_3param,
                 'Unet_2ch_input': Unet_take_two_channels_separately,
                 'Unet_lens_pos': Unet_lens_pos}
loss_functions = {'binary_crossentropy': keras.losses.BinaryCrossentropy(from_logits=True),
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
keys_old = params.keys()

# Datasets
partition = {}


data_dict = {}

for index in range(11):
    path = args.directory_bulk + 'all_maps_batch' + str(index) + '.pkl'
    data_dict_tmp = pkl.load(open(path, "rb"))
    data_dict = {**data_dict, **data_dict_tmp}

# ls_maps = np.asarray([int(key) for key in data_dict.keys()])
if params['include_lens_pos']:
    all_lens_pos = pkl.load(open(lens_pos_path, "rb"))

ls_maps = np.asarray(list(data_dict.keys())) #np.loadtxt(args.list_IDs_directory, dtype=int)

random.seed(10)
shuffler = np.random.permutation(len(ls_maps))

shuffler = random.sample(list(shuffler), int(sample_size))
ls_maps = ls_maps[shuffler]
n_maps = len(ls_maps)

indx1 = np.arange(int(0.8 * n_maps), dtype=int)
indx2 = np.arange(int(0.8 * n_maps), int(0.9 * n_maps))
indx3 = np.arange(int(0.9 * n_maps), n_maps)


# strategy = tf.distribute.MirroredStrategy()
# print("Number of devices: {}".format(strategy.num_replicas_in_sync))

if saved_model:
    partition_direc = saved_model_path.split("model")[0] + 'sample_set_indexes.pkl'
    file_partition = open(partition_direc, 'rb')
    partition = pkl.load(file_partition)

    file_params = open(saved_model_path.split('model')[0] + 'params.pkl', 'rb')
    params_saved_ = pkl.load(file_params)

    # Generators
    params_saved = {}
    # params_saved = {key: params_saved[key] for key in params.keys()}
    for key in params.keys():
        if key in list(params_saved.keys()):
            params_saved[key] = params_saved_[key]
        else:
            params_saved[key] = params[key]

    print('Model params are: ')
    print(json.dumps(params_saved, indent=len(params_saved.keys())))

    if params['include_lens_pos']:
        training_generator = DataGenerator2(partition['train'], data_dict, dict2=all_lens_pos, **params)
        validation_generator = DataGenerator2(partition['validation'], data_dict, dict2=all_lens_pos, **params)
    else:
        training_generator = DataGenerator2(partition['train'], data_dict, **params_saved)
        validation_generator = DataGenerator2(partition['validation'], data_dict, **params_saved)

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

    params_saved['shuffle'] = False
    if params['include_lens_pos']:
        test_generator = DataGenerator2(test_set_index, data_dict, dict2=all_lens_pos, **params)
    else:
        test_generator = DataGenerator2(test_set_index, data_dict, **params_saved)
    # test_generator = DataGenerator(partition['test'], **params_saved)
    # test_set_index = random.sample(list(partition['test']), n_test_set)
    # generator_indexes = test_generator.get_generator_indexes()

    print('Loading a trained model...')
    # with strategy.scope():
    if loss_function_key == 'lc_loss':
        autoencoder = keras.models.load_model(saved_model_path, custom_objects= \
            {'lc_loglikelihood': lc_loss_func(metric=lc_loss_function_metric)})
    else:
        autoencoder = keras.models.load_model(saved_model_path,
                                              custom_objects={'custom_loglikelihood': loss_functions[loss_function_key]})
else:
    partition['train'] = ls_maps[indx1]
    partition['validation'] = ls_maps[indx2]
    partition['test'] = ls_maps[indx3]

    f = open(output_dir + 'sample_set_indexes.pkl', 'wb')
    pkl.dump(partition, f)

    # Generators
    if params['include_lens_pos']:
        training_generator = DataGenerator2(partition['train'], data_dict, dict2=all_lens_pos, **params)
        validation_generator = DataGenerator2(partition['validation'], data_dict, dict2=all_lens_pos, **params)
    else:
        training_generator = DataGenerator2(partition['train'], data_dict, **params)
        validation_generator = DataGenerator2(partition['validation'], data_dict, **params)

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

    params['shuffle'] = False
    if params['include_lens_pos']:
        test_generator = DataGenerator2(test_set_index, data_dict, dict2=all_lens_pos, **params)
    else:
        test_generator = DataGenerator2(test_set_index, data_dict, **params)
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
    params['loss_function_metric'] = lc_loss_function_metric
    params['optimizer'] = optimizer_key
    params['model_design'] = model_design_key

    print(json.dumps(params, indent=len(params.keys())))

    print('Saving the parameters of the run...')
    file = open(output_dir + 'params.pkl', 'wb')
    pkl.dump(params, file)

    # Design the model

    # with strategy.scope():
    if model_design_key == 'Unet_NF':
        autoencoder = model_designs[model_design_key](int((dim / params['res_scale']) * crop_scale),
                                                      flow_label=flow_label,
                                                      z_size=z_size, n_flows=n_flows)
        print('Desigining the network...')

    elif model_design_key.startswith('VAE'):
        if model_design_key == 'VAE_Unet_Resnet':
            encoder = vae_encoder(z_size, int((dim / params['res_scale']) * crop_scale), params['n_channels'],
                                  'relu')
        else:
            encoder = vae_encoder_3params(z_size, int((dim / params['res_scale']) * crop_scale), 3,
                                          params['n_channels'], 'relu')
        autoencoder = VAE(encoder,
                          vae_decoder(z_size, 'relu'))
        print('Desigining the network...')

    elif model_design_key == 'Unet_resnet_3param':
        autoencoder = Unet_resnet_3param(int((dim / params['res_scale']) * crop_scale), 3, params['n_channels'],
                                         'relu')
        print('Desigining the network...')

    elif crop_scale != 1. and model_design_key == 'Unet2':
        autoencoder = model_design_Unet2(int((dim / params['res_scale']) * crop_scale), first_down_sampling=4)
        print('Desigining the network...')

    elif model_design_key == 'Unet_lens_pos':
        autoencoder = Unet_lens_pos(int((dim / params['res_scale'])), 50)

    else:
        autoencoder = model_designs[model_design_key](int((dim / params['res_scale']) * crop_scale),
                                                      params['n_channels'])
        print('Desigining the network...')
display_model(autoencoder)
# with strategy.scope():
if training_plan == 'simple':
    model_compile(autoencoder, lr_rate, model_design_key, loss=loss_functions[loss_function_key],
                  optimizer_=optimizers[optimizer_key])
    # Train the model
    autoencoder_history = model_fit2(autoencoder, n_epochs,
                                     training_generator, validation_generator,
                                     filepath=output_dir,
                                     early_callback_=early_callback,
                                     early_callback_type=early_callback_type)
elif training_plan == 'changing_lc_loss':
    autoencoder_history = model_compile_change_lc_loss(autoencoder, lr_rate,
                                                       lc_loss_function_metric,
                                                       n_epochs,
                                                       training_generator, validation_generator,
                                                       output_dir,
                                                       optimizer_=optimizers[optimizer_key],
                                                       second_coeff=10,
                                                       early_callback_=False, early_callback_type='early_stop')
elif training_plan == 'changing_lr':
    callback = keras.callbacks.LearningRateScheduler(scheduler)
    model_compile(autoencoder, 0.01, model_design_key, loss=loss_functions[loss_function_key],
                  optimizer_=optimizers[optimizer_key])
    autoencoder_history = model_fit(autoencoder, n_epochs,
                                    training_generator, validation_generator,
                                    callback=[callback])


print('Saving the model...')
if not model_design_key.startswith('VAE'):
    autoencoder.save(output_dir +
                     'model_' +
                     str(params['dim']) +
                     '_' + str(params['batch_size']) +
                     '_' + str(lr_rate))
else:
    tf.saved_model.save(autoencoder, output_dir +
                        'model_' +
                        str(params['dim']) +
                        '_' + str(params['batch_size']) +
                        '_' + str(lr_rate))
print('Saving the model history...')
with open(output_dir + 'model_history.pkl', 'wb') as file_pi:
    pkl.dump(autoencoder_history.history, file_pi)

print('Plotting the loss function...')
fig = fig_loss(autoencoder_history.history)
plt.savefig(output_dir + 'loss.png')

print('Making predictions on the test set...')

# x_test = test_generator.map_reader(test_set_index, output='map_norm')
# generator_indexes = test_generator.get_generator_indexes()
# x_test = x_test[generator_indexes]
# test_set_index = test_set_index[generator_indexes]
# test_IDs = test_set_index[:, 0]
output_autoencoder = autoencoder.predict(test_generator)
test_IDs = test_set_index[:n_test_set]



print('Plotting the input/output compare figure...')
for i in tqdm(range(len(test_IDs))):
    ID = test_IDs[i]
    x_test = data_dict[ID]  # test_generator.map_reader([ID], output='map_norm') #[i, :, :, 0]
    fig = compareinout2D(output_autoencoder[i], np.asarray(x_test),
                         int((dim / params['res_scale']) * crop_scale))
    fig.savefig(output_dir + 'compare_' + str(ID) + '.png', bbox_inches='tight')
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

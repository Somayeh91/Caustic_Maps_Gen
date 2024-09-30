import argparse
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pkl
import json
from input_generators import data_gererators_set_up_AD, data_generator_set_up_kgs_bt
from more_info import *
import tensorflow as tf
from more_info import model_parameters, running_params
from maps_util import *
from training_utils import *
from plotting_utils import fig_loss, compareinout, compareinout_bt_to_kgs, analyze_bt_to_kgs_results
from loss_functions import *
import dill

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
    parser.add_argument('-batch_size', action='store', default=None, help='Batch size for the autoencoder.')
    parser.add_argument('-n_epochs', action='store', default=None, help='Number of epochs.')
    parser.add_argument('-n_channels', action='store', default=None, help='Number of channels.')
    parser.add_argument('-shuffle', action='store', default=None, help='shuffle data?')
    parser.add_argument('-lr_rate', action='store', default=None, help='Learning rate.')
    parser.add_argument('-res_scale', action='store', default=10,
                        help='With what scale should the original maps resoultion be reduced.')
    parser.add_argument('-crop_scale', action='store', default=None,
                        help='With what scale should the original maps be cropped and multipled by 4.')
    parser.add_argument('-date', action='store',
                        default='test',
                        help='Specify the directory where the results should be stored.')
    parser.add_argument('-list_IDs_directory', action='store',
                        default='./../data/ID_maps_selected_kappa_equal_gamma.dat',
                        help='Specify the directory where the list of map IDs is stored.')
    parser.add_argument('-output_directory', action='store',
                        default=None,
                        help='Specify the output directory otherwise it will create a new directory.')
    parser.add_argument('-test_set_size', action='store',
                        default=10,
                        help='Size of the test set.'),
    parser.add_argument('-sample_size', action='store',
                        default=None,
                        help='Size of the sample data set.'
                        )
    parser.add_argument('-model_design', action='store',
                        default='Unet2',
                        help='Which model design you want.'
                        )
    parser.add_argument('-saved_model_path', action='store',
                        default='./../../../fred/oz108/skhakpas/results/23-03-09-01-39-47/model_10000_8_1e-05',
                        help='In retrain_test mode, give the path to an already-saved model.'
                        )
    parser.add_argument('-saved_LSR_path', action='store',
                        default='./../../../fred/oz108/skhakpas/results/23-12-14-01-12-07/LSR_23-12-14-01-12'
                                '-07_samplesize_12342.npy',
                        help='In kgs_bt models, give the path to an already-saved bottleneck.'
                        )
    parser.add_argument('-loss_function', action='store',
                        default=None,
                        help='What is the loss function?'
                        )
    parser.add_argument('-optimizer', action='store',
                        default=None,
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
    parser.add_argument('-train_set_selection', action='store',
                        default='random',
                        help='choose between random, k=g, retrain_old, retrain_random, retrain_k=g')
    parser.add_argument('-early_callback', action='store',
                        default=None,
                        help='Do you want to add early calback when fitting the model?')
    parser.add_argument('-model_input_format', action='store',
                        default=None,
                        help='Does the model get x as data and y as targets? xy, other options are xx, x')
    parser.add_argument('-mode', action='store',
                        default='train_test',
                        help='choose between train_test, test, retrain_test')
    parser.add_argument('-test_IDs', action='store',
                        default=None,
                        help='if test_set_selection=given, give it a list of test IDs in .txt format')
    parser.add_argument('-add_lens_pos', action='store',
                        default=False,
                        help='Do you need lens positions as input?')
    parser.add_argument('-add_map_units', action='store',
                        default=False,
                        help='Do you need map units of R_E as input?')
    parser.add_argument('-n_test_set', action='store',
                        default=10,
                        help='Number of test objects.')
    parser.add_argument('-evaluation_metric', action='store',
                        default='ssm',
                        help='Name of the metric to use to evaluate AD maps.')
    # Parses through the arguments and saves them within the keyword args
    arguments = parser.parse_args()
    return arguments


args = parse_options()
model_design_key = args.model_design
model_params = model_parameters[model_design_key]


model_params, running_params = update_dicts(model_params, running_params, args)
output_dir = running_params['output_dir']

print('Model parameters are:')
[print(f"{key}: {value}") for key, value in model_params.items()]
print('----------------------')
print('Parameters of running the model are:')
[print(f"{key}: {value}") for key, value in running_params.items()]
print('----------------------')

if not running_params['mode'] == 'test':
    file = open(output_dir + 'params.pkl', 'wb')
    dill.dump(running_params, file)

    file = open(output_dir + 'model_params.pkl', 'wb')
    dill.dump(model_params, file)

# Setting up the data and cimpiling the model
print('Setting up the inputs...')
if not (model_design_key.startswith('kgs') or model_design_key.startswith('bt')):
    training_generator, \
    validation_generator, \
    test_generator, \
    running_params_, \
    model_params_, \
    partition = data_gererators_set_up_AD(running_params, model_params)
    print('Setting up the model in mode %s...' % running_params['mode'])
    if running_params['mode'].startswith('retrain') or \
            running_params['mode'] == 'test' or \
            running_params['mode'] == 'evaluate':
        running_params_old = running_params_
        model_params_old = model_params_
        model_ID = running_params_old['saved_model_path'].split('/')[-2]
        model_file = running_params_old['saved_model_path'].split('/')[-1]
        model = read_AD_model(model_ID, model_file, model_params_old['default_loss'])
    else:
        model = set_up_model(model_design_key, model_params, running_params)
else:
    train_object_X, train_object_Y, test_object_X, test_object_Y, ids_train, ids_test = data_generator_set_up_kgs_bt(
        model_design_key,
        running_params)
    print(ids_test)
    if running_params['mode'].startswith('retrain') or \
            running_params['mode'] == 'test' or \
            running_params['mode'] == 'evaluate':
        model = read_cnn_model_with_weights(running_params['saved_model_path'])
    else:
        model = set_up_model(model_design_key, model_params, running_params)

display_model(model)

if running_params['mode'] != 'test':
    # Compile the model
    print('Compiling the model %s...' % model_design_key)
    model = compile_model(model,
                          running_params['learning_rate'],
                          model_params['default_optimizer'],
                          model_params['default_loss'])

    # Fit the model
    print('Fitting the model %s for %i epochs with loss_function=%s...' % (model_design_key,
                                                                           running_params['n_epochs'],
                                                                           model_params['default_loss']))
    if not (model_design_key.startswith('kgs') or model_design_key.startswith('bt')):
        model_history = fit_model(model_design_key,
                                  model,
                                  running_params['n_epochs'],
                                  training_generator,
                                  y_train=None,
                                  x_validation=validation_generator,
                                  y_validation=None,
                                  filepath=None,
                                  early_callback_=running_params['early_callback'],
                                  use_multiprocessing=True)
    else:
        model_history = fit_model(model_design_key,
                                  model,
                                  running_params['n_epochs'],
                                  train_object_X,
                                  y_train=train_object_Y,
                                  x_validation=None,
                                  y_validation=None,
                                  filepath=None,
                                  early_callback_=running_params['early_callback'],
                                  use_multiprocessing=True)

    print('Saving the model...')
    model_json = model.to_json()

    with open(output_dir + "model_%i_%i_%s.json" % (running_params['dim'],
                                                    running_params['batch_size'],
                                                    str(running_params['learning_rate'])), 'w') as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(output_dir + "model_weights_%i_%i_%s.h5" % (running_params['dim'],
                                                                   running_params['batch_size'],
                                                                   str(running_params['learning_rate'])))

    print('Saving the model history...')
    with open(output_dir + 'model_history.pkl', 'wb') as file_pi:
        pkl.dump(model_history.history, file_pi)

    print('Plotting the loss function...')
    fig = fig_loss(model_history.history)
    plt.savefig(output_dir + 'loss.png')

print('Making predictions on the test set...')
if not (model_design_key.startswith('kgs') or model_design_key.startswith('bt')):
    output = model.predict(test_generator)
    test_IDs = partition['test']
    return_testimg_by_id = lambda a: test_generator.map_reader([test_IDs[a]], output='map_norm')
    plot_dim = int((running_params['dim'] / running_params['res_scale']) * model_params['crop_scale'])
else:
    output = model.predict(test_object_X)
    test_IDs = ids_test
    if not model_design_key == 'bt_to_kgs':
        plot_dim = running_params['dim']
        return_testimg_by_id = lambda a: test_object_Y[a]

if not model_design_key == 'bt_to_kgs':
    print('Plotting the input/output compare figure...')
    for i in tqdm(range(len(test_IDs))):
        ID = test_IDs[i]
        x_test = return_testimg_by_id(i)

        fig = compareinout(output[i],
                           np.asarray(x_test),
                           plot_dim,
                           output_dir,
                           model_design_key,
                           ID)
    # Evaluating the model by comparing decoder_generated maps from CNN_generated LSR with the Gmaps
    AD_model_ID = running_params['saved_LSR_path'].split('/')[8]
    print(ids_test)
    metric = evaluate_kgs_generated_maps(test_IDs,
                                         AD_model_ID,
                                         output,
                                         running_params,
                                         metric=running_params['evaluation_metric'])
    print('Evaluating the model by comparing decoder_generated maps from CNN_generated LSR with the Gmaps:')
    print('%s metric for %i maps using the %s model is %.3f.' % (running_params['evaluation_metric'],
                                                                 len(test_IDs),
                                                                 model_design_key,
                                                                 metric))
else:
    # analyze_bt_to_kgs_results(output, test_object_Y, output_dir)
    compareinout_bt_to_kgs(output, test_object_Y, output_dir)
    print('Evaluating the model by comparing generated kgs from CNN with true kgs:')
    print('Average L2 metric for %i maps using the %s model is for kappa=%.3f, gamma=%.3f, and s=%.3f.' %
          (len(test_IDs),
           model_design_key,
           np.mean(np.linalg.norm(test_object_Y[:, 0] - output[:, 0])),
           np.mean(np.linalg.norm(test_object_Y[:, 1] - output[:, 1])),
           np.mean(np.linalg.norm(test_object_Y[:, 2] - output[:, 2]))
           ))


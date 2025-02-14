from loss_functions import tweedie_loss_func, \
    lc_loss_func, \
    custom_loss_func

from conv_utils import convolve_map
import numpy as np
from keras.models import load_model
from more_info import best_AD_models_info
import random
import pandas as pd
import pickle as pkl
import json
from more_info import directory_bulk, len_pos_directory, loss_functions, optimizers
from keras.models import model_from_json
from skimage.transform import resize
import os, sys
import keras
import h5py
from tensorflow.keras.preprocessing.sequence import pad_sequences

map_direc = './../../../fred/oz108/GERLUMPH_project/DATABASES/gerlumph_db/'
model_direc = './../../../fred/oz108/skhakpas/results/'


# map_direc = '/Users/somayeh/Downloads/maps/'
# model_direc = '/Users/somayeh/Downloads/job_results_downloaded/'


def read_AD_model(direc, cost_label):
    """
    Loads an autoencoder(AD) model from the specified directory,
    handling custom loss functions if needed.

    Parameters:
    -----------
    direc : str
        Path to the saved model directory or file.
    cost_label : str or None
        Custom loss function identifier used during model training.
        - If `None`, the model is loaded with default settings.
        - If provided, the function converts the label to the corresponding loss function
          using `convert_custom_loss_labels()`.

    Returns:
    --------
    autoencoder : keras.Model
        The loaded Keras autoencoder model.

    Notes:
    ------
    - Calls `convert_custom_loss_labels(cost_label)` to obtain the appropriate loss function
      if a custom loss is specified.
    - Uses `load_model()` to restore the saved model.
    - If `cost_label` is `None`, the model is loaded without specifying custom objects.
    - Assumes `convert_custom_loss_labels()` is a predefined function that maps loss labels
      to their corresponding implementations.
    """

    call_loss_label, call_loss_function = convert_loss_labels(cost_label)
    if not call_loss_label is None:
        autoencoder = load_model(direc,
                                 custom_objects={call_loss_label: call_loss_function})
    else:
        autoencoder = load_model(direc)
    return autoencoder


def read_json_model(directory, cost_label):
    # Load the JSON file
    call_loss_label, call_loss_function = convert_loss_labels(cost_label)

    json_file = open(directory, 'r')

    loaded_model_json = json_file.read()
    json_file.close()

    # Reconstruct the model
    if not call_loss_label is None:
        autoencoder = model_from_json(loaded_model_json, custom_objects={call_loss_label: call_loss_function})
    else:
        autoencoder = model_from_json(loaded_model_json)
    return autoencoder


def load_model_weights(model, directory):
    model.load_weights(directory)
    return model

def read_saved_model(directory,
                     model_ID,
                     model_file,
                     cost_label=None,
                     mode='json'):
    """
    Loads a saved machine learning model from the specified directory, supporting different storage formats.

    Parameters:
    -----------
    directory : str
        Path to the directory where the model is stored.
    model_ID : str
        Unique identifier for the model within the directory.
    model_file : str
        Base filename of the model (without extension).
    cost_label : str or None, default=None
        Custom loss function label, used when loading the model.
    mode : str, default='json'
        Specifies the format of the saved model:
        - 'json': Loads a model from a JSON architecture file and corresponding HDF5 weight file.
        - 'old_version': Loads a model using `read_AD_model()` (assumed to handle older saved models).

    Returns:
    --------
    model : keras.Model
        The loaded Keras model.

    Notes:
    ------
    - If `mode='json'`, the function:
        1. Loads the model architecture from a JSON file (`model_file.json`).
        2. Loads the model weights from a corresponding HDF5 file (`model_file_weights.h5`).
    - If `mode='old_version'`, the function loads the model using `read_AD_model()`,
      which may support models saved in an older format.
    - Assumes `read_json_model()` and `load_model_weights()` are predefined functions
      for handling JSON-based model loading.
    - Assumes `read_AD_model()` is available to load older models with optional custom loss functions.
    """

    if mode == 'json':
        print('reading model in path=%s' % (directory + model_ID + '/' + model_file + '.json'))
        model = read_json_model(directory + model_ID + '/' + model_file + '.json', cost_label)
        print('Loading weights in path=%s' % (directory + model_ID + '/' + model_file + '_weights.h5'))
        model = load_model_weights(model, directory + model_ID + '/' + model_file + '_weights.h5')

    if mode == 'old_version':
        model = read_AD_model(directory + model_ID + '/' + model_file, cost_label)

    if model is None:
        print('No model was found.')
        sys.exit()

    return model


def AD_decoder(AD, LSR_input_layer_name, LSR_output_layer_name):
    recons_output = AD.get_layer(LSR_output_layer_name).output
    recons_input = AD.get_layer(LSR_input_layer_name).input
    model_rec = keras.models.Model(inputs=recons_input, outputs=recons_output)
    return model_rec


def read_cnn_model_with_weights(saved_model_path):
    file_params = open(saved_model_path + 'params.pkl', 'rb')
    running_params = pkl.load(file_params)
    model_path = saved_model_path + "model_%i_%i_%s.json" % (running_params['input_size'],
                                                             running_params['batch_size'],
                                                             str(running_params['learning_rate']))
    weights_path = saved_model_path + "model_weights_%i_%i_%s.h5" % (running_params['input_size'],
                                                                     running_params['batch_size'],
                                                                     str(running_params['learning_rate']))
    # load json and create model
    json_file = open(model_path, 'r')

    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(weights_path)
    print("Loaded model from disk")
    return loaded_model


def read_binary_map(ID, input_size, output_size=1000, to_mag=True):
    f1 = open(map_direc + str(ID) + "/map.bin", "rb")
    map_tmp = np.fromfile(f1, 'i', -1, "")
    map = (np.reshape(map_tmp, (-1, input_size)))
    scaling_factor = input_size // output_size

    if to_mag:
        mag_convertor = read_map_meta(ID)[0]
        if scaling_factor != 0:
            if input_size == 4096:
                return map[0::scaling_factor, 0::scaling_factor][12:-12, 12:-12] * mag_convertor
            else:
                return map[0::scaling_factor, 0::scaling_factor] * mag_convertor
        else:
            return map * mag_convertor
    else:
        if scaling_factor != 0:
            if input_size == 4096:
                return map[0::scaling_factor, 0::scaling_factor][12:-12, 12:-12]
            else:
                return map[0::scaling_factor, 0::scaling_factor]
        else:
            return map


def read_map_meta(ID):
    f2 = open(map_direc + str(ID) + "/mapmeta.dat", "r")
    lines = f2.readlines()

    # print(lines)

    k, g, s = float(lines[3].split(' ')[0]), float(lines[3].split(' ')[1]), float(lines[3].split(' ')[2])
    mag_avg = float(lines[0].split(' ')[0])
    ray_avg = float(lines[0].split(' ')[1].split('/')[0])

    mag_convertor = np.abs(mag_avg / ray_avg)

    return mag_convertor, k, g, s, mag_avg, ray_avg


def read_map_meta_for_file(input_dir,
                           output_filename,
                           output_dir):
    print('Reading the file...')
    all_ID = np.loadtxt(input_dir, usecols=(0,), dtype=int)
    print('The file contains %i IDs.' % len(all_ID))
    print('Reading Meta Data...')
    results = np.asarray([read_map_meta(ID) for ID in all_ID])
    print('Meta Data successfully read.')

    df = pd.DataFrame({'ID': all_ID,
                       'const': results[:, 0],
                       'k': results[:, 1],
                       'g': results[:, 2],
                       's': results[:, 3],
                       'mag_avg': results[:, 4],
                       'ray_avg': results[:, 5]})
    df.to_csv(output_dir + output_filename)


def norm_maps(map, offset=0.004, norm_min=-3, norm_max=6):
    '''
    This function normalized the maps when map values are in units of magnification.
    :param map: input map in units of magnification
    :param offset: an offset value to un-zero map values
    :param norm_min: minimum value for min-max normalization (obtained from all maps)
    :param norm_max: maximum value for min-max normalization (obtained from all maps)
    :return: an output map with pixel values between 0 and 1e
    '''
    temp = np.log10(map + offset)
    return NormalizeData(temp, data_max=norm_max, data_min=norm_min)


def reverse_norm(map, norm_min=-3, norm_max=6):
    return (map * (norm_max - norm_min)) + norm_min


def reverse_norm_maps(map, coeff, offset=0.004, norm=True, norm_min=-3, norm_max=6):
    if norm:
        temp = (map * (norm_max - norm_min)) + norm_min
    else:
        temp = map
    return (((10 ** temp) - offset) / coeff).astype('int32')


def reverse_maps_into_mag(map, offset=0.004, norm=True, norm_min=-3, norm_max=6):
    if norm:
        temp = (map * (norm_max - norm_min)) + norm_min
    else:
        temp = map
    return ((10 ** temp) - offset).astype('float16')


def prepare_cuttout_map(ID, rsrc=0):
    map_cuttout = read_binary_map(ID)
    mag_convertor = read_map_meta(ID)[0]

    models = best_AD_models_info['job_names']
    model_files = best_AD_models_info['job_model_filename']
    cost_labels = best_AD_models_info['job_cost_labels']
    shape_ = map_cuttout.shape[0]

    map_cuttout = norm_maps(map_cuttout,
                            mag_convertor)
    images_cutout_exmp = np.zeros((len(models) + 1,
                                   shape_,
                                   shape_))

    for m, model in enumerate(models):
        autoencoder = read_AD_model(model, model_files[m], cost_labels[m])
        AD_map = autoencoder.predict(map_cuttout.reshape((1, shape_, shape_, 1)))
        AD_map = AD_map.reshape((shape_, shape_))

        if rsrc == 0:
            images_cutout_exmp[m + 1, :, :] = AD_map
            if m == 0:
                images_cutout_exmp[0, :, :] = map_cuttout

        else:
            AD_map_conv = convolve_map(AD_map,
                                       rsrc)
            images_cutout_exmp[m + 1, :, :] = AD_map_conv

            if m == 0:
                map_conv = convolve_map(map_cuttout,
                                        rsrc)
                images_cutout_exmp[0, :, :] = map_conv

    return images_cutout_exmp


def img_cut_out_generator(ID, rsrc, size=200):
    images = prepare_cuttout_map(ID, rsrc=rsrc)
    imgs = np.zeros((len(images), size, size))
    for i, im in enumerate(images):
        imgs[i, :, :] = im[0:size, 0:size]
    return imgs


def chi2_calc(f, x):
    return np.sum((f - x) ** 2)


def resize(image, output_shape):
    output_shape = tuple(output_shape)
    output_ndim = len(output_shape)
    input_shape = image.shape
    if output_ndim > image.ndim:
        # append dimensions to input_shape
        input_shape += (1,) * (output_ndim - image.ndim)
        image = np.reshape(image, input_shape)
    elif output_ndim == image.ndim - 1:
        # multichannel case: append shape of last axis
        output_shape = output_shape + (image.shape[-1],)
    elif output_ndim < image.ndim:
        raise ValueError("output_shape length cannot be smaller than the "
                         "image number of dimensions")

    return image, output_shape


def eval_maps_selection(num=100, seed=33):
    random.seed(seed)
    path = './../data/'
    IDs = np.loadtxt(path + 'GD1_ids_list2.txt', dtype=int)[:, 0]
    IDs_selected = random.sample(list(IDs), num)
    with open(path + 'eval_maps_%imaps_seed%i.txt' % (num, seed), 'w') as file:
        for ID in IDs_selected:
            file.write(str(ID) + '\n')
        file.close()
    return IDs_selected


def split_data(keys,
               train_percentage=0.8,
               valid_percentage=0.1):
    partition = {}
    ls_maps = keys
    random.seed(10)
    shuffler = np.random.permutation(len(ls_maps))
    sample_size = len(ls_maps)

    shuffler = random.sample(list(shuffler), int(sample_size))
    ls_maps = ls_maps[shuffler]
    n_maps = len(ls_maps)

    if train_percentage > 0.98:
        indx1 = np.arange(n_maps, dtype=int)
        indx2 = np.array([])
        indx3 = np.array([])
    else:
        indx1 = np.arange(int(train_percentage * n_maps), dtype=int)
        indx2 = np.arange(int(train_percentage * n_maps),
                          int((train_percentage + valid_percentage) * n_maps))
        indx3 = np.arange(int((train_percentage + valid_percentage) * n_maps), n_maps)

    partition['train'] = ls_maps[indx1]
    partition['validation'] = ls_maps[indx2]
    partition['test'] = ls_maps[indx3]

    return partition


def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]


def read_me_creator(
        output_dir,
        num_maps,
        num_lc,
        IDs,
        rsrcs,
        args):
    with open(output_dir + 'read_me.txt', 'w') as file:
        file.write("The code evaluation.py was run on %i maps with IDs: \n" % num_maps)
        # file.write(str(report)+'\n')
        for ID in IDs:
            file.write(str(ID) + ' ,')
        file.write('\n')
        file.write('Maps were convolved with source sizes: ')
        for rsrc in rsrcs:
            file.write('%.1f ,' % rsrc)
        file.write('\n')
        file.write('%i lightcurves were generated for each map and its convolved versions.' % num_lc)
        file.write('\n')
        file.write('Results are saved at %s' % output_dir)

        file.write(json.dumps(args, indent=len(args.keys())))

        file.close()


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


def update_dicts(model_dict, train_dict, arguments):
    if arguments.loss_function is None:
        pass
    else:
        model_dict['default_loss'] = loss_functions[arguments.loss_function]
    model_dict['loss_label'] = arguments.loss_function
    if arguments.optimizer is None:
        pass
    else:
        model_dict['default_optimizer'] = optimizers[arguments.optimizer]
    if arguments.n_channels is None:
        pass
    else:
        model_dict['n_channels'] = arguments.n_channels
    if arguments.crop_scale is None:
        pass
    else:
        model_dict['crop_scale'] = arguments.crop_scale
    if arguments.add_lens_pos is None:
        pass
    else:
        model_dict['include_lens_pos'] = arguments.add_lens_pos
    if arguments.add_map_units is None:
        pass
    else:
        model_dict['include_map_units'] = arguments.add_map_units
    if arguments.model_input_format is None:
        pass
    else:
        model_dict['input_output_format'] = arguments.model_input_format
    if arguments.flow_label is None:
        pass
    else:
        model_dict['flow_label'] = arguments.flow_label
    if arguments.n_flows is None:
        pass
    else:
        model_dict['n_flows'] = arguments.n_flows
    if arguments.bottleneck_size is None:
        pass
    else:
        model_dict['z_size'] = arguments.bottleneck_size

    ############### Check running parameters ###############
    if arguments.lr_rate is None:
        pass
    else:
        train_dict['learning_rate'] = float(arguments.lr_rate)
    if arguments.batch_size is None:
        pass
    else:
        train_dict['batch_size'] = int(arguments.batch_size)
    if arguments.input_size is None:
        pass
    else:
        train_dict['input_size'] = int(arguments.input_size)
    if arguments.output_size is None:
        pass
    else:
        train_dict['output_size'] = int(arguments.output_size)
    if arguments.res_scale is None:
        pass
    else:
        train_dict['res_scale'] = int(arguments.res_scale)
    if arguments.n_epochs is None:
        pass
    else:
        train_dict['n_epochs'] = int(arguments.n_epochs)
    if arguments.sample_size is None:
        pass
    else:
        train_dict['sample_size'] = int(arguments.sample_size)
    if arguments.saved_model_path is None:
        pass
    else:
        train_dict['saved_model_path'] = arguments.saved_model_path

    if arguments.saved_model_format is None:
        pass
    else:
        train_dict['saved_model_format'] = arguments.saved_model_format

    if arguments.saved_LSR_path is None:
        pass
    else:
        train_dict['saved_LSR_path'] = arguments.saved_LSR_path
    if arguments.mode is None:
        pass
    else:
        train_dict['mode'] = arguments.mode
    if arguments.shuffle is None:
        pass
    else:
        train_dict['shuffle'] = arguments.shuffle
    if arguments.directory is None:
        pass
    else:
        train_dict['path'] = arguments.directory
    if arguments.output_directory is None:
        os.system("mkdir ./../../../fred/oz108/skhakpas/results/" + str(arguments.date))
        output_dir = './../../../fred/oz108/skhakpas/results/' + arguments.date + '/'
        train_dict['output_dir'] = output_dir
    else:
        train_dict['output_dir'] = arguments.output_directory

    train_dict['n_test_set'] = int(arguments.n_test_set)
    train_dict['train_selection'] = arguments.train_set_selection
    train_dict['test_selection'] = arguments.test_set_selection

    if train_dict['test_selection'] == 'given':
        train_dict['test_IDs'] = [int(item) for item in arguments.test_IDs.split(',')]
    else:
        train_dict['test_IDs'] = arguments.test_IDs
    if arguments.early_callback is None:
        train_dict['early_callback'] = None
    else:
        train_dict['early_callback'] = arguments.early_callback.split(',')

    train_dict['evaluation_metric'] = arguments.evaluation_metric

    if train_dict['train_selection'] == 'random' or train_dict['train_selection'] == 'retrain_random':
        train_dict['conv_const'] = './../data/all_maps_meta_kgs.csv'
        train_dict['list_IDs_directory'] = './../data/GD1_ids_list.txt'
    elif train_dict['train_selection'] == 'k=g' or train_dict['train_selection'] == 'retrain_k=g':
        train_dict['conv_const'] = './../data/maps_selected_kappa_equal_gamma.csv'
        train_dict['list_IDs_directory'] = './../data/ID_maps_selected_kappa_equal_gamma.dat'
    elif train_dict['train_selection'] == 'repeated_kg' or train_dict['train_selection'] == 'retrain_repeated_kg':
        train_dict['conv_const'] = './../data/all_maps_meta_kgs.csv'
        train_dict['list_IDs_directory'] = './../data/gd0_IDs.dat'

    return model_dict, train_dict


def h5_file_reader(path):
    with h5py.File(path, "r") as f:
        # Print all root level object names (aka keys)
        # these can be group or dataset names
        print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        print(type(f[a_group_key]))

        # If a_group_key is a group name,
        # this gets the object names in the group and returns as a list
        data = list(f[a_group_key])

        # If a_group_key is a dataset name,
        # this gets the dataset values and returns as a list
        maps = list(f['maps'])
        IDs = list(f['IDs'])
        # preferred methods to get dataset values:
        ds_obj = f[a_group_key]  # returns as a h5py dataset object
        ds_arr = f[a_group_key][()]
    return maps, IDs


def masking_arrays(raw_array, raw_array_maxs=482.816, raw_array_mins=-482.797, MAX_ARRAY_SIZE=244784):
    tmp = np.concatenate((raw_array[:, 0].reshape((len(raw_array), 1)), raw_array[:, 1].reshape((len(raw_array), 1))),
                         axis=1)
    output = NormalizeData(tmp, data_max=raw_array_maxs, data_min=raw_array_mins)
    return pad_sequences([output], maxlen=MAX_ARRAY_SIZE, padding="post", dtype="float32")


def read_all_maps(n_batches):
    # data_dict = {}
    #
    # for index in range(n_batches):
    #     path = directory_bulk + 'all_maps_batch' + str(index) + '.pkl'
    #     data_dict_tmp = pkl.load(open(path, "rb"))
    #     data_dict = {**data_dict, **data_dict_tmp}
    maps, IDs = h5_file_reader(directory_bulk + '4096pix_maps.h5')
    maps = np.asarray(maps)
    IDs = np.asarray(IDs)
    data_dict = dict(zip(IDs, maps))
    return data_dict


# def read_all_data():
#     data_dict = {}
#
#     for index in range(11):
#         path = './../../../fred/oz108/skhakpas/all_maps_batch' + str(index) + '.pkl'
#         data_dict_tmp = pkl.load(open(path, "rb"))
#         data_dict = {**data_dict, **data_dict_tmp}
#
#     return data_dict


def read_all_lens_pos():
    all_lens_pos = pkl.load(open(len_pos_directory, "rb"))
    return all_lens_pos


def process_read_lens_pos(ID, masking=True):
    lens_pos = np.loadtxt(map_direc + str(ID) + "/lens_pos.dat")
    if masking:
        lens_pos = masking_arrays(lens_pos)
    return lens_pos[0]


def scale_images(images, new_shape=(1000, 1000, 1)):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)


def check_shapes(x, y):
    if x.shape == y.shape:
        return True
    else:
        return False


def split_data(n_maps, batch_size, test_selection, n_test_set):
    """
        Splits a dataset of maps into training, validation, and test indices based on predefined fractions.

        Parameters:
        -----------
        n_maps : int
            Total number of maps available in the dataset.
        batch_size : int
            The batch size used for training; ensures that training and validation sets have at least `batch_size` samples.
        test_selection : str
            Specifies the method for selecting test samples. Options include:
            - 'sorted': Selects a sorted subset of test samples.
            - 'random': Randomly selects test samples.
            - 'given': Uses a predefined set of test samples.
            - Other values default to a proportion-based test split.
        n_test_set : int
            Number of test samples to be selected when applicable.

        Returns:
        --------
        indx1 : numpy.ndarray
            Indices for the training set (default: first 80% of the dataset).
        indx2 : numpy.ndarray
            Indices for the validation set (default: next 10% of the dataset).
        indx3 : numpy.ndarray
            Indices for the test set (default: last 10% of the dataset, adjusted based on `test_selection`).

        Notes:
        ------
        - The dataset is split into:
          - 80% training (`indx1`).
          - 10% validation (`indx2`).
          - 10% test (`indx3`), with adjustments based on `test_selection` and `n_test_set`.
        - If `batch_size` is larger than the training or validation set size, the function adjusts `indx2` and `indx3` accordingly.
        - If `n_test_set` is greater than the available test samples, the function dynamically adjusts the test fraction.
        - The function exits with an error message if `batch_size` exceeds the total sample size.
        """
    indx1 = np.arange(int(0.8 * n_maps), dtype=int)
    indx2 = np.arange(int(0.8 * n_maps), int(0.9 * n_maps))
    indx3 = np.arange(int(0.9 * n_maps), n_maps)
    fract = batch_size / n_maps
    if len(indx1) < batch_size:
        print('Batch size should not be larger than the sample size.')
        sys.exit()
    if len(indx2) < batch_size:
        indx2 = np.arange(int((1 - (2 * fract)) * n_maps) - 1, int((1 - fract) * n_maps) + 1)
    if test_selection == 'sorted' or test_selection == 'random' or test_selection == 'given':
        if n_test_set > len(indx3):
            fract2 = n_test_set / n_maps
            indx3 = np.arange(int((1 - fract2) * n_maps) - 1, n_maps)
    else:
        if len(indx3) < batch_size:
            indx3 = np.arange(int((1 - fract) * n_maps) - 1, n_maps)

    return indx1, indx2, indx3


def convert_loss_labels(label):
    if label == 'tweedie':
        return 'tweedie_loglikelihood', loss_functions[label]
    elif label == 'lc_loss':
        return 'lc_loglikelihood', loss_functions[label]
    elif label == 'custom':
        return 'custom_loglikelihood', loss_functions[label]
    else:
        return None, None

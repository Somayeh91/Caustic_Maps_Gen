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
import os
import keras

map_direc = './../../../fred/oz108/GERLUMPH_project/DATABASES/gerlumph_db/'
model_direc = './../../../fred/oz108/skhakpas/results/'


# map_direc = '/Users/somayeh/Downloads/maps/'
# model_direc = '/Users/somayeh/Downloads/job_results_downloaded/'


def read_AD_model(model_ID, model_file, cost_label):
    if cost_label.startswith('lc'):
        if cost_label == 'lc_bce':
            autoencoder = load_model(model_direc +
                                     model_ID + '/' + model_file,
                                     custom_objects={'tweedie_loglikelihood': tweedie_loss_func(0.5)})
        else:
            metric = cost_label.split('_')[1]
            autoencoder = load_model(model_direc +
                                     model_ID + '/' + model_file,
                                     custom_objects={'lc_loglikelihood': lc_loss_func(metric)})
    elif cost_label == 'custom':
        autoencoder = load_model(model_direc +
                                 model_ID + '/' + model_file,
                                 custom_objects={'custom_loglikelihood': custom_loss_func()})
    else:
        autoencoder = load_model(model_direc +
                                 model_ID + '/' + model_file)
    return autoencoder

def AD_decoder(AD, LSR_input_layer_name, LSR_output_layer_name):
    recons_output = AD.get_layer(LSR_output_layer_name).output
    recons_input = AD.get_layer(LSR_input_layer_name).input
    model_rec = keras.models.Model(inputs=recons_input, outputs=recons_output)
    return model_rec

def read_cnn_model_with_weights(saved_model_path):
    file_params = open(saved_model_path + 'params.pkl', 'rb')
    running_params = pkl.load(file_params)
    model_path = saved_model_path + "model_%i_%i_%s.json" % (running_params['dim'],
                                                             running_params['batch_size'],
                                                             str(running_params['learning_rate']))
    weights_path = saved_model_path + "model_weights_%i_%i_%s.h5" % (running_params['dim'],
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


def read_binary_map(ID, scaling_factor=10, to_mag=False):
    f1 = open(map_direc + str(ID) + "/map.bin", "rb")
    map_tmp = np.fromfile(f1, 'i', -1, "")
    map = (np.reshape(map_tmp, (-1, 10000)))

    if to_mag:
        mag_convertor = read_map_meta(ID)[0]
        return map[0::scaling_factor, 0::scaling_factor] * mag_convertor
    else:
        return map[0::scaling_factor, 0::scaling_factor]


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


def norm_maps(map, coeff, offset=0.004, norm=True, norm_min=-3, norm_max=6):
    temp = np.log10(map * coeff + offset)
    if norm:
        return NormalizeData(temp, data_max=norm_max, data_min=norm_min)
    else:
        return temp


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


def read_all_data():
    data_dict = {}

    for index in range(11):
        path = './../../../fred/oz108/skhakpas/all_maps_batch' + str(index) + '.pkl'
        data_dict_tmp = pkl.load(open(path, "rb"))
        data_dict = {**data_dict, **data_dict_tmp}

    return data_dict


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
    if arguments.dim is None:
        pass
    else:
        train_dict['dim'] = int(arguments.dim)
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

    train_dict['n_test_set'] = arguments.n_test_set
    train_dict['test_IDs'] = [int(item) for item in arguments.test_IDs.split(',')]
    train_dict['train_selection'] = arguments.train_set_selection
    train_dict['test_selection'] = arguments.test_set_selection
    train_dict['early_callback'] = arguments.early_callback
    train_dict['evaluation_metric'] = arguments.evaluation_metric

    if train_dict['train_selection'] == 'random' or train_dict['train_selection'] == 'retrain_random':
        train_dict['conv_const'] = './../data/all_maps_meta_kgs.csv'
        train_dict['list_IDs_directory'] = './../data/GD1_ids_list.txt'
    elif train_dict['train_selection'] == 'k=g' or train_dict['train_selection'] == 'retrain_k=g':
        train_dict['conv_const'] = './../data/maps_selected_kappa_equal_gamma.csv'
        train_dict['list_IDs_directory'] = './../data/ID_maps_selected_kappa_equal_gamma.dat'

    return model_dict, train_dict


def read_all_maps(n_batches):
    data_dict = {}

    for index in range(n_batches):
        path = directory_bulk + 'all_maps_batch' + str(index) + '.pkl'
        data_dict_tmp = pkl.load(open(path, "rb"))
        data_dict = {**data_dict, **data_dict_tmp}
    return data_dict


def read_all_lens_pos():
    all_lens_pos = pkl.load(open(len_pos_directory, "rb"))
    return all_lens_pos


def scale_images(images, new_shape=(1000, 1000, 1)):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)

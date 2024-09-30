from tensorflow import keras
import random
import numpy as np
import pickle as pkl
from maps_util import read_all_maps
from my_models import *
from maps_util import NormalizeData, read_all_lens_pos, read_binary_map
from more_info import best_AD_models_info
from maps_util import read_AD_model
from metric_utils import calculate_ks_metric, process_FID
from FID_calculator import read_inceptionV3
from prepare_maps import process_lc4


def display_model(model):
    model.summary()


def read_saved_model(path):
    autoencoder_model = keras.models.load_model(path)
    return autoencoder_model


def scheduler(epoch, lr):
    step = epoch // 50
    if step == 0:
        return lr
    elif step == 1:
        return lr / 10
    elif step == 2:
        return lr / 100
    elif step == 3:
        return lr / 1000
    elif step > 3:
        return lr / 10000


def compile_model(model,
                  learning_rate,
                  optimizer,
                  loss):
    model.compile(optimizer=optimizer(learning_rate=learning_rate),
                  loss=loss)
    return model


def fit_model(model_design_key,
              model,
              epochs,
              x_train,
              y_train=None,
              x_validation=None,
              y_validation=None,
              filepath=None,
              early_callback_=None,
              use_multiprocessing=True):
    ec = []
    if early_callback_ is not None:
        if early_callback_ == 'early_stop':
            ec.append(EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, min_delta=0.00001))
        elif early_callback_ == 'model_checkpoint':
            ec.append(ModelCheckpoint(
                filepath=filepath,
                save_freq='epoch'))
        elif early_callback_ == 'changing_lr':
            ec.append(keras.callbacks.LearningRateScheduler(scheduler))

    if not (model_design_key.startswith('kgs') or model_design_key.startswith('bt')):
        history = model.fit_generator(generator=x_train,
                                      validation_data=x_validation,
                                      epochs=epochs,
                                      callbacks=ec,
                                      use_multiprocessing=use_multiprocessing)

    else:
        history = model.fit(x_train,
                            y_train,
                            epochs=epochs,
                            batch_size=epochs,
                            validation_split=0.2,
                            callbacks=ec)
    return history


def prepare_input_to_fit_keras_ADs(running_params, model_params):
    sample_size = running_params['sample_size']
    mode = running_params['mode']
    test_selection = running_params['test_selection']
    train_selection = running_params['train_selection']
    output_dir = running_params['output_dir']
    n_test_set = running_params['n_test_set']

    if mode == 'train_test':

        if train_selection == 'k=g':
            ls_maps = np.loadtxt(running_params['list_IDs_directory'], dtype=int)
            data_dict = None
        else:
            maps_per_batch = 1122
            n_batches = int(((sample_size / maps_per_batch) - (sample_size / maps_per_batch) % 1))
            data_dict = read_all_maps(n_batches)
            all_keys = data_dict.keys()
            ls_maps = random.sample(list(all_keys), int(sample_size))

        partition = {}

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
        all_test_set = ls_maps[indx3]

        if test_selection == 'random':
            partition['test'] = np.asarray(random.sample(list(all_test_set), n_test_set))
        elif test_selection == 'all_test':
            partition['test'] = all_test_set
        elif test_selection == 'all_train':
            partition['test'] = partition['train']
        elif test_selection == 'all_data':
            partition['test'] = ls_maps
        elif test_selection == 'sorted':
            partition['test'] = np.sort(all_test_set)[:n_test_set]
        elif test_selection == 'given':
            partition['test'] = np.loadtxt(running_params['test_IDs'])

        f = open(output_dir + 'sample_set_indexes.pkl', 'wb')
        pkl.dump(partition, f)

    elif mode == 'retrain_test' or mode == 'test':
        if train_selection == 'retrain_old':
            partition_direc = running_params['saved_model_path'].split("model")[0] + 'sample_set_indexes.pkl'
            file_partition = open(partition_direc, 'rb')
            partition = pkl.load(file_partition)

            file_params = open(running_params['saved_model_path'].split('model')[0] + 'params.pkl', 'rb')
            new_running_params = pkl.load(file_params)

            file_params = open(running_params['saved_model_path'].split('model')[0] + 'model_params.pkl', 'rb')
            model_params = pkl.load(file_params)
            all_test_set = partition['test']
            ls_maps = list(partition['train']) + list(partition['validation']) + list(partition['test'])
            data_dict = None
            running_params = new_running_params

        else:
            if train_selection == 'random' or train_selection == 'retrain_random':
                maps_per_batch = 1122
                n_batches = int(((sample_size / maps_per_batch) - (sample_size / maps_per_batch) % 1))
                data_dict = read_all_maps(n_batches)
                all_keys = data_dict.keys()
                ls_maps = random.sample(list(all_keys), int(sample_size))

            elif train_selection == 'retrain_k=g':
                ls_maps = np.loadtxt(running_params['list_IDs_directory'], dtype=int)
                data_dict = None

            partition = {}

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
            all_test_set = ls_maps[indx3]

        if test_selection == 'random':
            partition['test'] = np.asarray(random.sample(list(all_test_set), n_test_set))
        elif test_selection == 'all_test':
            partition['test'] = all_test_set
        elif test_selection == 'all_train':
            partition['test'] = partition['train']
        elif test_selection == 'all_data':
            partition['test'] = ls_maps
        elif test_selection == 'sorted':
            partition['test'] = np.sort(all_test_set)[:n_test_set]
        elif test_selection == 'given':
            partition['test'] = np.loadtxt(running_params['test_IDs'])

    print('Train set size=%i, Validation set size=%i, Test set size=%i. ' % (len(partition['train']),
                                                                             len(partition['validation']),
                                                                             len(partition['test'])))

    return partition, data_dict, running_params, model_params


def prepare_input_for_kgs_bt(model_design_key, running_params):
    test_selection = running_params['test_selection']
    train_selection = running_params['train_selection']
    n_test_set = running_params['n_test_set']
    print('Reading in data')
    all_params_ = pd.read_csv('./../data/all_maps_meta_kgs.csv')
    bottleneck_ = np.load(running_params['saved_LSR_path'])
    IDs_ = np.loadtxt('./../data/GD1_ids_list.txt', dtype=int)

    # An empty array to save the 2D histograms if needed, if not, it will remain zero
    lens_pos_ = np.zeros((len(IDs_), 20, 20, 1))

    all_params = all_params_[['ID', 'k', 'g', 's']].values
    x_params_ = np.asarray([all_params[:, 1:][all_params[:, 0] == ID][0] for ID in IDs_])

    if model_design_key == 'kgs_lens_pos_to_bt':
        all_lens_pos = read_all_lens_pos()

        # Looping through all LSRs
        for i, key in enumerate(IDs_):

            if all_lens_pos[key].shape[0] == 0:
                pass
            elif all_lens_pos[key].shape[0] != 0 and len(all_lens_pos[key].shape) == 1:
                lens_pos_[i, :, :, 0] = np.histogram2d(np.asarray([all_lens_pos[key]])[:, 0],
                                                       np.asarray([all_lens_pos[key]])[:, 1],
                                                       bins=20)[0]
            else:
                lens_pos_[i, :, :, 0] = np.histogram2d(all_lens_pos[key][:, 0],
                                                       all_lens_pos[key][:, 1],
                                                       bins=20)[0]

        print('Max of all lens pos 2D hists: ', np.max(lens_pos_))
        lens_pos_ = lens_pos_ / np.max(lens_pos_)

    if train_selection == 'k=g' or train_selection == 'retrain_k=g':
        offset = 0.3
        kappa = x_params_[:, 0]
        gamma = x_params_[:, 1]
        # Here we limit the paramater space to where gamma-offset<kappa<gamma+offset
        indx = np.where((kappa < gamma + offset) & (x_params_[:, 0] > gamma - offset))

        bottleneck = bottleneck_[indx]
        IDs = IDs_[indx]
        x_params = x_params_[indx]
        if model_design_key == 'kgs_lens_pos_to_bt':
            lens_pos = lens_pos_[indx]
    else:
        bottleneck = bottleneck_
        IDs = IDs_
        x_params = x_params_
        if model_design_key == 'kgs_lens_pos_to_bt':
            lens_pos = lens_pos_

    print('Normalizing the bottleneck by the maximum = %.5f' % np.max(bottleneck))
    bottleneck = bottleneck / np.max(bottleneck)
    n_lc = len(x_params)
    bottleneck_10 = np.zeros((n_lc * 10, 50, 50, 1))
    x_params_10 = np.zeros((n_lc * 10, 3))
    lens_pos_10 = np.zeros((n_lc * 10, 20, 20, 1))
    IDs_10 = np.zeros((n_lc * 10, 1))

    for i in range(n_lc):
        # print(i)
        for k in range(10):
            # print(k,i+i*k)
            bottleneck_10[10 * i + k] = bottleneck[i].reshape((50, 50, 1))
            x_params_10[10 * i + k] = x_params[i]
            lens_pos_10[10 * i + k] = lens_pos[i]
            IDs_10[10 * i + k] = IDs[i]

    n = len(x_params_10)
    train_n = int(0.9 * n)

    shuffler = np.random.permutation(n)
    bottleneck_10 = bottleneck_10[shuffler]
    x_params_10 = x_params_10[shuffler]
    IDs_10 = IDs_10[shuffler]
    y_train_ = x_params_10[:train_n]
    bt_train = bottleneck_10[:train_n]
    y_test_ = x_params_10[train_n:]
    bt_test_ = bottleneck_10[train_n:]

    if model_design_key == 'kgs_lens_pos_to_bt':
        lens_pos_10 = lens_pos_10[shuffler]
        lens_pos_train = lens_pos_10[:train_n]
        lens_pos_test_ = lens_pos_10[train_n:]
    else:
        lens_pos_train = np.zeros((len(bt_train)))
        lens_pos_test_ = np.zeros((len(bt_test_)))

    ids_train = IDs_10[:train_n]
    ids_test_ = IDs_10[train_n:]
    kgs_train = np.zeros(y_train_.shape)
    kgs_test_ = np.zeros(y_test_.shape)

    kgs_train[:, 0] = NormalizeData(y_train_[:, 0], np.max(y_train_[:, 0]), np.min(y_train_[:, 0]))
    kgs_train[:, 1] = NormalizeData(y_train_[:, 1], np.max(y_train_[:, 1]), np.min(y_train_[:, 1]))
    kgs_train[:, 2] = y_train_[:, 2]
    kgs_test_[:, 0] = NormalizeData(y_test_[:, 0], np.max(y_test_[:, 0]), np.min(y_test_[:, 0]))
    kgs_test_[:, 1] = NormalizeData(y_test_[:, 1], np.max(y_test_[:, 1]), np.min(y_test_[:, 1]))
    kgs_test_[:, 2] = y_test_[:, 2]

    if test_selection == 'random':
        kgs_test = kgs_test_
        bt_test = bt_test_
        lens_pos_test = lens_pos_test_
        ids_test = ids_test_
    elif test_selection == 'all_test':
        kgs_test = kgs_test_
        bt_test = bt_test_
        lens_pos_test = lens_pos_test_
        ids_test = ids_test_
    elif test_selection == 'all_train':
        kgs_test = kgs_train
        bt_test = bt_train
        lens_pos_test = lens_pos_train
        ids_test = ids_train
    elif test_selection == 'all_data':
        kgs_test = np.concatenate((kgs_test_, kgs_train), axis=0)
        bt_test = np.concatenate((bt_test_, bt_train), axis=0)
        lens_pos_test = np.concatenate((lens_pos_test_, lens_pos_train), axis=0)
        ids_test = np.concatenate((ids_test_, ids_train), axis=0)
    elif test_selection == 'sorted':
        kgs_test = kgs_test_[:n_test_set]
        bt_test = bt_test_[:n_test_set]
        lens_pos_test = lens_pos_test_[:n_test_set]
        ids_test = ids_test_[:n_test_set]
    elif test_selection == 'given':
        indexes = [True if id in running_params['test_IDs'] else False for id in IDs]
            # [np.where(IDs == id) for id in running_params['test_IDs']]
        bt_test = bottleneck[indexes]
        lens_pos_test = lens_pos[indexes]
        kgs_test = x_params[indexes]
        kgs_test[:, 0] = NormalizeData(kgs_test[:, 0], np.max(kgs_test[:, 0]), np.min(kgs_test[:, 0]))
        kgs_test[:, 1] = NormalizeData(kgs_test[:, 1], np.max(kgs_test[:, 1]), np.min(y_test_[:, 1]))
        ids_test = ids_test_[indexes]

    print('Train set size=%i, Validation set size=%i, Test set size=%i. ' % (len(bt_train),
                                                                             int(0.2 * len(bt_train)),
                                                                             len(bt_test)))
    return bt_train, bt_test, kgs_train, kgs_test, lens_pos_train, lens_pos_test, ids_train, ids_test, IDs


def set_up_model(model_design_key, model_params, running_params):
    dim = running_params['dim']
    crop_scale = model_params['crop_scale']
    res_scale = running_params['res_scale']

    if model_design_key.startswith('VAE'):
        if model_design_key == 'VAE_Unet_Resnet':
            encoder = vae_encoder(int((dim / res_scale) * crop_scale),
                                  z_size=model_params['z_size'],
                                  n_channels=model_params['n_channels'],
                                  af='relu')
        else:
            encoder = vae_encoder_3params(int((dim / res_scale) * crop_scale),
                                          input2=3,
                                          z_size=model_params['z_size'],
                                          n_channels=model_params['n_channels'],
                                          af='relu')
        model = VAE(encoder,
                    vae_decoder(model_params['z_size'], 'relu'))

    elif crop_scale != 1. and model_design_key == 'Unet2':
        model = model_params['model_function'](int((dim / res_scale) * crop_scale), first_down_sampling=4)

    elif model_design_key == 'kgs_to_bt' or \
            model_design_key == 'bt_to_kgs' or \
            model_design_key == 'kgs_lens_pos_to_bt':

        model = model_params['model_function'](model_params['input_side'],
                                               input2_shape=model_params['input_side2'])

    else:
        model = model_params['model_function'](int((dim / res_scale) * crop_scale),
                                               input2=model_params['input_side2'],
                                               n_channels=model_params['n_channels'],
                                               z_size=model_params['z_size'],
                                               flow_label=model_params['flow_label'],
                                               n_flows=model_params['n_flows'],
                                               first_down_sampling=model_params['first_down_sampling'],
                                               af='relu')

    return model


def evaluate_kgs_generated_maps(IDs, AD_model_ID, LSR_generated, running_params, metric):
    model_file = np.asarray(best_AD_models_info['job_model_filename'])[np.asarray(best_AD_models_info['job_names']) == AD_model_ID][0]
    cost_label = np.asarray(best_AD_models_info['job_cost_labels'])[np.asarray(best_AD_models_info['job_names']) == AD_model_ID][0]

    path_to_model = running_params['saved_LSR_path'].split('LSR')[0]
    model = read_AD_model(AD_model_ID, model_file, cost_label)
    dim = running_params['dim']
    output_dir = running_params['output_dir']

    if dim == 50:
        decoder_output = model.get_layer('conv2d_transpose_6').output
        decoder_input = model.get_layer('conv2d_transpose').input
    else:
        print('Unknown name of the decoder input and output layers.')
        return None
    decoder = keras.models.Model(inputs=decoder_input, outputs=decoder_output)
    map_reconstruction = decoder.predict(LSR_generated)
    maps = np.zeros((len(map_reconstruction), 1000, 1000, 1))
    metric_all = np.zeros((len(IDs)))
    for i, ID in enumerate(IDs):
        map_i = NormalizeData(np.log10(read_binary_map(ID, scaling_factor=10, to_mag=True) + 0.004))
        maps[i] = map_i
        map_new = map_reconstruction[i]
        if metric == 'ssm':
            metric_all[i] = calculate_ks_metric(map_i, map_new, test_name='anderson')

        elif metric == 'lc_sim':
            metric_all[i] = process_lc4([model + '_' + str(ID),
                                         map_i,
                                         map_new,
                                         None,
                                         1000,
                                         1000,
                                         'AD',
                                         'random',
                                         output_dir + '%s-metric_for_ID_%i' % (metric, ID),
                                         False])
    if metric == 'fid':
        inceptionv3 = read_inceptionV3()
        return process_FID([maps,
                            map_reconstruction,
                            inceptionv3])
    else:
        return np.median(metric_all)

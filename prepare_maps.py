from metric_utils import *
from plotting_utils import plot_example_lc, plot_example_conv
from maps_util import read_binary_map, read_map_meta, norm_maps, read_AD_model
from conv_utils import convolve_map, lc_gen_set, lc_gen_one
import time
import multiprocessing as mp
import pickle as pkl


def process_read(ID_):
    ID, i = ID_[0], ID_[1]
    print('Step %i' % i)
    map_ = read_binary_map(ID)
    mag_convertor = read_map_meta(ID)[0]
    map_ = norm_maps(map_,
                     mag_convertor)

    return map_


def process_conv(ls):
    map_ = ls[0]
    rsrc = ls[1]
    map_conv = convolve_map(map_,
                            rsrc)

    return map_conv


def process_AD_predict_one_map(input):
    AD = input[0]
    map_ = input[1]
    shapes = map_.shape
    AD_maps = AD.predict(map_.reshape((1, shapes[0], shapes[0], 1)))
    AD_maps = AD_maps.reshape((shapes[0], shapes[0]))
    return AD_maps


def process_save_LSR_one_map(input):
    LSR_layer = input[0]
    map_ = input[1]
    shapes = map_.shape
    AD_LSR = LSR_layer.predict(map_.reshape((1, shapes[0], shapes[0], 1)))
    AD_LSR = AD_LSR.reshape((AD_LSR.shape[1], AD_LSR.shape[1]))
    return AD_LSR


def process_lc(ls):
    map_ = ls[0]
    n_lc = ls[1]
    (lc_set, origin_end) = lc_gen_set(n_lc,
                                      map_,
                                      nxpix=1000,
                                      nypix=1000,
                                      factor=1,
                                      tot_r_e=12.5,
                                      nsamp=100,
                                      distance=10,  # in maps units
                                      dt=(22 * 365) / 1.5,
                                      set_seed=1)

    return [lc_set, origin_end]


def process_lc2(ls, label):
    # map_ID = ls[0]
    map_ = ls[0]
    n_lc = ls[1]
    lc_picked = ls[2]

    (lc_set, origin_end) = lc_gen_set(n_lc,
                                      map_,
                                      nxpix=1000,
                                      nypix=1000,
                                      factor=1,
                                      tot_r_e=12.5,
                                      nsamp=100,
                                      distance=10,  # in maps units
                                      dt=(22 * 365) / 1.5,
                                      set_seed=np.nan)

    # np.save('./../../../fred/oz108/skhakpas/results/save_lcs_lc_distance_metric/lc_Set_%s' % label, lc_set)
    # np.save('./../../../fred/oz108/skhakpas/results/save_lcs_lc_distance_metric/or_end_Set_%s' % label, origin_end)

    # print(lc_set[:, 1, :])

    return [process_fit_lc2([lc_picked[i], lc_set[:, 1, :]]) for i in range(len(lc_picked))]


def process_lc2b(ls):
    lc_picked = ls[0]
    lcs = ls[1]

    return [process_fit_lc2([lc_picked[i], lcs]) for i in range(len(lc_picked))]


def process_lc3(input):
    import tensorflow, keras

    iter, ID, rsrcs, steps, num_lc, model_params, mode_select, output_directory = input[0], input[1], input[2], input[
        3], input[4], input[5], input[6], input[7]
    # print('Iteration %i'%iter)
    if isinstance(rsrcs, list):
        rsrcs_len = 1
    else:
        rsrcs_len = len(rsrcs) + 1

    if isinstance(ID, list):
        ID0 = ID[0]
        ID1 = ID[1]
        map_0 = process_read([ID0, ID0])
        map_ = process_read([ID1, ID1])
    else:
        ID0 = ID
        map_0 = process_read([ID0, ID0])
        map_ = map_0

    if mode_select == 'all':
        modes = ['true', 'true_conv', 'AD', 'AD_conv']
    elif mode_select == 'trues':
        modes = ['true', 'true_conv']
    elif mode_select == 'ADs':
        modes = ['AD', 'AD_conv']
    elif mode_select == 'true':
        modes = ['true']

    lcs_picked_conv = {}
    # print('Generating %i picked light curves for map ID= %i' % (steps, ID0))
    seedd = 0
    lcs_picked = np.array([lc_gen_one(map_0, seedd + i) for i in range(steps)])
    # np.save('./../../../fred/oz108/skhakpas/results/save_lcs_lc_distance_metric/lcs_picked', lcs_picked)
    all_lc_fit_lc_metric_all_map = np.zeros((2, rsrcs_len, steps), dtype=np.float16)
    # all_AD_lc_fit_lc_metric_all_map = np.zeros((rsrcs_len, steps), dtype=np.float16)
    # print(lcs_picked)
    for mode in modes:
        # print('Doing mode %s' % mode)

        if len(mode.split('_')) > 1:
            if not bool(lcs_picked_conv):
                for r, rsrc in enumerate(rsrcs):
                    map_conv_0 = process_conv([map_0, rsrc])
                    if isinstance(ID, list):
                        map_conv = process_conv([map_, rsrc])
                    else:
                        map_conv = map_conv_0
                    seedd = 0
                    lcs_picked_conv[str(rsrc)] = np.array(
                        [lc_gen_one(map_conv_0, seedd + i) for i in range(steps)])
                    # np.save('./../../../fred/oz108/skhakpas/results/save_lcs_lc_distance_metric/lcs_picked_conv_%s' % (
                    #     str(rsrc)), lcs_picked_conv[str(rsrc)])

        if mode == 'true':

            all_lc_fit_lc_metric_all_map[0, 0, :] = process_lc2([map_, num_lc, lcs_picked])
            # print(all_lc_fit_lc_metric_all_map[0, 0,:])
        elif mode == 'true_conv':
            for r, rsrc in enumerate(rsrcs):
                all_lc_fit_lc_metric_all_map[0, r + 1, :] = process_lc2([map_conv, num_lc,
                                                                         lcs_picked_conv[str(rsrc)]])

        elif mode.startswith('AD'):
            # if np.isnan(autoencoder):
            try:
                print(autoencoder)
            except NameError:
                autoencoder = read_AD_model(model_params[0],
                                            model_params[1],
                                            model_params[2])
                map_AD = process_AD_predict_one_map([autoencoder, map_])
            if mode == 'AD':
                all_lc_fit_lc_metric_all_map[1, 0, :] = process_lc2([map_AD, num_lc, lcs_picked])
            elif mode == 'AD_conv':
                for r, rsrc in enumerate(rsrcs):
                    map_AD_conv = process_conv([map_AD, rsrc])
                    all_lc_fit_lc_metric_all_map[1, r + 1, :] = process_lc2([map_AD_conv, num_lc,
                                                                             lcs_picked_conv[
                                                                                 str(rsrc)]])
    np.save(output_directory, all_lc_fit_lc_metric_all_map)


def process_lc4(input):
    iter, maps, map_AD, rsrcs, steps, num_lc, mode_select, mode_rand, output_directory, save_output = input[0], input[
        1], input[2], \
                                                                                                      input[3], input[
                                                                                                          4], input[5], \
                                                                                                      input[6], input[
                                                                                                          7], input[8], \
                                                                                                      input[9]
    # print('Iteration %i'%iter)
    model_ID = iter.split('_')[0]
    map_ID = iter.split('_')[1]
    if isinstance(rsrcs, list):
        rsrcs_len = 1
    else:
        rsrcs_len = len(rsrcs) + 1

    if isinstance(maps, list):
        map_0 = maps[0]
        map_ = maps[1]
    else:
        map_0 = maps
        map_ = map_0

    if mode_select == 'all':
        modes = ['true', 'true_conv', 'AD', 'AD_conv']
    elif mode_select == 'trues':
        modes = ['true', 'true_conv']
    elif mode_select == 'ADs':
        modes = ['AD', 'AD_conv']
    elif mode_select == 'true':
        modes = ['true']
    elif mode_select == 'AD':
        modes = ['AD']

    lcs_picked_conv = {}
    # print('Generating %i picked light curves for map ID= %i' % (steps, ID0))
    if mode_rand == 'random':
        seedd = np.nan
        lcs_picked = np.array([lc_gen_one(map_0, seedd) for i in range(steps)])
    elif mode_rand == 'fixed':
        seedd = 0
        lcs_picked = np.array([lc_gen_one(map_0, seedd + i) for i in range(steps)])
    all_lc_fit_lc_metric_all_map = np.zeros((2, rsrcs_len, steps), dtype=np.float16)
    np.save('./../../../fred/oz108/skhakpas/results/save_lcs_lc_distance_metric/lcs_picked_%s' % map_ID, lcs_picked)
    lcs_picked = [lcs_picked[i][0] for i in range(len(lcs_picked))]

    for mode in modes:
        # print('Doing mode %s' % mode)

        if len(mode.split('_')) > 1:
            if not bool(lcs_picked_conv):
                map_conv = []
                for r, rsrc in enumerate(rsrcs):
                    map_conv_0 = process_conv([map_0, rsrc])
                    if isinstance(maps, list):
                        map_conv.append(process_conv([map_, rsrc]))
                    else:
                        map_conv.append(map_conv_0)
                    seedd = 0
                    lcs_picked_conv[str(rsrc)] = np.array(
                        [lc_gen_one(map_conv_0, seedd + i) for i in range(steps)])
                    np.save(
                        './../../../fred/oz108/skhakpas/results/save_lcs_lc_distance_metric/lcs_picked_%s_conv_%s' % (
                            map_ID, str(rsrc)), lcs_picked_conv[str(rsrc)])
                    lcs_picked_conv[str(rsrc)] = [lcs_picked_conv[str(rsrc)][i][0] for i in
                                                  range(len(lcs_picked_conv[str(rsrc)]))]

        if mode == 'true':

            all_lc_fit_lc_metric_all_map[0, 0, :] = process_lc2([map_, num_lc, lcs_picked], mode + '_ID%s' % (map_ID))
            # print(all_lc_fit_lc_metric_all_map[0, 0,:])
        elif mode == 'true_conv':
            for r, rsrc in enumerate(rsrcs):
                all_lc_fit_lc_metric_all_map[0, r + 1, :] = process_lc2([map_conv[r], num_lc,
                                                                         lcs_picked_conv[str(rsrc)]],
                                                                        mode + '_%s_ID%s' % (rsrc, map_ID))
        elif mode.startswith('AD'):
            if mode == 'AD':
                all_lc_fit_lc_metric_all_map[1, 0, :] = process_lc2([map_AD, num_lc, lcs_picked],
                                                                    mode + '_' + str(iter))
            elif mode == 'AD_conv':
                for r, rsrc in enumerate(rsrcs):
                    map_AD_conv = process_conv([map_AD, rsrc])
                    all_lc_fit_lc_metric_all_map[1, r + 1, :] = process_lc2([map_AD_conv, num_lc,
                                                                             lcs_picked_conv[str(rsrc)]],
                                                                            mode + '_' + str(iter) + '_%s' % rsrc)
    if save_output:
        np.save(output_directory, all_lc_fit_lc_metric_all_map)


def prepare_maps(list_ID,
                 rsrcs,
                 model_ID,
                 model_file,
                 cost_label,
                 num_lc,
                 output_direc,
                 conv_AD_maps=True,
                 plot_exp_conv=False,
                 gen_lc=True,
                 plot_exp_lc=False,
                 save_maps=True,
                 save_lcs=True,
                 verbose=False):
    num_cores = mp.cpu_count()
    num_to_process = min(num_cores, len(list_ID))
    pool = mp.Pool(processes=num_to_process)
    print('Number of cores found: ', mp.cpu_count())

    maps_dict = {'ID': np.asarray(list_ID)}
    AD_maps_dict = {'ID': np.asarray(list_ID)}

    lcs_dict = {'ID': np.asarray(list_ID)}
    AD_lcs_dict = {'ID': np.asarray(list_ID)}

    if verbose:
        print('Reading all the true maps...')
    temp0 = pool.map(process_read,
                     [[ID, i] for i, ID in enumerate(list_ID)])
    maps_dict['true_maps'] = np.asarray(temp0)

    if len(rsrcs) != 0:
        for r, rsrc in enumerate(rsrcs):
            if verbose:
                print('Convolving the true maps with a source size of %.1f R_E.' % rsrc)

            temp = pool.map(process_conv,
                            [[maps_dict['true_maps'][i], rsrc] for i in range(len(list_ID))])
            maps_dict[str(rsrc)] = np.asarray(temp)

            if plot_exp_conv:
                ID1 = list_ID[0]
                map1 = maps_dict['true_maps'][maps_dict['ID'] == ID1][0]
                map1_conv = maps_dict[str(rsrc)][maps_dict['ID'] == ID1][0]

                plot_example_conv(map1, map1_conv,
                                  output_direc + '%i_%.1f_%s_exp.png' % (list_ID[0],
                                                                         rsrc,
                                                                         model_ID))
            if gen_lc:
                if verbose:
                    print('Generating %i lightcurves for the true maps convolved with a source size of %.1f R_E.' % (
                        num_lc, rsrc))
                start_time = time.time()
                lc_temp = pool.map(process_lc,
                                   [[maps_dict[str(rsrc)][i], num_lc] for i in range(len(list_ID))])

                if r == 0:
                    if verbose:
                        print('Generating %i lightcurves for the unconvolved true maps' % num_lc)
                    lc_temp_true = pool.map(process_lc,
                                            [[maps_dict['true_maps'][i], num_lc] for i in range(len(list_ID))])
                    lcs_dict['true_lcs'] = np.asarray([lc_temp_true[i][0] for i in range(len(list_ID))])
                    lcs_dict['true_lcs_orig'] = np.asarray([lc_temp_true[i][1] for i in range(len(list_ID))])

                lcs_dict['conv_lcs_' + str(rsrc)] = np.asarray([lc_temp[i][0] for i in range(len(list_ID))])
                lcs_dict['conv_lcs_orig_' + str(rsrc)] = np.asarray([lc_temp[i][0] for i in range(len(list_ID))])

                end_time = time.time()  # Record the end time
                total_runtime = end_time - start_time
                if verbose:
                    print('time for %s lightcurves at source size %.1f is %.1f seconds' % (num_lc, rsrc, total_runtime))

                if plot_exp_lc:
                    if verbose:
                        print('Generating example plots of the produced lightcurves for rsrc=%.1f' % rsrc)
                    ID1 = list_ID[0]
                    map1 = maps_dict['true_maps'][maps_dict['ID'] == ID1][0]
                    map1_conv = maps_dict[str(rsrc)][maps_dict['ID'] == ID1][0]
                    lc_set1 = lcs_dict['true_lcs'][lcs_dict['ID'] == ID1][0][0]
                    lc_set2 = lcs_dict['conv_lcs_' + str(rsrc)][lcs_dict['ID'] == ID1][0][0]
                    origin1 = lcs_dict['true_lcs_orig'][lcs_dict['ID'] == ID1][0][0]
                    origin2 = lcs_dict['conv_lcs_orig_' + str(rsrc)][lcs_dict['ID'] == ID1][0][0]
                    lc_direc = output_direc + 'map_%i_%.1f_%s_exp_lc.png' % (list_ID[0], rsrc, model_ID)
                    plot_example_lc(list_ID[0],
                                    [map1, map1_conv],
                                    [lc_set1, lc_set2],
                                    [origin1, origin2],
                                    rsrc,
                                    lc_direc)
    else:

        if gen_lc:
            if verbose:
                print('Generating %i lightcurves for the unconvolved true maps' % (num_lc))

            lc_temp_true = pool.map(process_lc,
                                    [[maps_dict['true_maps'][i], num_lc] for i in range(len(list_ID))])
            lcs_dict['true_lcs'] = np.asarray([lc_temp_true[i][0] for i in range(len(list_ID))])
            lcs_dict['true_lcs_orig'] = np.asarray([lc_temp_true[i][1] for i in range(len(list_ID))])

    if save_maps:
        if verbose:
            print('Saving the maps...')
        pkl.dump(maps_dict,
                 open(output_direc + 'true_maps.pkl', 'wb'))
    if gen_lc and save_lcs:
        if verbose:
            print('Saving the lightcurves...')
        pkl.dump(lcs_dict,
                 open(output_direc + 'true_lcs.pkl', 'wb'))

    if conv_AD_maps:
        if verbose:
            print('Predicting AD maps for model %s.' % model_ID)
        normed_maps = maps_dict['true_maps']
        # print(normed_maps.shape)
        shape_ = normed_maps[0].shape[0]
        autoencoder = read_AD_model(model_ID, model_file, cost_label)
        AD_maps = autoencoder.predict(normed_maps.reshape((len(list_ID), shape_, shape_, 1)))
        AD_maps = AD_maps.reshape((len(list_ID), shape_, shape_))

        AD_maps_dict['AD_maps'] = AD_maps

        if len(rsrcs) != 0:

            for r, rsrc in enumerate(rsrcs):

                if verbose:
                    print('Convolving AD maps for model %s with a source size of %.1f.' % (model_ID, rsrc))

                temp = pool.map(process_conv,
                                [[AD_maps[i], rsrc] for i in range(len(list_ID))])
                AD_maps_dict[str(rsrc)] = np.asarray(temp)

                if plot_exp_conv:
                    ID1 = list_ID[0]
                    map1 = AD_maps_dict['AD_maps'][AD_maps_dict['ID'] == ID1][0]
                    map1_conv = AD_maps_dict[str(rsrc)][AD_maps_dict['ID'] == ID1][0]

                    plot_example_conv(map1, map1_conv,
                                      output_direc + '%i_%.1f_%s_AD_exp.png' % (list_ID[0],
                                                                                rsrc,
                                                                                model_ID))

                if gen_lc:
                    if verbose:
                        print('Generating %i lightcurves for the convolved AD maps with rsrc=%0.1f' % (num_lc, rsrc))

                    AD_lc_temp = pool.map(process_lc,
                                          [[AD_maps_dict[str(rsrc)][i], num_lc] for i in range(len(list_ID))])

                    if r == 0:
                        if verbose:
                            print('Generating %i lightcurves for the unconvolved AD maps' % (num_lc))
                        AD_lc_temp_true = pool.map(process_lc,
                                                   [[AD_maps_dict['AD_maps'][i], num_lc] for i in range(len(list_ID))])
                        AD_lcs_dict['AD_lcs'] = np.asarray([AD_lc_temp_true[i][0] for i in range(len(list_ID))])
                        AD_lcs_dict['AD_lcs_orig'] = np.asarray([AD_lc_temp_true[i][1] for i in range(len(list_ID))])

                    AD_lcs_dict['conv_lcs_' + str(rsrc)] = np.asarray(
                        [AD_lc_temp_true[i][0] for i in range(len(list_ID))])
                    AD_lcs_dict['conv_lcs_orig_' + str(rsrc)] = np.asarray(
                        [AD_lc_temp_true[i][0] for i in range(len(list_ID))])

                    if plot_exp_lc:
                        ID1 = list_ID[0]
                        map1 = AD_maps_dict['AD_maps'][AD_maps_dict['ID'] == ID1][0]
                        map1_conv = AD_maps_dict[str(rsrc)][AD_maps_dict['ID'] == ID1][0]
                        lc_set1 = AD_lcs_dict['AD_lcs'][AD_lcs_dict['ID'] == ID1][0][0]
                        lc_set2 = AD_lcs_dict['conv_lcs_' + str(rsrc)][AD_lcs_dict['ID'] == ID1][0][0]
                        origin1 = AD_lcs_dict['AD_lcs_orig'][AD_lcs_dict['ID'] == ID1][0][0]
                        origin2 = AD_lcs_dict['conv_lcs_orig_' + str(rsrc)][AD_lcs_dict['ID'] == ID1][0][0]
                        lc_direc = output_direc + 'AD_map_%i_%.1f_%s_exp_lc.png' % (list_ID[0], rsrc, model_ID)

                        plot_example_lc(list_ID[0],
                                        [map1, map1_conv],
                                        [lc_set1, lc_set2],
                                        [origin1, origin2],
                                        rsrc,
                                        lc_direc)
        else:
            if gen_lc:
                if verbose:
                    print('Generating %i lightcurves for the unconvolved AD maps' % num_lc)
                AD_lc_temp_true = pool.map(process_lc,
                                           [[AD_maps_dict['true_maps'][i], num_lc] for i in range(len(list_ID))])
                AD_lcs_dict['AD_lcs'] = np.asarray([AD_lc_temp_true[i][0] for i in range(len(list_ID))])
                AD_lcs_dict['AD_lcs_orig'] = np.asarray([AD_lc_temp_true[i][1] for i in range(len(list_ID))])

        if save_maps:
            pkl.dump(AD_maps_dict,
                     open(output_direc + 'AD_maps_%s.pkl' % model_ID, 'wb'))
        if save_maps and save_lcs:
            pkl.dump(AD_lcs_dict,
                     open(output_direc + 'AD_lcs_%s.pkl' % model_ID, 'wb'))

    return maps_dict, \
               AD_maps_dict, \
               lcs_dict, \
               AD_lcs_dict

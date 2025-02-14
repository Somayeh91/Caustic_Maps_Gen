import multiprocessing as mmp
from multiprocessing import Pool
import argparse
from more_info import best_AD_models_info
from prepare_maps import *
from plotting_utils import *
from conv_utils import *
from metric_utils import *
from maps_util import *
from FID_calculator import *
import keras


def parse_options():
    """Function to handle options speficied at command line
    """
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('-output_directory', action='store',
                        default='./../../../fred/oz108/skhakpas/results/',
                        help='Specify the directory where the results should be stored.')
    parser.add_argument('-input_directory', action='store',
                        default='./../../../fred/oz108/skhakpas/results/',
                        help='Specify the directory where the results should be stored.')
    parser.add_argument('-list_IDs_directory', action='store',
                        default='./../data/GD1_ids_list.txt',
                        help='Specify the directory where the list of map IDs is stored.')
    parser.add_argument('-rsrc', action='store',
                        default='0.1,0.5',
                        help='A list of source sizes for convolution, separated by commas.')
    parser.add_argument('-input_map_size', action='store',
                        default=10000,
                        help='size of each side the original maps in pixels.')
    parser.add_argument('-output_map_size', action='store',
                        default=1000,
                        help='size of each side the maps with reduced resolution in pixels.')
    parser.add_argument('-AD_model_ID', action='store',
                        default='23-12-14-01-12-07',
                        help='AD model name.')
    parser.add_argument('-AD_model_file_name', action='store',
                        default='model_10000_8_0.0001',
                        help='AD model file name.')
    parser.add_argument('-AD_model_cost_label', action='store',
                        default='custom',
                        help='AD model cost label.')
    parser.add_argument('-LSR_size', action='store',
                        default=50,
                        help='dimension of the latent space representation')
    parser.add_argument('-num_lc', action='store',
                        default=1,
                        help='If gen_lc True, we need to define the number of lc.')
    parser.add_argument('-saved_maps_dir', action='store',
                        default='',
                        help='If read_conv is False, you need to give and address to where the saved maps are.')
    parser.add_argument('-FID_metric_calc', action='store',
                        default=False,
                        help='Do you want to calculate FID_metric?')
    parser.add_argument('-FID_metric_plot', action='store',
                        default=False,
                        help='Do you want to plot the FID_metric?')
    parser.add_argument('-KS_metric_calc', action='store',
                        default=False,
                        help='Do you want to calculate KS_metric?')
    parser.add_argument('-KS_test_name', action='store',
                        default='KS',
                        help='What similarty test? KS or anderson')
    parser.add_argument('-fit_lc_metric_calc', action='store',
                        default=False,
                        help='Do you want to calculate fit_lc_metric?')
    parser.add_argument('-fit_lc_metric_plot', action='store',
                        default=False,
                        help='Do you want to plot examples of the fit_lc_metric?')
    parser.add_argument('-kg_fit_lc_metric_plot', action='store',
                        default=False,
                        help='Do you want to plot the gamma-kappa fit_lc_metric?')
    parser.add_argument('-KS_metric_plot', action='store',
                        default=False,
                        help='Do you want to plot the KS_metric?')
    parser.add_argument('-FID_metric_per_rsrc_plot', action='store',
                        default=False,
                        help='Do you want to plot the FID_metric  per rsrc?')
    parser.add_argument('-KS_metric_per_rsrc_plot', action='store',
                        default=False,
                        help='Do you want to plot the KS_metric  per rsrc?')
    parser.add_argument('-lc_distance_metric', action='store',
                        default=False,
                        help='Do you want to get the lc metric with uncertainty?')
    parser.add_argument('-per_map', action='store',
                        default=True,
                        help='Running mode of lc_distance_metric.')
    parser.add_argument('-cross_map', action='store',
                        default=False,
                        help='Running mode of lc_distance_metric.')
    parser.add_argument('-lc_distance_per_map100_for_saved_lc', action='store',
                        default=False,
                        help='Do you want to get the lc metric with uncertainty for the saved lcs of 100 maps?')
    parser.add_argument('-save_lcs', action='store',
                        default=False,
                        help='If true, save the generated lightcurves.')
    parser.add_argument('-save_maps', action='store',
                        default=False,
                        help='If true, save the generated lightcurves.')
    parser.add_argument('-date', action='store',
                        default='',
                        help='Date and time of the job run.')
    parser.add_argument('-verbose', action='store',
                        default=False,
                        help='Set verbose.')
    parser.add_argument('-gen_ID_list', action='store',
                        default=False,
                        help='Do you want to generate a list of random IDs or read from an existing list?')
    parser.add_argument('-gen_lc', action='store',
                        default=False,
                        help='Do you just want to generate lightcurves for a list of maps and save them?')
    parser.add_argument('-saved_before', action='store',
                        default=False,
                        help='Do you want to save the results in a directory that already exists?')
    parser.add_argument('-save_LSR', action='store',
                        default=False,
                        help='Do you want to save the latent space of an AD model?')
    parser.add_argument('-uncertainty_calculater_steps', action='store',
                        default=1000,
                        help='Do you want to save the results in a directory that already exists?')
    parser.add_argument('-to_mag', action='store',
                        default=True,
                        help='Do you want to keep maps in units of magnification or ray counts? Note that the units '
                             'of light curves will be changed as well.')
    parser.add_argument('-norm_map_values', action='store',
                        default=True,
                        help='Do you want to normalize map values? It is recommended for training machine learning. '
                             'Note that light curve units will be affected as well.')

    # Parses through the arguments and saves them within the keyword args
    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    start_time = time.time()
    print('Setting up the initial parameters...')
    args = parse_options()
    gen_ID_list = args.gen_ID_list
    gen_lc = args.gen_lc
    saved_before = args.saved_before
    save_LSR = args.save_LSR
    if gen_ID_list:
        list_ID = eval_maps_selection(num=12342, seed=33)
    else:
        list_ID = np.loadtxt(args.list_IDs_directory, dtype=int)

    output_direc = args.output_directory
    input_direc = args.input_directory
    input_map_size = int(args.input_map_size)
    output_map_size = int(args.output_map_size)
    model_ID = args.AD_model_ID
    model_file = args.AD_model_file_name
    cost_label = args.AD_model_cost_label
    steps = int(args.uncertainty_calculater_steps)
    FID_metric_calc = args.FID_metric_calc
    KS_metric_calc = args.KS_metric_calc
    KS_test_name = args.KS_test_name
    fit_lc_metric_calc = args.fit_lc_metric_calc
    num_lc = int(args.num_lc)
    LSR_size = int(args.LSR_size)
    FID_metric_plot = args.FID_metric_plot
    KS_metric_plot = args.KS_metric_plot
    FID_metric_per_rsrc_plot = args.FID_metric_per_rsrc_plot
    KS_metric_per_rsrc_plot = args.KS_metric_per_rsrc_plot
    fit_lc_metric_plot = args.fit_lc_metric_plot
    kg_fit_lc_metric_plot = args.kg_fit_lc_metric_plot
    lc_distance_metric = args.lc_distance_metric
    per_map = args.per_map
    cross_map = args.cross_map
    lc_distance_per_map100_for_saved_lc = args.lc_distance_per_map100_for_saved_lc
    verbose = args.verbose
    rsrcs = np.asarray([float(rsrc) for rsrc in args.rsrc.split(',')])
    n_models = len(best_AD_models_info['data'])
    models = best_AD_models_info['job_names']
    model_files = best_AD_models_info['job_model_filename']
    cost_labels = best_AD_models_info['job_cost_labels']
    LSR_layer_names = best_AD_models_info['LSR_layer_name']
    FID_metrics = np.zeros((n_models, len(rsrcs) + 1))
    FID_metrics_unc = np.zeros((n_models, len(rsrcs) + 1, steps))
    KS_metrics = np.zeros((n_models, len(rsrcs) + 1))
    KS_metrics_unc = np.zeros((n_models, len(rsrcs) + 1, len(list_ID)))
    fit_lc_metrics = np.zeros((n_models + 1, len(rsrcs) + 1, len(list_ID)))
    fit_lc_per_map_metrics = np.zeros((n_models + 1, len(rsrcs) + 1, len(list_ID)))
    to_mag = args.to_mag
    norm_map_values = args.norm_map_values

    all_lc_fit_lc_metric_all_map = np.zeros((n_models + 1, len(rsrcs) + 1, len(list_ID), steps))
    shape_ = 1000
    save_lcs = args.save_lcs
    save_maps = args.save_maps
    date = args.date

    if not saved_before:
        os.system("mkdir " + str(output_direc) + str(date))
        output_direc = output_direc + str(date) + '/'
    rsrc_for_plot = 0

    if save_LSR:
        LSR_layer_names_tmp = LSR_layer_names[models == model_ID]
        autoencoder = read_AD_model(model_ID,
                                    model_file,
                                    cost_label)
        bottleneck_output = autoencoder.get_layer(LSR_layer_names_tmp).output
        model_bottleneck = keras.models.Model(inputs=autoencoder.input, outputs=bottleneck_output)
        LSR = np.array(
            [process_save_LSR_one_map([model_bottleneck, process_read([ID_, input_map_size, output_map_size, True, True])]) for i, ID_ in enumerate(list_ID)])
        np.save(output_direc + 'LSR_%s_samplesize_%i' % (model_ID, len(list_ID)), LSR)

    if gen_lc:
        num_cores = 64  # mp.cpu_count()
        # num_to_process = min(num_cores, len(list_ID))
        print('Using %i cores.' % num_cores)

        # pool = mmp.Pool(processes=num_to_process)
        print('Set up the pool..')

        '''We define four steps for each map:
        1) (mode == 'true') we use the true version, generate num_lc lightcurves compare to
        num_lc_picked number of picked lcs and record the min
        2) (mode == 'true_conv') we repeat the same process for the convolved version of the same map
        3) (mode == 'AD') we repeat the same process for the AD version of the same map in step 1
        4) (mode == 'AD_conv') we repeat the same process for the convolved version of the AD map in step 3'''

        modes = ['true', 'true_conv', 'AD', 'AD_conv']
        lcs_picked_conv = {}
        print('setting up the list ID...')
        # list_ID = np.array([23350, 43542, 46645, 48465, 46035])  # list_ID[:]
        # (0.3, 0.5, 0.5), (0.2, 1.5, 0.5), (1, 0.6, 0.5), (1.6, 0.2, 0.5), (1.6, 1.5, 0.5)

        if len(list_ID) % num_cores == 0:
            num_processes = int(len(list_ID) / num_cores)
        else:
            num_processes = int(len(list_ID) / num_cores) + 1
        m = -1
        model = models[m]
        print('Running for model %s:' % model)
        model_param = [model,
                       model_files[m],
                       cost_labels[m]]
        autoencoder = read_AD_model(model_param[0],
                                    model_param[1],
                                    model_param[2])
        mode_rand = 'fixed'
        all_lc = []
        for n in range(num_processes):
            print('Batch %i/%i of %i maps:' % (n, num_processes, num_cores))
            if n == num_processes - 1:
                list_ID_tmp = list_ID[n * num_cores:]
            else:
                list_ID_tmp = list_ID[n * num_cores:(n + 1) * num_cores]
            with Pool(processes=num_cores) as pool:
                maps = pool.map(process_read,
                                [[ID,
                                  input_map_size,
                                  output_map_size,
                                  to_mag,
                                  norm_map_values] for i, ID in enumerate(list_ID_tmp)])
                maps_tmp = maps




            map_AD = [process_AD_predict_one_map([autoencoder, map_]) for map_ in maps]
            output_direc2 = output_direc + 'model_%s_' % model
            mode_select = 'all'
            all_maps = {'maps': maps, 'AD_maps': map_AD, 'ID': list_ID_tmp}
            # pkl.dump(all_maps, open(output_direc + 'saved_maps_batch_%i.pkl' % n,
            #                         'wb'))

            with Pool(processes=num_cores) as pool:
                tmp = pool.map(process_lc_gen,
                               [[model + '_' + str(list_ID_tmp[i]),
                                 maps_tmp_,
                                 map_AD[i],
                                 rsrcs,
                                 num_lc,
                                 mode_select,
                                 mode_rand,
                                 output_direc2 + '_ID_%i' % list_ID_tmp[i],
                                 True] for i, maps_tmp_ in
                                enumerate(maps_tmp)])
            # all_lc += tmp
            # dict_temp = dict(zip([str(id) for id in list_ID_tmp],
            #                      all_lc))

        # np.save(output_direc + 'saved_lcs_%s_720_maps' % (mode_rand), all_lc)





    if FID_metric_calc:
        num_cores = 20
        print('Using %i cores.' % num_cores)
        # list_ID = list_ID[:2]
        inceptionv3 = read_inceptionV3()
        # else:
        list_ID = pd.read_csv('./../data/all_maps_meta_kgs.csv')['ID'].values

        ADs = []

        for m, model in enumerate(models):
            autoencoder = read_AD_model(model,
                                        model_files[m],
                                        cost_labels[m])
            ADs.append(autoencoder)
        for i in range(steps):
            print('Step %i/%i of %i maps:' % (i, steps, num_cores))
            ID_set = random.sample(list(list_ID), num_cores)
            with Pool(processes=num_cores) as pool:
                maps = pool.map(process_read, [[ID,
                                                input_map_size,
                                                output_map_size,
                                                to_mag,
                                                norm_map_values] for i, ID in enumerate(ID_set)])
            for m, model in enumerate(models):
                print('Model %s:' % model)
                AD_maps = np.array([process_AD_predict_one_map([ADs[m], map_]) for map_ in maps])
                FID_metrics_unc[m, 0, i * num_cores:(i + 1) * num_cores] = process_FID([np.asarray(maps),
                                                                                        AD_maps,
                                                                                        inceptionv3])
                for r, rsrc in enumerate(rsrcs):
                    maps_conv = np.array([process_conv([map_, rsrc]) for map_ in maps])
                    AD_maps_conv = np.array([process_conv([map_, rsrc]) for map_ in AD_maps])

                    FID_metrics_unc[m, r + 1, i * num_cores:(i + 1) * num_cores] = process_FID([maps_conv,
                                                                                                AD_maps_conv,
                                                                                                inceptionv3])
        np.save(output_direc + 'FID_metric_unc_%i_steps'%steps, FID_metrics_unc)

        FID_metrics = np.zeros((len(FID_metrics_unc), 2 * (len(rsrcs) + 1)))
        for i in range(len(FID_metrics_unc)):
            for j in range(len(rsrcs) + 1):
                FID_metrics[i, 2 * j] = np.mean(FID_metrics_unc[i, j, :])
                FID_metrics[i, 2 * j + 1] = np.std(FID_metrics_unc[i, j, :])
                FID_metrics[i, 2 * j + 2] = np.median(FID_metrics_unc[i, j, :])
                FID_metrics[i, 2 * j + 3] = np.percentile(FID_metrics_unc[i, j, :], 16)
                FID_metrics[i, 2 * j + 4] = np.percentile(FID_metrics_unc[i, j, :], 84)

        if FID_metric_plot:
            imgs = img_cut_out_generator(43657, rsrc=0, size=200)
            metric_comparison_plot(FID_metrics[:, :3],
                                   'FID',
                                   imgs,
                                   output_direc,
                                   mode='no_unc')
        if FID_metric_per_rsrc_plot:
            metric_comparison_per_rsrc_plot(np.concatenate(([0], rsrcs), axis=0),
                                            FID_metrics,
                                            'FID',
                                            output_direc,
                                            mode='unc')

    if KS_metric_calc:

        num_cores = 20
        print('Using %i cores.' % num_cores)
        # list_ID = list_ID[:2]

        if len(list_ID) % num_cores == 0:
            num_processes = int(len(list_ID) / num_cores)
        else:
            num_processes = int(len(list_ID) / num_cores) + 1

        ADs = []

        for m, model in enumerate(models):
            autoencoder = read_AD_model(model,
                                        model_files[m],
                                        cost_labels[m])
            ADs.append(autoencoder)

        for n in range(num_processes):
            print('Batch %i/%i of %i maps:' % (n, num_processes, num_cores))
            if n == num_processes - 1:
                list_ID_tmp = list_ID[n * num_cores:]
            else:
                list_ID_tmp = list_ID[n * num_cores:(n + 1) * num_cores]
            with Pool(processes=num_cores) as pool:
                maps = pool.map(process_read, [[ID,
                                                input_map_size,
                                                output_map_size,
                                                to_mag,
                                                norm_map_values] for i, ID in enumerate(list_ID_tmp)])

            for m, model in enumerate(models):
                print('Model %s:' % model)
                AD_maps = np.array([process_AD_predict_one_map([ADs[m], map_]) for map_ in maps])
                if n == num_processes - 1:
                    with Pool(processes=num_cores) as pool:
                        KS_metrics_unc[m, 0, n * num_cores:] = pool.map(process_KS,
                                                                        [[maps[i].flatten(),
                                                                          AD_maps[i].flatten(),
                                                                          KS_test_name] for i in
                                                                         range(len(list_ID_tmp))])
                else:
                    with Pool(processes=num_cores) as pool:
                        KS_metrics_unc[m, 0, n * num_cores:(n + 1) * num_cores] = pool.map(process_KS,
                                                                                           [[maps[i].flatten(),
                                                                                             AD_maps[i].flatten(),
                                                                                             KS_test_name] for i in
                                                                                            range(len(list_ID_tmp))])

                for r, rsrc in enumerate(rsrcs):
                    maps_conv = np.array([process_conv([map_, rsrc]) for map_ in maps])
                    AD_maps_conv = np.array([process_conv([map_, rsrc]) for map_ in AD_maps])
                    if n == num_processes - 1:
                        with Pool(processes=num_cores) as pool:
                            KS_metrics_unc[m, r + 1, n * num_cores:] = pool.map(process_KS,
                                                                                [[maps_conv[i].flatten(),
                                                                                  AD_maps_conv[i].flatten(),
                                                                                  KS_test_name] for i in
                                                                                 range(len(list_ID_tmp))])
                    else:
                        with Pool(processes=num_cores) as pool:
                            KS_metrics_unc[m, r + 1, n * num_cores:(n + 1) * num_cores] = pool.map(process_KS,
                                                                                                   [[maps_conv[
                                                                                                         i].flatten(),
                                                                                                     AD_maps_conv[
                                                                                                         i].flatten(),
                                                                                                     KS_test_name] for i
                                                                                                    in
                                                                                                    range(len(
                                                                                                        list_ID_tmp))])
        np.save(output_direc + 'KS_metric_unc', KS_metrics_unc)
        KS_metrics = np.zeros((len(KS_metrics_unc), 5 * (len(rsrcs) + 1)))
        for i in range(len(KS_metrics_unc)):
            for j in range(len(rsrcs) + 1):
                KS_metrics[i, 2 * j] = np.mean(KS_metrics_unc[i, j, :])
                KS_metrics[i, 2 * j + 1] = np.std(KS_metrics_unc[i, j, :])
                KS_metrics[i, 2 * j + 2] = np.median(KS_metrics_unc[i, j, :])
                KS_metrics[i, 2 * j + 3] = np.percentile(KS_metrics_unc[i, j, :], 16)
                KS_metrics[i, 2 * j + 4] = np.percentile(KS_metrics_unc[i, j, :], 84)

        if KS_metric_plot:
            imgs = img_cut_out_generator(43657, rsrc=0, size=200)
            metric_comparison_plot(KS_metrics[:, :3],
                                   'KS',
                                   imgs,
                                   output_direc,
                                   mode='no_unc')
        if KS_metric_per_rsrc_plot:
            metric_comparison_per_rsrc_plot(np.concatenate(([0], rsrcs), axis=0),
                                            KS_metrics,
                                            'KS',
                                            output_direc,
                                            mode='unc')

    if lc_distance_metric:
        num_cores = 10  # mp.cpu_count()
        # num_to_process = min(num_cores, len(list_ID))
        print('Using %i cores.' % num_cores)

        # pool = mmp.Pool(processes=num_to_process)
        print('Set up the pool..')

        '''We define four steps for each map:
        1) (mode == 'true') we use the true version, generate num_lc lightcurves compare to
        num_lc_picked number of picked lcs and record the min
        2) (mode == 'true_conv') we repeat the same process for the convolved version of the same map
        3) (mode == 'AD') we repeat the same process for the AD version of the same map in step 1
        4) (mode == 'AD_conv') we repeat the same process for the convolved version of the AD map in step 3'''

        modes = ['true', 'true_conv', 'AD', 'AD_conv']
        lcs_picked_conv = {}
        print('setting up the list ID...')
        # list_ID = np.array([23350, 43542, 46645, 48465, 46035])  # list_ID[:]
        # (0.3, 0.5, 0.5), (0.2, 1.5, 0.5), (1, 0.6, 0.5), (1.6, 0.2, 0.5), (1.6, 1.5, 0.5)

        if len(list_ID) % num_cores == 0:
            num_processes = int(len(list_ID) / num_cores)
        else:
            num_processes = int(len(list_ID) / num_cores) + 1

        if cross_map:
            print('Running in cross_map mode...')
            # ID_ref = 23350  # k, g, s = 0.3, 0.5, 0.5
            ID_ref = 33378  # k, g, s = 1.25, 1.4, 0.0
            mode_label = 'crossmap_refID_%i' % ID_ref
            map_ref = process_read([ID_ref, input_map_size, output_map_size, to_mag, norm_map_values])
            per_map = False
            mode_rand = 'fixed'
        elif per_map:
            print('Running in per_map mode...')
            mode_label = 'permap'
            mode_rand = 'random'

        for m, model in enumerate(models):
            if m != 5:
                continue
            # m = -1
            print('Running for model %s:' % model)
            for n in range(num_processes):
                print('Batch %i/%i of %i maps:' % (n, num_processes, num_cores))
                if n == num_processes - 1:
                    list_ID_tmp = list_ID[n * num_cores:]
                else:
                    list_ID_tmp = list_ID[n * num_cores:(n + 1) * num_cores]
                with Pool(processes=num_cores) as pool:
                    maps = pool.map(process_read, [[ID,
                                                    input_map_size,
                                                    output_map_size,
                                                    to_mag,
                                                    norm_map_values] for i, ID in enumerate(list_ID_tmp)])
                    if per_map:
                        maps_tmp = maps
                    elif cross_map:
                        maps_tmp = [[map_ref, map_] for map_ in maps]
                if m == 0:
                    all_maps = {'maps': maps, 'ID': list_ID_tmp}
                    # pkl.dump(all_maps, open(output_direc + 'saved_maps_batch_%i.pkl' % n,
                    #                         'wb'))

                model_param = [model,
                               model_files[m],
                               cost_labels[m]]
                autoencoder = read_AD_model(model_param[0],
                                            model_param[1],
                                            model_param[2])
                map_AD = [process_AD_predict_one_map([autoencoder, map_]) for map_ in maps]
                print('Starting lc fit metric calculation with uncertainty for model %s' % model)
                output_direc2 = output_direc + 'model_%s_' % model + mode_label
                if m == 0:
                    mode_select = 'all'
                    with Pool(processes=num_cores) as pool:
                        tmp = pool.map(process_lc4,
                                       [[model + '_' + str(list_ID[i]),
                                         maps_tmp_,
                                         map_AD[i],
                                         rsrcs,
                                         steps,
                                         num_lc,
                                         mode_select,
                                         mode_rand,
                                         output_direc2 + '_ID_%i' % list_ID_tmp[i],
                                         True] for i, maps_tmp_ in
                                        enumerate(maps_tmp)])

                else:
                    mode_select = 'all'
                    mode_select = 'all'
                    with Pool(processes=num_cores) as pool:
                        tmp = pool.map(process_lc4,
                                       [[model + '_' + str(list_ID[i]),
                                         maps_tmp_,
                                         map_AD[i],
                                         rsrcs,
                                         steps,
                                         num_lc,
                                         mode_select,
                                         mode_rand,
                                         output_direc2 + '_ID_%i' % list_ID_tmp[i],
                                         True] for i, maps_tmp_ in
                                        enumerate(maps_tmp)])

    if lc_distance_per_map100_for_saved_lc:
        num_cores = 10
        print('Found %i cores.' % num_cores)
        print(mmp.get_start_method())
        pool = mmp.Pool(processes=num_cores)
        print('Finished setting up the pool..')

        print('Reading the input lcs...')
        lcs = pkl.load(open(input_direc + 'lcs_%s_num10000.pkl' % models[0], 'rb'))
        if not gen_ID_list and saved_before:
            list_ID = lcs['ID']
        n_picked_lc = 1000
        fit_lc_metrics_unc_saved_lc = np.zeros((len(models) + 1, len(rsrcs) + 1, len(list_ID), n_picked_lc))

        print('Doing mode true...')
        fit_lc_metrics_unc_saved_lc[0, 0, :, :] = pool.map(process_lc2b,
                                                           [[lcs['true_lcs'][i][:n_picked_lc, 1, :],
                                                             lcs['true_lcs'][i][n_picked_lc:, 1, :]] for i in
                                                            range(len(list_ID))])
        for r, rsrc in enumerate(rsrcs):
            print('Doing mode true conv (rsrc = %0.1f)...' % rsrc)
            fit_lc_metrics_unc_saved_lc[0, r + 1, :, :] = pool.map(process_lc2b,
                                                                   [[lcs['conv_lcs_' + str(rsrc)][i][:n_picked_lc, 1,
                                                                     :],
                                                                     lcs['conv_lcs_' + str(rsrc)][i][n_picked_lc:, 1,
                                                                     :]] for i in
                                                                    range(len(list_ID))])
        for m, model in enumerate(models):
            AD_lcs = pkl.load(open(input_direc + 'AD_lcs_%s_num10000.pkl' % models[m], 'rb'))
            print('Doing mode AD...')
            fit_lc_metrics_unc_saved_lc[m + 1, 0, :, :] = pool.map(process_lc2b,
                                                                   [[lcs['true_lcs'][i][:n_picked_lc, 1, :],
                                                                     AD_lcs['AD_lcs'][i][n_picked_lc:, 1, :]] for i in
                                                                    range(len(list_ID))])

            for r, rsrc in enumerate(rsrcs):
                print('Doing mode AD conv (rsrc = %0.1f)...' % rsrc)
                fit_lc_metrics_unc_saved_lc[m + 1, r + 1, :, :] = pool.map(process_lc2b,
                                                                           [[lcs['conv_lcs_' + str(rsrc)][i][
                                                                             :n_picked_lc, 1, :],
                                                                             AD_lcs['conv_lcs_' + str(rsrc)][i][
                                                                             n_picked_lc:, 1, :]]
                                                                            for i in range(len(list_ID))])
        np.save(output_direc + 'lc_distance_metric_per_map100.npy', fit_lc_metrics_unc_saved_lc)

    read_me_creator(
                    output_direc,
                    len(list_ID),
                    num_lc,
                    list_ID,
                    rsrcs,
                    vars(args))

    end_time = time.time()  # Record the end time
    total_runtime = end_time - start_time
    print(
        'Total time to read and convolve %i maps at %i source sizes and generate %i lightcurves for each of them is '
        '%.1f seconds' % (
            len(list_ID), len(rsrcs), num_lc, total_runtime))

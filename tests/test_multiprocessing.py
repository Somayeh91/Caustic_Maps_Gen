import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import time
from more_info import models_info
from prepare_maps import process_lc3, process_read, process_AD_predict_one_map, process_lc4
from maps_util import read_AD_model

list_ID = np.loadtxt('./../data/eval_1000_maps.txt', dtype=int)[:100]
print('Running for %i maps...'%len(list_ID))
rsrcs = np.array([0.1, 0.5])
steps = 100
num_lc = 100
n_models = len(models_info['data'])
models = models_info['job_names']
model_files = models_info['job_model_filename']
cost_labels = models_info['job_cost_labels']
m = 0
model_param = [models[m],
               model_files[m],
               cost_labels[m]]
func_name = 'process_lc3'
mode_select = 'all'
output_directory = './../../../fred/oz108/skhakpas/results/24-03-24-03-00-19/lc_metric_plots/test_process_lc3/'

autoencoder = read_AD_model(model_param[0],
                            model_param[1],
                            model_param[2])
start_time = time.time()
num_cores = 10
# pool = Pool(processes=num_cores)
n_proc = num_cores
if len(list_ID) // num_cores == 0:
    num_processes = int(len(list_ID) / num_cores)
else:
    num_processes = int(len(list_ID) / num_cores) + 1

start_time = time.time()
print('Testing function %s run in pool.map:' % func_name)
for n in range(num_processes):
    print('Batch %i/%i of %i maps:'%(n, num_processes, num_cores))
    if n == num_processes - 1:
        list_ID_tmp = list_ID[n * num_cores:]
    else:
        list_ID_tmp = list_ID[n * num_cores:(n + 1) * num_cores]

    with Pool(processes=n_proc) as pool:
        maps = pool.map(process_read, [[ID, i] for i, ID in enumerate(list_ID_tmp)])
        # pool.join()
    map_AD = [process_AD_predict_one_map([autoencoder, map_]) for map_ in maps]

    args2 = [[i,
              map_,
              map_AD[i],
              rsrcs,
              steps,
              num_lc,
              mode_select,
              output_directory + 'model_%s_ID_%i_permap_true' % (models[m], list_ID_tmp[i])] for i, map_ in enumerate(maps)]
    with Pool(processes=n_proc) as pool:
        pool.map(process_lc4, args2)
end_time = time.time()
print('Time for using pool.map is %0.2f seconds' % (end_time - start_time))
# pool.join()
# pool.close()

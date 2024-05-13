import numpy as np
import argparse
import random
import math
from tqdm import tqdm
import pickle as pkl
import multiprocessing as mp

def parse_options():
    """Function to handle options speficied at command line
    """
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('-dim', action='store', default=10000,
                        help='What is the dimension of the maps.')
    parser.add_argument('-directory', action='store',
                        default='./../../../fred/oz108/GERLUMPH_project/DATABASES/gerlumph_db/',
                        help='Specify the directory where the maps are stored on the supercomputer.')
    parser.add_argument('-batch_size', action='store', default=600, help='Batch size for the autoencoder.')
    parser.add_argument('-res_scale', action='store', default=10,
                        help='With what scale should the original maps resoultion be reduced.')
    parser.add_argument('-list_IDs_directory', action='store',
                        default='./../data/gd0_similar_ks.dat',
                        help='Specify the directory where the list of map IDs is stored.')
    parser.add_argument('-output_dir', action='store',
                        default='./../../../fred/oz108/skhakpas/',
                        help='The directory to save the output.'
                             'Make sure you put / at the end of it!')
    parser.add_argument('-date', action='store',
                        default='',
                        help='Specify the directory where the results should be stored.')

    # Parses through the arguments and saves them within the keyword args
    return parser.parse_args()


def process_read(ID):
    f = open(maps_directory + str(ID) + "/map.bin", "rb")
    map_tmp = np.fromfile(f, 'i', -1, "")
    # print(map_tmp.shape)
    shape1 = int(np.sqrt(map_tmp.shape[0]))

    maps = (np.reshape(map_tmp, (-1, shape1)))
    if shape1 == 10000:
        return maps[0::10, 0::10].reshape((1000, 1000, 1))
    else:
        return maps[::4, ::4][:1000, :1000].reshape((1000, 1000, 1))





print('Reading in the arguments...')
args = parse_options()


ls_maps = np.loadtxt(args.list_IDs_directory, usecols=(0,), dtype=int)
batch_size = int(args.batch_size)
n_files = len(ls_maps) #math.trunc(len(ls_maps)/float(batch_size))
random.seed(10)
shuffler = np.random.permutation(len(ls_maps))
ls_maps = ls_maps[shuffler]
maps_directory = args.directory
res_scale = args.res_scale
dim = args.dim
side = int(dim/res_scale)
n_channels = 1
output_dir = args.output_dir
batch_map = {}


num_cores = mp.cpu_count()
print('Number of cores= %i'%num_cores)
tot_count = len(ls_maps)

print('Reading %i maps...'%tot_count)

while tot_count> 0:
    num_to_process = min(num_cores, len(ls_maps) )
    batch = ls_maps[:num_to_process]
    tot_count = tot_count-num_to_process
    ls_maps = ls_maps[num_to_process:]
    # print(batch)

    pool = mp.Pool(processes=num_to_process)
    maps_batch = pool.map(process_read, batch)
    pool.close()
    pool.join()

    for i, ID in enumerate(batch):
        batch_map[ID] = maps_batch[i]

    print('%i maps remains...'%tot_count)

# for i in tqdm(range(n_files)):
    
#     # print('batch ', i)
#     # for j, ID in enumerate(ls_maps[i*batch_size:(i+1)*batch_size]):
#     ID = ls_maps[i]
#     print(ID)
    
    
#     if shape1 == 10000:
#         batch_map[ID] = np.zeros((side, side, n_channels))
#         batch_map[ID] = maps[0::res_scale, 0::res_scale].reshape((side, side, n_channels))
#     else:
#         batch_map[ID] = np.zeros((shape1, shape1, n_channels))
#         batch_map[ID] = maps.reshape((shape1, shape1, n_channels))
#     # print(np.median(np.asarray(batch_map[ID]).flatten()))


f = open(output_dir + 'maps_similar_ks_1000x1000.pkl', 'wb')
pkl.dump(batch_map, f)

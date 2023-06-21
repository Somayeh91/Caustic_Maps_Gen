import numpy as np
import argparse
import random
import math
from tqdm import tqdm
import pickle as pkl

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
                        default='./../data/ID_maps_selected_kappa_equal_gamma.dat',
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


print('Reading in the arguments...')
args = parse_options()


ls_maps = np.loadtxt(args.list_IDs_directory, usecols=(0,), dtype=int)
batch_size = args.batch_size
n_files = math.trunc(len(ls_maps)/batch_size)
random.seed(10)
shuffler = np.random.permutation(len(ls_maps))
ls_maps = ls_maps[shuffler]
maps_directory = args.directory
res_scale = args.res_scale
dim = args.dim
side = int(dim/res_scale)
n_channels = 1
output_dir = args.output_dir

for i in tqdm(range(n_files)):
    batch_map = {}
    print('batch ', i)
    for j, ID in enumerate(ls_maps[i*batch_size:(i+1)*batch_size]):
        print(ID)
        batch_map[ID] = np.zeros((side, side, n_channels))
        f = open(maps_directory + str(ID) + "/map.bin", "rb")
        map_tmp = np.fromfile(f, 'i', -1, "")
        maps = (np.reshape(map_tmp, (-1, 10000)))
        batch_map[ID] = maps[0::res_scale, 0::res_scale].reshape((side, side, n_channels))
        print(np.median(np.asarray(batch_map[ID]).flatten()))


    f = open(output_dir + 'maps_batch'+str(i)+'.pkl', 'wb')
    pkl.dump(batch_map, f)

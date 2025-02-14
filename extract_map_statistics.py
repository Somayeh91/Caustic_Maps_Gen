import random
import argparse
import pandas as pd
from tqdm import tqdm
from convolver import *
import pickle as pkl
from multiprocessing import Pool
import multiprocessing as mp
from prepare_maps import process_get_maps_min_max, process_read_lens_pos
import h5py


def parse_options():
    """Function to handle options speficied at command line
    """
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('-directory', action='store',
                        default='./../../../fred/oz108/GERLUMPH_project/DATABASES/gerlumph_db/',
                        help='Specify the directory where the maps are stored on the supercomputer.')
    parser.add_argument('-output_directory', action='store',
                        default='./../data/',
                        help='Specify the directory where the results should be stored.')
    parser.add_argument('-list_IDs_directory', action='store',
                        default='./../data/all_all_IDs_list.dat',
                        help='Specify the directory where the list of map IDs is stored.')
    parser.add_argument('-input_map_size', action='store',
                        default=10000,
                        help='size of each side the original maps in pixels.')
    parser.add_argument('-output_map_size', action='store',
                        default=1000,
                        help='size of each side the maps with reduced resolution in pixels.')
    parser.add_argument('-test_set_size', action='store',
                        default=10,
                        help='Size of the test set.')
    parser.add_argument('-stat', action='store',
                        default=False,
                        help='Do you want to save min, max, mean, and median of all maps?')
    parser.add_argument('-conv_stat', action='store',
                        default=False,
                        help='Do you want to save min, max, mean, and median of all convolved maps?')
    parser.add_argument('-conv_rsrc', action='store',
                        default=1,
                        help='What is the radius of the source (in units of Einstein Radius) to convolve with maps.')
    parser.add_argument('-date', action='store',
                        default='',
                        help='Specify the directory where the results should be stored.')
    parser.add_argument('-save_lens_pos', action='store',
                        default=False,
                        help='Do you want to save the lens positions as well?')

    # Parses through the arguments and saves them within the keyword args
    arguments = parser.parse_args()
    return arguments


print('Setting up the initial parameters...')
args = parse_options()

# Datasets
partition = {}
ls_maps = np.loadtxt(args.list_IDs_directory, usecols=(0,), dtype=int)
random.seed(10)
shuffler = np.random.permutation(len(ls_maps))
ls_maps = ls_maps[shuffler]
n_maps = len(ls_maps)

map_direc = args.directory
input_map_size = int(args.input_map_size)
output_map_size = int(args.output_map_size)
n_test_set = int(args.test_set_size)
stat = args.stat
save_lens_pos = args.save_lens_pos
conv_stat = args.conv_stat
conv_rsrc = float(args.conv_rsrc)
test_set_index = random.sample(list(ls_maps), n_test_set)
output_direc = args.output_directory


num_cores = mp.cpu_count()
print('Number of cores= %i'%num_cores)
maps_lens_pos = {}
maps_lens_pos_values = []
maps_stat_values = []

if len(ls_maps) % num_cores == 0:
    num_processes = int(len(ls_maps) / num_cores)
else:
    num_processes = int(len(ls_maps) / num_cores) + 1

for n in range(num_processes):
    print('Batch %i/%i of %i maps:' % (n, num_processes, num_cores))
    if n == num_processes - 1:
        list_ID_tmp = ls_maps[n * num_cores:]
    else:
        list_ID_tmp = ls_maps[n * num_cores:(n + 1) * num_cores]
    if stat:
        with Pool(processes=num_cores) as pool:
            maps_stat_tmp = pool.map(process_get_maps_min_max, [[ID, input_map_size, output_map_size] for i, ID in enumerate(list_ID_tmp)])
        maps_stat_values += maps_stat_tmp
    if save_lens_pos:
        with Pool(processes=num_cores) as pool:
            maps_lens_pos_tmp = pool.map(process_read_lens_pos, [map_direc+str(ID) for ID in list_ID_tmp])

        for i, ID in enumerate(list_ID_tmp):
            maps_lens_pos[ID] = maps_lens_pos_tmp[i]


if stat:
    maps_stat_values = np.asarray(maps_stat_values)
    df = pd.DataFrame({'ID': ls_maps,
                       'mag_min': maps_stat_values[:, 0],
                       'mag_max': maps_stat_values[:, 1],
                       'log_mag_min': maps_stat_values[:, 2],
                       'log_mag_max': maps_stat_values[:, 3]})
    df.to_csv(output_direc + '4096pixel_maps_meta_kgs.csv')

if save_lens_pos:
    # lens_poss = dict(zip(ls_maps, maps_lens_pos_values))
    f = open(output_direc + '4096pixel_maps_lens_pos.pkl', 'wb')
    pkl.dump(maps_lens_pos, f)
    # with h5py.File(output_direc + '4096pix_maps_lens_pos.h5', "w") as f:
    #     # Create a compressed dataset
    #     f.create_dataset("lens_pos", data=maps_lens_pos_values, compression="gzip", compression_opts=9)
    #     f.create_dataset("IDs", data=ls_maps, compression="gzip", compression_opts=9)
    #

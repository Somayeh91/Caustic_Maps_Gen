import numpy as np
import argparse
from multiprocessing import Pool
import multiprocessing as mp
from prepare_maps import process_read
import h5py

def parse_options():
    """Function to handle options speficied at command line
    """
    parser = argparse.ArgumentParser(description='Process input parameters.')
    parser.add_argument('-dim', action='store', default=10000,
                        help='What is the dimension of the maps.')
    parser.add_argument('-directory', action='store',
                        default='./../../../fred/oz108/GERLUMPH_project/DATABASES/gerlumph_db/',
                        help='Specify the directory where the maps are stored on the supercomputer.')
    parser.add_argument('-input_map_size', action='store',
                        default=10000,
                        help='size of each side the original maps in pixels.')
    parser.add_argument('-output_map_size', action='store',
                        default=1000,
                        help='size of each side the maps with reduced resolution in pixels.')
    parser.add_argument('-batch_size', action='store', default=600, help='Batch size for the autoencoder.')
    parser.add_argument('-res_scale', action='store', default=10,
                        help='With what scale should the resolution of the original map be reduced.')
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




print('Reading in the arguments...')
args = parse_options()


ls_maps = np.loadtxt(args.list_IDs_directory, usecols=(0,), dtype=int)
batch_size = int(args.batch_size)
n_files = len(ls_maps) #math.trunc(len(ls_maps)/float(batch_size))


maps_directory = args.directory
input_map_size = int(args.input_map_size)
output_map_size = int(args.output_map_size)
res_scale = args.res_scale
dim = args.dim
side = int(dim/res_scale)
n_channels = 1
output_dir = args.output_dir
batch_map = {}


num_cores = mp.cpu_count()
print('Number of cores= %i'%num_cores)

maps_all = np.zeros((len(ls_maps), output_map_size, output_map_size, 1), dtype='float16')

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
    with Pool(processes=num_cores) as pool:
        maps = pool.map(process_read, [[ID, input_map_size, output_map_size, True, True] for i, ID in enumerate(list_ID_tmp)])

    maps_tmp = maps
    if n == num_processes - 1:
        maps_all[n * num_cores:, :, :, 0] = np.asarray(maps_tmp)
        print(np.mean(maps_all[n * num_cores:, :, :, 0].flatten()))
    else:
        maps_all[n * num_cores:(n + 1) * num_cores, :, :, 0] = np.asarray(maps_tmp)
        print(np.mean(maps_all[n * num_cores:(n + 1) * num_cores, :, :, 0].flatten()))




    # np.save(output_dir + 'full_maps_batch%i'%n, maps_tmp)
with h5py.File(output_dir+'4096pix_maps.h5', "w") as f:
    # Create a compressed dataset
    f.create_dataset("maps", data=maps_all, compression="gzip", compression_opts=9)
    f.create_dataset("IDs", data=ls_maps, compression="gzip", compression_opts=9)

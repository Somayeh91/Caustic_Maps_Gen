import random
import argparse
import pandas as pd
from tqdm import tqdm
from convolver import *
import pickle as pkl


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
    parser.add_argument('-test_set_size', action='store',
                        default=10,
                        help='Size of the test set.')
    parser.add_argument('-stat', action='store',
                        default=True,
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
ls_maps = np.loadtxt(args.list_IDs_directory, dtype=int)
random.seed(10)
shuffler = np.random.permutation(len(ls_maps))
ls_maps = ls_maps[shuffler]
n_maps = len(ls_maps)

map_direc = args.directory
n_test_set = int(args.test_set_size)
stat = args.stat
save_lens_pos = args.save_lens_pos
conv_stat = args.conv_stat
conv_rsrc = float(args.conv_rsrc)
test_set_index = random.sample(list(ls_maps), n_test_set)
output_direc = args.output_directory
mins = []
maxs = []
log_mins = []
log_maxs = []

mag_avg = []
ray_avg = []
cv = convolver(rsrc=conv_rsrc)

mins_conv = []
maxs_conv = []
log_conv_mins = []
log_conv_maxs = []
lens_poss = {}
ks = []
gs = []
ss = []
# mag_avg_conv = []
# ray_avg_conv = []

for ID in tqdm(ls_maps):

    f1 = open(map_direc + str(ID) + "/map.bin", "rb")
    map_tmp = np.fromfile(f1, 'i', -1, "")
    maps = (np.reshape(map_tmp, (-1, 10000)))
    f2 = open(map_direc + str(ID) + "/mapmeta.dat", "r")
    lines = f2.readlines()
    mag_convertor = float(lines[0].split(' ')[0])
    k, g, s = float(lines[3].split(' ')[0]), float(lines[3].split(' ')[1]), float(lines[3].split(' ')[2])
    maps = maps * mag_convertor
    if save_lens_pos:
        lens_pos = np.loadtxt(map_direc + str(ID) + "/lens_pos.dat")
    if stat:
        mins.append(np.min(maps))
        maxs.append(np.max(maps))
        ks.append(k)
        gs.append(g)
        ss.append(s)
        # log_mins.append(np.min(maps))
        # log_maxs.append(np.max(maps))
    if save_lens_pos:
        lens_poss[ID] = lens_pos

    mag_avg.append(float(lines[0].split(' ')[0]))
    ray_avg.append(float(lines[0].split(' ')[1].split('/')[0]))

    if conv_stat:
        cv.conv_map(maps)
        conv_map_tmp = cv.magcon
        mins_conv.append(np.min(conv_map_tmp))
        maxs_conv.append(np.max(conv_map_tmp))

if stat:
    df = pd.DataFrame({'ID': ls_maps,
                       'kappa': ks,
                       'gamma': gs,
                       's': ss,
                       'min': mins,
                       'max': maxs,
                       'mag_avg': mag_avg,
                       'ray_avg': ray_avg})
    df.to_csv(output_direc + 'all_all_maps_meta_kgs.csv')

if save_lens_pos:
    f = open(output_direc + 'all_maps_lens_pos.pkl', 'wb')
    pkl.dump(lens_poss, f)

if conv_stat:
    df = pd.DataFrame({'ID': ls_maps, 'min': mins_conv, 'max': maxs_conv, 'mag_avg': mag_avg, 'ray_avg': ray_avg})
    df.to_csv(output_direc + 'all_conv_maps_meta_rsrc_' + str(conv_rsrc) + '.csv')

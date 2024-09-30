import numpy as np
from scipy import stats
from random import shuffle
from FID_calculator import FID_preprocess, calculate_fid
from maps_util import chi2_calc
from prepare_maps import *


def process_KS(maps):
    map_1 = maps[0]
    map_2 = maps[1]
    test_name = maps[2]
    ks_metric = calculate_ks_metric(map_1, map_2, test_name)
    return ks_metric


def calculate_ks_metric(map1, map2, test_name):
    if test_name == 'KS':
        return stats.ks_2samp(map1, map2)[0]
    elif test_name == 'anderson':
        return stats.anderson_ksamp([map1, map2])[0] / len(map1)


def comapre_lc_metric(x, y, stat='mean'):
    f = (x - y) ** 2
    if stat == 'mean':
        return np.mean(f)
    elif stat == 'median':
        return np.median(f)


def calculate_lc_metric(all_lc, stat_1=np.mean, stat_2=np.mean):
    """
	This function takes in a list of shape (2, n_maps, n_lightcurves, n_samples):
	The first dimension is for true maps' lightcurves and AD maps' lightcurves,
	the second dimension is number of individual maps,
	the third dimension is for number of individual lightcurves per map,
	The last dimension is the number of data points in the light curves.

	The function comapre_lc_metric is calculated for
	All the maps' light curves are returned, and a single value is returned.

	Input: all_lc (list of shape (2, n_maps, n_lightcurves, n_samples))
	Output: lc_metric (1,)
	"""
    n_maps = len(all_lc[0])
    n_lc = len(all_lc[0][0])

    lc_metric_tmp = np.zeros((n_maps))

    for j in range(n_maps):
        l1 = np.mean([comapre_lc_metric((all_lc[0][j, i]),
                                        (all_lc[1][j, i]))
                      for i in
                      range(n_lc)])
        lc_metric_tmp[j] = stat_1(l1)

    return stat_2(lc_metric_tmp)


def calculate_FID_metric(model, input_images, output_images):
    """
    Make sure you run this before calling this function to get the model:
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

    This function takes two sets of images with
    arbitrary shapes as input and returns their FID score.

    input:
        model = the InceptionV3 model
        images1 = a set of input images in arbitrary shape of (image_index, x_dimension, y_dimension, n_channels)
        images2 = a set of output images in arbitrary shape of (image_index, x_dimension, y_dimension, n_channels)
    output:
        FID score which is a scalar.

    """
    input_images = FID_preprocess(input_images)
    # print('Loaded input', input_images.shape)
    shuffle(input_images)

    output_images = FID_preprocess(output_images)

    fid = calculate_fids(model, input_images, output_images)
    return fid


def process_FID(map_sets):
    maps_set1, maps_set2, inceptionv3 = map_sets[0], map_sets[1], map_sets[2]
    tmp = calculate_FID_metric(inceptionv3,
                               maps_set1,
                               maps_set2)
    return tmp


def fitting_lc_metric(lc1, lc2):
    return chi2_calc(lc1, lc2)


def process_fit_lc(lcs):
    lc_picked = lcs[0]
    alllc = lcs[1]
    metrics = np.zeros((len(alllc)))
    for l, lc in enumerate(alllc):
        metrics[l] = fitting_lc_metric(lc_picked, lc)
    return [np.min(metrics), np.argmin(metrics)]


def process_fit_lc2(lcs):
    # print('Starting fit lc process...')
    lc_picked = lcs[0]
    alllc = lcs[1]
    metrics = np.array([fitting_lc_metric(lc_picked, lc) for lc in alllc])
    return np.round(np.min(metrics), 3)


def process_fit_lc_returnall(lcs):
    # print('Starting fit lc process...')
    lc_picked = lcs[0]
    alllc = lcs[1]
    metrics = np.array([fitting_lc_metric(lc_picked, lc) for lc in alllc])
    return metrics

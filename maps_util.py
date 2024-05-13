from my_classes import tweedie_loss_func,\
					   lc_loss_func,\
					   custom_loss_func,\
					   NormalizeData
import numpy as np
from keras.models import load_model
from best_models_info import models_info
import random
import pandas as pd


map_direc = './../../../fred/oz108/GERLUMPH_project/DATABASES/gerlumph_db/'
model_direc = './../../../fred/oz108/skhakpas/results/'

# map_direc = '/Users/somayeh/Downloads/maps/'
# model_direc = '/Users/somayeh/Downloads/job_results_downloaded/'



def read_AD_model(model_ID, model_file, cost_label):


	if cost_label.startswith('lc'):
		if cost_label == 'lc_bce':
			autoencoder = load_model(model_direc+
									 model_ID+'/'+model_file,
									 custom_objects={'tweedie_loglikelihood': tweedie_loss_func(0.5)})
		else:
			metric = cost_label.split('_')[1]
			autoencoder = load_model(model_direc+
								model_ID+'/'+model_file,
						  custom_objects={'lc_loglikelihood': lc_loss_func(metric)})
	elif cost_label == 'custom':
		autoencoder = load_model(model_direc+
							   model_ID+'/'+model_file,
								 custom_objects={'custom_loglikelihood': custom_loss_func()})
	else:
		autoencoder = load_model(model_direc+
					   model_ID+'/'+model_file)
	return autoencoder

def read_binary_map(ID, scale=10, to_mag = False):
	f1 = open(map_direc + str(ID) + "/map.bin", "rb")
	map_tmp = np.fromfile(f1, 'i', -1, "")
	map = (np.reshape(map_tmp, (-1, 10000)))

	if to_mag:
		mag_convertor = read_map_meta(ID)[0]
		return map[0::scale,0::scale]*mag_convertor
	else:
		return map[0::scale,0::scale]

def read_map_meta(ID):
	f2 = open(map_direc + str(ID) + "/mapmeta.dat", "r")
	lines = f2.readlines()

	# print(lines)
	
	k, g, s = float(lines[3].split(' ')[0]), float(lines[3].split(' ')[1]), float(lines[3].split(' ')[2])
	mag_avg = float(lines[0].split(' ')[0])
	ray_avg = float(lines[0].split(' ')[1].split('/')[0])

	mag_convertor = np.abs(mag_avg/ray_avg)

	return mag_convertor, k, g, s, mag_avg, ray_avg

def read_map_meta_for_file(input_dir,
						   output_filename,
						   output_dir):
	print('Reading the file...')
	all_ID = np.loadtxt(input_dir, usecols=(0,), dtype=int)
	print('The file contains %i IDs.'%len(all_ID))
	print('Reading Meta Data...')
	results = np.asarray([read_map_meta(ID) for ID in all_ID])
	print('Meta Data successfully read.')
	
	df = pd.DataFrame({'ID': all_ID,
					   'const': results[:,0],
					   'k': results[:,1], 
					   'g': results[:,2], 
					   's': results[:,3], 
					   'mag_avg':results[:,4],
					   'ray_avg': results[:,5]})
	df.to_csv(output_dir + output_filename)	

	

def norm_maps(map, coeff, offset=0.004, norm=True, norm_min=-3, norm_max=6):
	temp = np.log10(map * coeff + offset)
	if norm:
		return NormalizeData(temp, data_max=norm_max, data_min=norm_min)
	else:
		return temp

def reverse_norm(map, norm_min=-3, norm_max=6):
	return (map*(norm_max-norm_min))+norm_min

def reverse_norm_maps(map, coeff, offset=0.004, norm=True, norm_min=-3, norm_max=6):
	if norm:
		temp = (map*(norm_max-norm_min))+norm_min
	else:
		temp = data
	return (((10**temp)-offset)/coeff).astype('int32')

def reverse_maps_into_mag(map, offset=0.004, norm=True, norm_min=-3, norm_max=6):
	if norm:
		temp = (map*(norm_max-norm_min))+norm_min
	else:
		temp = data
	return ((10**temp)-offset).astype('float16')


def prepare_cuttout_map(ID, rsrc = 0):
	map_cuttout = read_binary_map(ID)
	mag_convertor = read_map_meta(ID)[0]

	models = models_info['job_names']
	model_files = models_info['job_model_filename']
	cost_labels = models_info['job_cost_labels']
	shape_ = map_cuttout.shape[0]

	map_cuttout = norm_maps(map_cuttout, 
							mag_convertor)
	images_cutout_exmp = np.zeros((len(models)+1, 
								   shape_,
								   shape_))

	for m, model in enumerate(models):
		autoencoder = read_AD_model(model, model_files[m], cost_labels[m])
		AD_map = autoencoder.predict(map_cuttout.reshape((1, shape_, shape_, 1)))
		AD_map = AD_map.reshape((shape_, shape_))

		if rsrc == 0:
			images_cutout_exmp[m+1, :, :] = AD_map
			if m == 0:
				images_cutout_exmp[0, :, :] = map_cuttout

		else:
			AD_map_conv = convolve_map(AD_map, 
									rsrc)
			images_cutout_exmp[m+1, :, :] = AD_map_conv

			if m == 0:
				map_conv = convolve_map(map_cuttout, 
										   rsrc)
				images_cutout_exmp[0, :, :] = map_conv

	return images_cutout_exmp


def img_cut_out_generator(ID, rsrc, size=200):
	images = prepare_cuttout_map(ID, rsrc = rsrc)
	imgs = np.zeros((len(images), size, size))
	for i, im in enumerate(images):
		imgs[i, :, :] = im[0:size, 0:size]
	return imgs

def chi2_calc(f, x):
	return np.sum((f-x)**2)


def resize(image, output_shape):
	output_shape = tuple(output_shape)
	output_ndim = len(output_shape)
	input_shape = image.shape
	if output_ndim > image.ndim:
		# append dimensions to input_shape
		input_shape += (1, ) * (output_ndim - image.ndim)
		image = np.reshape(image, input_shape)
	elif output_ndim == image.ndim - 1:
		# multichannel case: append shape of last axis
		output_shape = output_shape + (image.shape[-1], )
	elif output_ndim < image.ndim:
		raise ValueError("output_shape length cannot be smaller than the "
						 "image number of dimensions")

	return image, output_shape

def eval_maps_selection(num = 100, seed = 33):
	random.seed(seed)
	path = './../data/'
	IDs = np.loadtxt(path+'GD1_ids_list2.txt', dtype=int)[:,0]
	IDs_selected = random.sample(list(IDs), num)
	with open(path+'eval_maps_%imaps_seed%i.txt'%(num, seed), 'w') as file:
		for ID in IDs_selected:
			file.write(str(ID)+ '\n')
		file.close()
	return IDs_selected

def read_all_data():
	data_dict = {}

	for index in range(11):
		path = './../../../fred/oz108/skhakpas/all_maps_batch' + str(index) + '.pkl'
		data_dict_tmp = pkl.load(open(path, "rb"))
		data_dict = {**data_dict, **data_dict_tmp}

	return data_dict

def split_data(keys, 
			   train_percentage = 0.8,
			   valid_percentage = 0.1):
	partition = {}
	ls_maps = keys
	random.seed(10)
	shuffler = np.random.permutation(len(ls_maps))

	shuffler = random.sample(list(shuffler), int(sample_size))
	ls_maps = ls_maps[shuffler]
	n_maps = len(ls_maps)

	if train_percentage>0.98:
		indx1 = np.arange(n_maps, dtype=int)
		indx2 = np.array([])
		indx3 = np.array([])
	else:
		indx1 = np.arange(int(train_percentage * n_maps), dtype=int)
		indx2 = np.arange(int(train_percentage * n_maps), 
						  int((train_percentage+valid_percentage) * n_maps))
		indx3 = np.arange(int((train_percentage+valid_percentage) * n_maps), n_maps)

	partition['train'] = ls_maps[indx1]
	partition['validation'] = ls_maps[indx2]
	partition['test'] = ls_maps[indx3]

	return partition

def read_me_creator(input_dir,
					output_dir, 
					num_maps,
					num_lc,
					IDs,
					rsrcs,
					args):

	# report = (str())

	# 'output_direc = ' + str(args.output_directory)
	# 'input_direc = 'args.input_directory
	# 'plot_exp_conv = 'args.plot_exp_conv
	# 'conv_AD_maps = 'args.conv_AD_maps
	# 'single_model_read = 'args.single_model_read
	# single_model_read_from_file = args.single_model_read_from_file
	# multi_models_read_maps = args.multi_models_read_maps
	# multi_models_read_lcs_from_file = args.multi_models_read_lcs_from_file
	# multi_models_read_maps_from_file = args.multi_models_read_maps_from_file
	# multi_models_read_lcs_and_maps_from_file = args.multi_models_read_lcs_and_maps_from_file
	# model_ID = args.AD_model_ID
	# model_file = args.AD_model_file_name
	# cost_label = args.AD_model_cost_label
	# lc_metric_calc = args.lc_metric_calc
	# FID_metric_calc = args.FID_metric_calc
	# KS_metric_calc = args.KS_metric_calc
	# fit_lc_metric_calc = args.fit_lc_metric_calc
	# gen_lc = args.gen_lc
	# num_lc = int(args.num_lc)
	# plot_exp_lc = args.plot_exp_lc
	# lc_metric_plot = args.lc_metric_plot
	# FID_metric_plot = args.FID_metric_plot
	# KS_metric_plot = args.KS_metric_plot
	# fit_lc_metric_plot = args.fit_lc_metric_plot
	# verbose = args.verbose
	# rsrcs = np.asarray([float(rsrc) for rsrc in args.rsrc.split(',')])
	# n_models = len(models_info['data'])
	# models = models_info['job_names']
	# model_files = models_info['job_model_filename']
	# cost_labels = models_info['job_cost_labels']
	# lc_metrics = np.zeros((n_models, len(rsrcs)+1))
	# FID_metrics = np.zeros((n_models, len(rsrcs)+1))
	# KS_metrics = np.zeros((n_models, len(rsrcs)+1))
	# fit_lc_metrics = np.zeros((n_models+1, len(rsrcs)+1, len(list_ID)))
	# shape_ = 1000
	# save_lcs = args.save_lcs
	# save_maps = args.save_maps

	with open(output_dir+'read_me.txt', 'w') as file:
		file.write("The code evaluation.py was run on %i maps with IDs: \n"%num_maps)
		file.write('data was read from %s \n'%input_dir)
		# file.write(str(report)+'\n')
		for ID in IDs:
			file.write(str(ID)+ ' ,')
		file.write('\n')
		file.write('Maps were convolved with source sizes: ')
		for rsrc in rsrcs:
			file.write('%.1f ,'%rsrc)
		file.write('\n')
		file.write('%i lightcurves were generated for each map and its convolved versions.'%num_lc)
		file.write('\n')
		file.write('Results are saved at %s'%output_dir)
		

		file.close()
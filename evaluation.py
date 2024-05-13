import multiprocessing as mp
import argparse
import time
import pickle as pkl
from best_models_info import models_info
import os
from prepare_maps import prepare_maps, process_lc, process_read, process_conv
from plotting_utils import *
from conv_utils import *
from metric_utils import *
from maps_util import *
from FID_calculator import *
import random
from my_classes import DataGenerator2
import matplotlib.pyplot as plt
import matplotlib as mpl

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
						default='./../data/all_all_IDs_list.dat',
						help='Specify the directory where the list of map IDs is stored.')
	parser.add_argument('-rsrc', action='store',
						default='0.1,0.5',
						help='A list of source sizes for convolution, separated by commas.')
	parser.add_argument('-metric', action='store',
						default='KS',
						help='Which metric you want to use to evaluate.')
	parser.add_argument('-scale_down', action='store',
						default=10,
						help='By what scale do you want to lower the resolution of the maps.')
	parser.add_argument('-single_model_read', action='store',
						default=False,
						help='If True, it reads a list of maps and convolves them.')
	parser.add_argument('-single_model_read_from_file', action='store',
						default=False,
						help='If True, it reads from file a list of saved maps and their convolved ones.')
	parser.add_argument('-multi_models_read_maps', action='store',
						default=False,
						help='If True, it reads a list of maps and convolves them for multiple models.')	
	parser.add_argument('-multi_models_read_lcs_from_file', action='store',
						default=False,
						help='If True, it reads a list of lcs from file.')	
	parser.add_argument('-multi_models_read_maps_from_file', action='store',
						default=False,
						help='If True, it reads a list of true and convolved maps from file.')	
	parser.add_argument('-multi_models_read_lcs_and_maps_from_file', action='store',
						default=False,
						help='If True, it reads a list of true and convolved maps and their lightcurves from file.')		
	parser.add_argument('-conv_AD_maps', action='store',
						default=False,
						help='If True, it predicts AD maps for a list of true maps and convolves them.')
	parser.add_argument('-AD_model_ID', action='store',
						default='23-12-14-01-12-07',
						help='AD model name.')
	parser.add_argument('-AD_model_file_name', action='store',
						default='model_10000_8_0.0001',
						help='AD model file name.')
	parser.add_argument('-AD_model_cost_label', action='store',
						default='custom',
						help='AD model cost label.')
	parser.add_argument('-gen_lc', action='store',
						default=False,
						help='If True, it generates lightcruves from a map.')
	parser.add_argument('-num_lc', action='store',
						default=1,
						help='If gen_lc True, we need to define the number of lc.')
	parser.add_argument('-saved_maps_dir', action='store',
						default='',
						help='If read_conv is False, you need to give and address to where the saved maps are.')
	parser.add_argument('-plot_exp_conv', action='store',
						default=False,
						help='Do you want to plot an example figure of the true map and convolved map?')
	parser.add_argument('-plot_exp_lc', action='store',
						default=False,
						help='Do you want to plot an example figure of a generated lightcurve?')
	parser.add_argument('-lc_metric_calc', action='store',
						default=False,
						help='Do you want to calculate lc_metric?')
	parser.add_argument('-lc_metric_plot', action='store',
						default=False,
						help='Do you want to plot the lc_metric?')
	parser.add_argument('-FID_metric_calc', action='store',
						default=False,
						help='Do you want to calculate FID_metric?')
	parser.add_argument('-FID_metric_plot', action='store',
						default=False,
						help='Do you want to plot the FID_metric?')
	parser.add_argument('-KS_metric_calc', action='store',
						default=False,
						help='Do you want to calculate KS_metric?')
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
	parser.add_argument('-lc_metric_per_rsrc_plot', action='store',
						default=False,
						help='Do you want to plot the lc_metric per rsrc?')
	parser.add_argument('-FID_metric_per_rsrc_plot', action='store',
						default=False,
						help='Do you want to plot the FID_metric  per rsrc?')
	parser.add_argument('-KS_metric_per_rsrc_plot', action='store',
						default=False,
						help='Do you want to plot the KS_metric  per rsrc?')
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
						help='Do ypu want to generate a list of random IDs or read from an existing list?')
	parser.add_argument('-save_before', action='store',
						default=False,
						help='Do you want to save the results in a directory that already exists?')

	

	
	# Parses through the arguments and saves them within the keyword args
	arguments = parser.parse_args()
	return arguments







if __name__ == "__main__":
	start_time = time.time()
	print('Setting up the initial parameters...')
	args = parse_options()
	gen_ID_list = args.gen_ID_list
	save_before = args.save_before
	if gen_ID_list:
		list_ID = eval_maps_selection(num = 1000, seed = 33)
	else:
		if save_before:
			list_ID = np.zeros((100))

		else:
			list_ID = np.loadtxt(args.list_IDs_directory, dtype=int)

	output_direc = args.output_directory
	input_direc = args.input_directory
	plot_exp_conv = args.plot_exp_conv
	conv_AD_maps = args.conv_AD_maps
	single_model_read = args.single_model_read
	single_model_read_from_file = args.single_model_read_from_file
	multi_models_read_maps = args.multi_models_read_maps
	multi_models_read_lcs_from_file = args.multi_models_read_lcs_from_file
	multi_models_read_maps_from_file = args.multi_models_read_maps_from_file
	multi_models_read_lcs_and_maps_from_file = args.multi_models_read_lcs_and_maps_from_file
	model_ID = args.AD_model_ID
	model_file = args.AD_model_file_name
	cost_label = args.AD_model_cost_label
	lc_metric_calc = args.lc_metric_calc
	FID_metric_calc = args.FID_metric_calc
	KS_metric_calc = args.KS_metric_calc
	fit_lc_metric_calc = args.fit_lc_metric_calc
	gen_lc = args.gen_lc
	num_lc = int(args.num_lc)
	plot_exp_lc = args.plot_exp_lc
	lc_metric_plot = args.lc_metric_plot
	FID_metric_plot = args.FID_metric_plot
	KS_metric_plot = args.KS_metric_plot
	lc_metric_per_rsrc_plot = args.lc_metric_per_rsrc_plot
	FID_metric_per_rsrc_plot = args.FID_metric_per_rsrc_plot
	KS_metric_per_rsrc_plot = args.KS_metric_per_rsrc_plot
	fit_lc_metric_plot = args.fit_lc_metric_plot
	kg_fit_lc_metric_plot = args.kg_fit_lc_metric_plot
	verbose = args.verbose
	rsrcs = np.asarray([float(rsrc) for rsrc in args.rsrc.split(',')])
	n_models = len(models_info['data'])
	models = models_info['job_names']
	model_files = models_info['job_model_filename']
	cost_labels = models_info['job_cost_labels']
	lc_metrics = np.zeros((n_models, len(rsrcs)+1))
	FID_metrics = np.zeros((n_models, len(rsrcs)+1))
	KS_metrics = np.zeros((n_models, len(rsrcs)+1))
	fit_lc_metrics = np.zeros((n_models+1, len(rsrcs)+1, len(list_ID)))
	fit_lc_per_map_metrics = np.zeros((n_models+1, len(rsrcs)+1, len(list_ID)))
	all_lc_fit_lc_metric_one_map = np.zeros((n_models+1, len(rsrcs)+1, num_lc))
	shape_ = 1000
	save_lcs = args.save_lcs
	save_maps = args.save_maps
	date = args.date
	
	if not save_before:
		os.system("mkdir "+ str(output_direc) + str(date))
		output_direc = output_direc+str(date)+'/'
	rsrc_for_plot = 0





	# if save_all_LSR:
	# 	best_trial = 5
	# 	best_model_ID = models_info['job_names']
	# 	best_model_file = models_info['job_model_filename']
	# 	best_model_cost_label = models_info['job_cost_labels']
	# 	all_data = read_all_data()
	# 	partition = split_data(list(all_data.keys),
	# 							train_percentage = 0.99,
	# 		   					valid_percentage = 0.005)
	# 	data_generator = DataGenerator2(partition['train'], all_data, **params)
	# 	autoencoder = read_AD_model(model_ID, model_file, cost_label)
	# 	AD_maps = autoencoder.predict(data_generator)

	# 	bt_ly_name = bottleneck_layer_name[model_design]
	#     bottleneck_output = autoencoder.get_layer(bt_ly_name).output
	#     model_bottleneck = keras.models.Model(inputs=autoencoder.input, outputs=bottleneck_output)
	#     print('Generating the latent space representation for ' + str(len(test_set_index)) + ' test objects...')
	#     bottleneck_predictions = model_bottleneck.predict(test_generator)		

	if single_model_read:
		maps, AD_maps, lcs, AD_lcs =prepare_maps(list_ID, 
												 rsrcs,
												 model_ID, 
												 model_file, 
												 cost_label,
												 num_lc,
												 output_direc,
												 conv_AD_maps=conv_AD_maps,
												 plot_exp_conv=plot_exp_conv,
												 gen_lc=gen_lc,
												 plot_exp_lc=plot_exp_lc, 
												 verbose=verbose)

	elif single_model_read_from_file:

		maps = pkl.load(input_direc+'true_maps.pkl')
		AD_maps = pkl.load(input_direc+'AD_maps_%s.pkl'%(model_ID))
		lcs = pkl.load(input_direc+'true_lcs.pkl')
		AD_lcs = pkl.load(input_direc+'AD_lcs_%s.pkl'%(model_ID))

	

	if multi_models_read_maps:

		for m, model in enumerate(models):
			print ('Reading maps for model %s'%model)
			maps, AD_maps, lcs, AD_lcs =prepare_maps(list_ID, 
													 rsrcs,
													 model, 
													 model_files[m], 
													 cost_labels[m],
													 num_lc,
													 output_direc,
													 conv_AD_maps=conv_AD_maps,
													 plot_exp_conv=plot_exp_conv,
													 gen_lc=gen_lc,
													 plot_exp_lc=plot_exp_lc, 
												 	 verbose=verbose)
			
			if save_maps:
				pkl.dump(maps, 
					 open(output_direc+'true_maps_%s.pkl'%model, 'wb'))
				pkl.dump(AD_maps, 
					 open(output_direc+'AD_maps_%s.pkl'%model, 'wb'))

			if save_lcs:
				pkl.dump(lcs, 
					 open(output_direc+'lcs_%s.pkl'%model, 'wb'))
				pkl.dump(AD_lcs, 
					 open(output_direc+'AD_lcs_%s.pkl'%model, 'wb'))

			if lc_metric_calc:
				if verbose:
					print('Calculating lc metric...')
				lc_metrics[m, 0] = calculate_lc_metric([maps['true_maps'], 
														AD_maps['AD_maps']])
				for r, rsrc in enumerate(rsrcs):
					lc_metrics[m, r+1] = calculate_lc_metric([maps[str(rsrc)], 
															  AD_maps[str(rsrc)]])
				np.save(output_direc+'lc_metric',lc_metrics)

			if FID_metric_calc:
				if verbose:
					print('Calculating FID metric...')
				inceptionv3 = read_inceptionV3()
				FID_metrics[m, 0] = calculate_FID_metric(inceptionv3, 
											   maps['true_maps'], 
											   AD_maps['AD_maps'])
				for r, rsrc in enumerate(rsrcs):
					FID_metrics[m, r+1] = calculate_FID_metric(inceptionv3, 
													 maps[str(rsrc)], 
													 AD_maps[str(rsrc)])
				np.save(output_direc+'FID_metric',FID_metrics)

			if KS_metric_calc:
				if verbose:
					print('Calculating KS metric...')
				num_cores = mp.cpu_count()
				num_to_process = min(num_cores, len(list_ID))
				pool = mp.Pool(processes=num_to_process) 
				if verbose:
					print('Number of cores found for the KS test: ', mp.cpu_count())
	
				KS_metrics[m, 0] = np.mean(pool.map(process_KS, 
												 [[maps['true_maps'][i].flatten(),
												   AD_maps['AD_maps'][i].flatten()] for i in range(len(list_ID))]))

				for r, rsrc in enumerate(rsrcs):
					KS_metrics[m, r+1] = np.mean(pool.map(process_KS, 
																	 [[maps[str(rsrc)][i].flatten(),
																	   AD_maps[str(rsrc)][i].flatten()] for i in range(len(list_ID))]))
				np.save(output_direc+'KS_metric',KS_metrics)


		
		

		if lc_metric_plot:
			imgs = img_cut_out_generator(43657, rsrc=0, size=200)			
			metric_comparison_plot(lc_metrics, 
								   'Lightcurve Matching',
								   imgs,
								   output_direc)
		if lc_metric_per_rsrc_plot:
			metric_comparison_per_rsrc_plot(np.concatenate(([0], rsrcs), axis=0),
										   lc_metrics, 
										   'Lightcurve Matching',
										   output_direc)
		if FID_metric_plot:
			imgs = img_cut_out_generator(43657, rsrc=0, size=200)			
			metric_comparison_plot(FID_metrics, 
								   'FID',
								   imgs,
								   output_direc)
		if FID_metric_per_rsrc_plot:
			metric_comparison_per_rsrc_plot(np.concatenate(([0], rsrcs), axis=0),
										   FID_metrics, 
										   'FID',
										   output_direc)

		if KS_metric_plot:
			imgs = img_cut_out_generator(43657, rsrc=0, size=200)			
			metric_comparison_plot(KS_metrics, 
								   'KS',
								   imgs,
								   output_direc)
		if KS_metric_per_rsrc_plot:
			metric_comparison_per_rsrc_plot(np.concatenate(([0], rsrcs), axis=0),
										   KS_metrics, 
										   'KS',
										   output_direc)

	elif multi_models_read_lcs_from_file:
		for m, model in enumerate(models):
			print ('Reading lcs for model %s from file...'%model)
			lcs = pkl.load(open(input_direc+'lcs_%s.pkl'%model, 'rb'))
			AD_lcs = pkl.load(open(input_direc+'AD_lcs_%s.pkl'%model, 'rb'))
			list_ID = lcs['ID']

		if lc_metric_calc:
			if verbose:
				print('Calculating lc metric...')
			lc_metrics[m, 0] = calculate_lc_metric([lcs['true_lcs'], 
													AD_lcs['AD_lcs']])
			for r, rsrc in enumerate(rsrcs):
				lc_metrics[m, r+1] = calculate_lc_metric([lcs['conv_lcs_'+str(rsrc)], 
															  AD_lcs['conv_lcs_'+str(rsrc)]])

			np.save(output_direc+'lc_metric',lc_metrics)
			
			if lc_metric_plot:
				imgs = img_cut_out_generator(43657, rsrc=0, size=200)			
				metric_comparison_plot(lc_metrics,
									   'Lightcurve Matching', 
									   imgs,
									   output_direc)

	elif multi_models_read_lcs_and_maps_from_file:
		for m, model in enumerate(models):
			print ('Reading maps and lcs for model %s from file...'%model)
			maps = pkl.load(open(input_direc+'true_maps_%s.pkl'%model, 'rb'))
			AD_maps = pkl.load(open(input_direc+'AD_maps_%s.pkl'%model, 'rb'))
			lcs = pkl.load(open(input_direc+'lcs_%s.pkl'%model, 'rb'))
			AD_lcs = pkl.load(open(input_direc+'AD_lcs_%s.pkl'%model, 'rb'))
			list_ID = maps['ID']

			if plot_exp_conv:
				ID1 = list_ID[0]
				map1 = maps['true_maps'][maps['ID']==ID1][0]
				ADmap1 = AD_maps['AD_maps'][AD_maps['ID']==ID1][0]
				for rsrc in rsrcs:
					map1_conv = maps[str(rsrc)][maps['ID']==ID1][0]
					plot_example_conv(map1, map1_conv, 
									  output_direc+'%i_%.1f_%s_exp.png'%(ID1, 
																		 rsrc, 
																		 model))
					ADmap1_conv = AD_maps[str(rsrc)][AD_maps['ID']==ID1][0]
					plot_example_conv(ADmap1, ADmap1_conv, 
									  output_direc+'%i_%.1f_%s_AD_exp.png'%(ID1, 
																		 rsrc, 
																		 model))
			if plot_exp_lc:
				ID1 = list_ID[0]

				map1 = maps['true_maps'][maps['ID']==ID1][0]
				map1_conv = maps[str(rsrc)][maps['ID']==ID1][0]
				lc_set1 = lcs['true_lcs'][lcs['ID']==ID1][0][0]
				lc_set2 = lcs['conv_lcs_'+str(rsrc)][lcs['ID']==ID1][0][0]
				origin1 = lcs['true_lcs_orig'][lcs['ID']==ID1][0][0]
				origin2 = lcs['conv_lcs_orig_'+str(rsrc)][lcs['ID']==ID1][0][0]
				



				ADmap1 = AD_maps['AD_maps'][AD_maps['ID']==ID1][0]
				ADmap1_conv = AD_maps[str(rsrc)][AD_maps['ID']==ID1][0]
				ADlc_set1 = AD_lcs['AD_lcs'][AD_lcs['ID']==ID1][0][0]
				ADlc_set2 = AD_lcs['conv_lcs_'+str(rsrc)][AD_lcs['ID']==ID1][0][0]
				ADorigin1 = AD_lcs['AD_lcs_orig'][AD_lcs['ID']==ID1][0][0]
				ADorigin2 = AD_lcs['conv_lcs_orig_'+str(rsrc)][AD_lcs['ID']==ID1][0][0]
				

				for rsrc in rsrcs:
					lc_direc = output_direc+'map_%i_%.1f_%s_exp_lc.png'%(list_ID[0], rsrc, model)
					ADlc_direc = output_direc+'AD_map_%i_%.1f_%s_exp_lc.png'%(list_ID[0], rsrc, model)
					plot_example_lc(ID1,
									[map1, map1_conv], 
									[lc_set1, lc_set2], 
									[origin1, origin2], 
									rsrc, 
									lc_direc)

					plot_example_lc(ID1,
									[ADmap1, ADmap1_conv], 
									[ADlc_set1, ADlc_set2], 
									[ADorigin1, ADorigin2], 
									rsrc, 
									ADlc_direc)

			if lc_metric_calc:
				if verbose:
					print('Calculating lc metric...')
				lc_metrics[m, 0] = calculate_lc_metric([lcs['true_lcs'], 
														AD_lcs['AD_lcs']])
				for r, rsrc in enumerate(rsrcs):
					lc_metrics[m, r+1] = calculate_lc_metric([lcs['conv_lcs_'+str(rsrc)], 
															  AD_lcs['conv_lcs_'+str(rsrc)]])
				np.save(output_direc+'lc_metric',lc_metrics)

			if FID_metric_calc:
				if verbose:
					print('Calculating FID metric...')
				inceptionv3 = read_inceptionV3()
				FID_metrics[m, 0] = calculate_FID_metric(inceptionv3, 
											   maps['true_maps'], 
											   AD_maps['AD_maps'])
				for r, rsrc in enumerate(rsrcs):
					FID_metrics[m, r+1] = calculate_FID_metric(inceptionv3, 
													 maps[str(rsrc)], 
													 AD_maps[str(rsrc)])
				np.save(output_direc+'FID_metric',FID_metrics)

			if KS_metric_calc:
				if verbose:
					print('Calculating KS metric...')
				num_cores = mp.cpu_count()
				num_to_process = min(num_cores, len(list_ID))
				pool = mp.Pool(processes=num_to_process) 
				if verbose:
					print('Number of cores found for the KS test: ', mp.cpu_count())
				

				KS_metrics[m, 0] = np.mean(pool.map(process_KS, 
									 [[maps['true_maps'][i].flatten(), 
									   AD_maps['AD_maps'][i].flatten()] for i in range(len(list_ID))]))
				for r, rsrc in enumerate(rsrcs):
					KS_metrics[m, r+1] = np.mean(pool.map(process_KS, 
									 [[maps[str(rsrc)][i].flatten(), 
									   AD_maps[str(rsrc)][i].flatten()] for i in range(len(list_ID))]))
				np.save(output_direc+'KS_metric',KS_metrics)


		if lc_metric_plot:
			imgs = img_cut_out_generator(43657, rsrc=0, size=200)			
			metric_comparison_plot(lc_metrics,
								   'Lightcurve Matching', 
								   imgs,
								   output_direc)
		if lc_metric_per_rsrc_plot:
			metric_comparison_per_rsrc_plot(np.concatenate(([0], rsrcs), axis=0),
										   lc_metrics, 
										   'Lightcurve Matching',
										   output_direc)
		if FID_metric_plot:
			imgs = img_cut_out_generator(43657, rsrc=0, size=200)			
			metric_comparison_plot(FID_metrics, 
								   'FID',
								   imgs,
								   output_direc)
		if FID_metric_per_rsrc_plot:
			metric_comparison_per_rsrc_plot(np.concatenate(([0], rsrcs), axis=0),
										   FID_metrics, 
										   'FID',
										   output_direc)
		if KS_metric_plot:
			imgs = img_cut_out_generator(43657, rsrc=0, size=200)			
			metric_comparison_plot(KS_metrics, 
								   'KS',
								   imgs,
								   output_direc)
		if KS_metric_per_rsrc_plot:
			metric_comparison_per_rsrc_plot(np.concatenate(([0], rsrcs), axis=0),
										   KS_metrics, 
										   'KS',
										   output_direc)


	elif multi_models_read_maps_from_file:

		for m, model in enumerate(models):
			print ('Reading maps for model %s from file...'%model)
			maps = pkl.load(open(input_direc+'true_maps_%s.pkl'%model, 'rb'))
			AD_maps = pkl.load(open(input_direc+'AD_maps_%s.pkl'%model, 'rb'))
			list_ID = maps['ID']
		
			if gen_lc:
				lcs = {}
				lcs['ID'] = np.asarray(list_ID)
				AD_lcs = {}
				AD_lcs['ID'] = np.asarray(list_ID)
				num_cores = mp.cpu_count()
				num_to_process = min(num_cores, len(list_ID))
				pool = mp.Pool(processes=num_to_process)
				if len(rsrcs)!=0:	
					for r, rsrc in enumerate(rsrcs):			
						lc_temp = pool.map(process_lc, 
										[[maps[str(rsrc)][i], num_lc] for i in range(len(list_ID))]) 
						AD_lc_temp = pool.map(process_lc, 
										[[AD_maps[str(rsrc)][i], num_lc] for i in range(len(list_ID))]) 

						if r == 0:
							start_time = time.time()
							lc_temp_true = pool.map(process_lc, 
										[[maps['true_maps'][i], num_lc] for i in range(len(list_ID))])
							lcs['true_lcs'] = np.asarray([lc_temp_true[i][0] for i in range(len(list_ID))])
							lcs['true_lcs_orig'] = np.asarray([lc_temp_true[i][1] for i in range(len(list_ID))])
							end_time = time.time()  # Record the end time
							total_runtime = end_time - start_time 
							print ('Total time to %i lightcurves for %i maps = %.1f seconds'%(num_lc, len(list_ID), total_runtime))

							
							AD_lc_temp_true = pool.map(process_lc, 
										[[AD_maps['AD_maps'][i], num_lc] for i in range(len(list_ID))])
							AD_lcs['AD_lcs'] = np.asarray([AD_lc_temp_true[i][0] for i in range(len(list_ID))])
							AD_lcs['AD_lcs_orig'] = np.asarray([AD_lc_temp_true[i][1] for i in range(len(list_ID))])

						lcs['conv_lcs_'+str(rsrc)] = np.asarray([lc_temp[i][0] for i in range(len(list_ID))])
						lcs['conv_lcs_orig_'+str(rsrc)] = np.asarray([lc_temp[i][1] for i in range(len(list_ID))])
						AD_lcs['conv_lcs_'+str(rsrc)] = np.asarray([AD_lc_temp[i][0] for i in range(len(list_ID))])
						AD_lcs['conv_lcs_orig_'+str(rsrc)] = np.asarray([AD_lc_temp[i][1] for i in range(len(list_ID))])

				else:
					lc_temp_true = pool.map(process_lc, 
										[[maps['true_maps'][i], num_lc] for i in range(len(list_ID))])
					lcs['true_lcs'] = np.asarray(lc_temp_true)[:, 0]
					lcs['true_lcs_orig'] = np.asarray(lc_temp_true)[:, 1]

					AD_lc_temp_true = pool.map(process_lc, 
										[[AD_maps['AD_maps'][i], num_lc] for i in range(len(list_ID))])
					AD_lcs['AD_lcs'] = np.asarray(AD_lc_temp_true)[:, 0]
					AD_lcs['AD_lcs_orig'] = np.asarray(AD_lc_temp_true)[:, 1]

				if save_lcs:
					pkl.dump(lcs, 
					open(output_direc+'lcs_%s_num%i.pkl'%(model, num_lc), 'wb'))
					pkl.dump(AD_lcs, 
					open(output_direc+'AD_lcs_%s_num%i.pkl'%(model, num_lc), 'wb'))

	if fit_lc_metric_calc:
		lcs = pkl.load(open(input_direc+'lcs_%s_num10000.pkl'%models[0], 'rb'))
		list_ID = lcs['ID']

		ID_ = 2
		ID_picked0 = list_ID[ID_]
		print(ID_picked0)

		if fit_lc_metric_calc_with_uncertainty:
			steps = 10000
			pick_map_id = 0
			
			for s in range(steps):
				temp_picked_lc = lcs['true_lcs'][0][s, 1, :]
				temp_other_lcs = lcs['true_lcs'][0][:, 1, :][~s]

				all_lc_fit_lc_metric_one_map[0, 0, s] = 


		maps_sel, AD_maps_empty, lcs_sel, AD_lcs_empty =prepare_maps([ID_picked0], 
												 rsrcs,
												 model_ID= [], 
												 model_file = [], 
												 cost_label = [],
												 num_lc=1,
												 output_direc=output_direc,
												 conv_AD_maps=False,
												 plot_exp_conv=False,
												 gen_lc=True,
												 plot_exp_lc=False, 
												 save_maps=False,
												 save_lcs=False,
												 verbose=False)
		picked_lc0 = lcs_sel['true_lcs'][0][0, 1, :]
		plot_axis = 'kg'

		if kg_fit_lc_metric_plot:
			print('Printing %s plots...'%(plot_axis))

		num_cores = mp.cpu_count()
		num_to_process = min(num_cores, len(list_ID))
		pool = mp.Pool(processes=num_to_process)
		
		for m, model in enumerate(models):
			print ('Reading lcs for model %s from file...'%model)
			AD_lcs = pkl.load(open(input_direc+'AD_lcs_%s_num10000.pkl'%model, 'rb'))
			

			kgs = np.zeros((len(list_ID), 3))
			for i, ID in enumerate(list_ID):
				_, k, g, s, _, _ = read_map_meta(ID)
				kgs[i] = [k, g, s]

			if m == 0:
				print('The chosen lightcurve is from unconvolved map with ID=%i'%ID_picked0)

				lc_temp0 = pool.map(process_fit_lc, 
							[[picked_lc0, lcs['true_lcs'][i][:, 1, :]] for i in range(len(list_ID))])
				lc_temp01 = pool.map(process_fit_lc, 
							[[lcs['true_lcs'][i][0, 1, :], lcs['true_lcs'][i][1:, 1, :]] for i in range(len(list_ID))])
		
				fit_lc_metrics[0, 0, :] = np.asarray([lc_temp0[i][0] for i in range(len(list_ID))])
				fit_lc_per_map_metrics[0, 0, :] = np.asarray([lc_temp01[i][0] for i in range(len(list_ID))])
				fit_lc_metrics_argmins = np.asarray([lc_temp0[i][1] for i in range(len(list_ID))])
				
				if kg_fit_lc_metric_plot:
					kgs_ref = kgs[ID_]
					print('Referenc kgs is %.1f, %.1f, %.1f'%(k, g, s))
					metrics_tmp = np.log10((fit_lc_metrics[0, 0, :]/fit_lc_per_map_metrics[0, 0, :]))
					metrics_per_map_tmp = ((fit_lc_per_map_metrics[0, 0, :]/fit_lc_per_map_metrics[0, 0, :]))
					k_closest = kgs[:,0][np.argmin(metrics_tmp)]
					g_closest = kgs[:,1][np.argmin(metrics_tmp)]
					s_closest = kgs[:,2][np.argmin(metrics_tmp)]
					kgs_closest = np.array([k_closest,
											g_closest,
											s_closest])
					title = plot_axis+'_fitted_true_lc_rsrc%.1f.png'%(0)
					title_per_map = plot_axis+'_fitted_true_lc_per_map_rsrc%.1f.png'%(0)

					print('Title is %s'%title)
					plot_kgs_lc_distance_metric(kgs=kgs,
							    lc_metric=metrics_tmp, 
							    kgs_ref=kgs_ref, 
							    kgs_closest=kgs_closest, 
							    output_direc=output_direc,
							    plot_name=title,
							    axis = plot_axis, 
							    type_= 'scatter'
							    )
					plot_kgs_lc_distance_metric(kgs=kgs,
							    lc_metric=metrics_per_map_tmp, 
							    kgs_ref=np.nan, 
							    kgs_closest=kgs_closest, 
							    output_direc=output_direc,
							    plot_name=title_per_map,
							    axis = plot_axis, 
							    type_= 'scatter'
							    )


			if fit_lc_metric_plot:
				fig, axs = plt.subplots(len(rsrcs)+1, 1, figsize = (20, 7))

			lc_temp2 = pool.map(process_fit_lc, 
								[[picked_lc0, AD_lcs['AD_lcs'][i][:, 1, :]] for i in range(len(list_ID))])
			lc_temp21 = pool.map(process_fit_lc, 
								[[lcs['true_lcs'][i][0, 1, :], AD_lcs['AD_lcs'][i][:, 1, :]] for i in range(len(list_ID))]) 

			fit_lc_metrics[m+1, 0, :] = np.asarray([lc_temp2[i][0] for i in range(len(list_ID))])
			fit_lc_per_map_metrics[m+1, 0, :] = np.asarray([lc_temp21[i][0] for i in range(len(list_ID))])
			fit_lc_metrics_argmins2 = np.asarray([lc_temp2[i][1] for i in range(len(list_ID))])

			if kg_fit_lc_metric_plot:

				kgs_ref = kgs[ID_]
				metrics_tmp = np.log10((fit_lc_metrics[m+1, 0, :]/fit_lc_per_map_metrics[0, 0, :]))
				metrics_per_map_tmp = ((fit_lc_per_map_metrics[m+1, 0, :]/fit_lc_per_map_metrics[0, 0, :]))
				k_closest = kgs[:,0][np.argmin(metrics_tmp)]
				g_closest = kgs[:,1][np.argmin(metrics_tmp)]
				s_closest = kgs[:,2][np.argmin(metrics_tmp)]
				kgs_closest = np.array([k_closest,
										g_closest,
										s_closest])
				title = plot_axis+'_fitted_AD_lc_%s_rsrc%.1f.png'%(model, 0)
				title_per_map = plot_axis+'_fitted_AD_lc_per_map_%s_rsrc%.1f.png'%(model, 0)

				plot_kgs_lc_distance_metric(kgs=kgs,
						    lc_metric=metrics_tmp, 
						    kgs_ref=kgs_ref, 
						    kgs_closest=kgs_closest, 
						    output_direc=output_direc,
						    plot_name=title,
						    axis = plot_axis, 
						    type_= 'scatter'
						    )		
				plot_kgs_lc_distance_metric(kgs=kgs,
							    lc_metric=metrics_per_map_tmp, 
							    kgs_ref=np.nan, 
							    kgs_closest=kgs_closest, 
							    output_direc=output_direc,
							    plot_name=title_per_map,
							    axis = plot_axis, 
							    type_= 'scatter'
							    )		

			for r, rsrc in enumerate(rsrcs):
				picked_lc1 = lcs_sel['conv_lcs_'+str(rsrc)][0][0, 1, :]
				ID_picked1 = ID_picked0
				print('The chosen lightcurve is from convolved map (rsrc=%.1f) with ID=%i'%(rsrc, ID_picked1))
				
				if m==0:
					lc_temp1 = pool.map(process_fit_lc, 
										[[picked_lc1, lcs['conv_lcs_'+str(rsrc)][i][:, 1, :]] for i in range(len(list_ID))])
					lc_temp11 = pool.map(process_fit_lc, 
										[[lcs['conv_lcs_'+str(rsrc)][i][0, 1, :], lcs['conv_lcs_'+str(rsrc)][i][1:, 1, :]] for i in range(len(list_ID))]) 
					fit_lc_metrics[0, r+1, :] = np.asarray([lc_temp1[i][0] for i in range(len(list_ID))])
					fit_lc_per_map_metrics[0, r+1, :] = np.asarray([lc_temp11[i][0] for i in range(len(list_ID))])
					fit_lc_metrics_argmins1 = np.asarray([lc_temp1[i][1] for i in range(len(list_ID))])
					if kg_fit_lc_metric_plot:
						kgs_ref = kgs[ID_]
						metrics_tmp = np.log10((fit_lc_metrics[0, r+1, :]/fit_lc_per_map_metrics[0, r+1, :]))
						metrics_per_map_tmp = ((fit_lc_per_map_metrics[0, r+1, :]/fit_lc_per_map_metrics[0, r+1, :]))
						k_closest = kgs[:,0][np.argmin(metrics_tmp)]
						g_closest = kgs[:,1][np.argmin(metrics_tmp)]
						s_closest = kgs[:,2][np.argmin(metrics_tmp)]
						kgs_closest = np.array([k_closest,
												g_closest,
												s_closest])
						title = plot_axis+'_fitted_true_lc_rsrc%.1f.png'%(rsrc)
						title_per_map = plot_axis+'_fitted_true_lc_per_map_rsrc%.1f.png'%(rsrc)

						plot_kgs_lc_distance_metric(kgs=kgs,
								    lc_metric=metrics_tmp, 
								    kgs_ref=kgs_ref, 
								    kgs_closest=kgs_closest, 
								    output_direc=output_direc,
								    plot_name=title,
								    axis = plot_axis, 
								    type_= 'scatter'
								    )
						plot_kgs_lc_distance_metric(kgs=kgs,
							    lc_metric=metrics_per_map_tmp, 
							    kgs_ref=np.nan, 
							    kgs_closest=kgs_closest, 
							    output_direc=output_direc,
							    plot_name=title_per_map,
							    axis = plot_axis, 
							    type_= 'scatter'
							    )
						
				
				lc_temp3 = pool.map(process_fit_lc, 
									[[picked_lc1, AD_lcs['conv_lcs_'+str(rsrc)][i][:, 1, :]] for i in range(len(list_ID))]) 
				lc_temp31 = pool.map(process_fit_lc, 
									[[lcs['conv_lcs_'+str(rsrc)][i][0, 1, :], AD_lcs['conv_lcs_'+str(rsrc)][i][:, 1, :]] for i in range(len(list_ID))]) 
				fit_lc_metrics[m+1, r+1, :] = np.asarray([lc_temp3[i][0] for i in range(len(list_ID))])
				fit_lc_per_map_metrics[m+1, r+1, :] = np.asarray([lc_temp31[i][0] for i in range(len(list_ID))])
				fit_lc_metrics_argmins3 = np.asarray([lc_temp3[i][1] for i in range(len(list_ID))])
				if kg_fit_lc_metric_plot:
					kgs_ref = kgs[ID_]
					metrics_tmp = np.log10((fit_lc_metrics[m+1, r+1, :]/fit_lc_per_map_metrics[0, r+1, :]))
					metrics_per_map_tmp = ((fit_lc_per_map_metrics[m+1, r+1, :]//fit_lc_per_map_metrics[0, r+1, :]))
					k_closest = kgs[:,0][np.argmin(metrics_tmp)]
					g_closest = kgs[:,1][np.argmin(metrics_tmp)]
					s_closest = kgs[:,2][np.argmin(metrics_tmp)]
					kgs_closest = np.array([k_closest,
											g_closest,
											s_closest])
					title = plot_axis+'_fitted_AD_lc_%s_rsrc%.1f.png'%(model, rsrc)
					title_per_map = plot_axis+'_fitted_AD_lc_per_map_%s_rsrc%.1f.png'%(model, rsrc)

					plot_kgs_lc_distance_metric(kgs=kgs,
							    lc_metric=metrics_tmp, 
							    kgs_ref=kgs_ref, 
							    kgs_closest=kgs_closest, 
							    output_direc=output_direc,
							    plot_name=title,
							    axis = plot_axis, 
							    type_= 'scatter'
							    )
					plot_kgs_lc_distance_metric(kgs=kgs,
							    lc_metric=metrics_per_map_tmp, 
							    kgs_ref=np.nan, 
							    kgs_closest=kgs_closest, 
							    output_direc=output_direc,
							    plot_name=title_per_map,
							    axis = plot_axis, 
							    type_= 'scatter'
							    )					
			
			np.save(output_direc+'fit_lc_metrics',fit_lc_metrics)
			np.save(output_direc+'fit_lc_per_map_metrics',fit_lc_per_map_metrics)
			
			print('For model %s:\n'%model)
			ID_pred_true = list_ID[np.argmin(fit_lc_metrics[0, 0, :])]
			ID_pred_AD = list_ID[np.argmin(fit_lc_metrics[m+1, 0, :])]
			if fit_lc_metric_plot:
				axs[0].plot(lcs_sel['true_lcs'][0][0, 0, :],
							lcs_sel['true_lcs'][0][0, 1, :],
							'-',
							color = '#59b359',
							label='ID=%i Picked Light Curves'%lcs_sel['ID'][0])
				# axs[0].plot(AD_lcs['AD_lcs'][ID_][0, 0, :],
				# 			AD_lcs['AD_lcs'][ID_][0, 1, :],
				# 			'-',
				# 			color = '#ffaa00',
				# 			label='ID=%i AD version of Picked Light Curves'%AD_lcs['ID'][ID_])
				argmin_lc = fit_lc_metrics_argmins[np.argmin(fit_lc_metrics[0, 0, :])]
				axs[0].plot(lcs['true_lcs'][np.argmin(fit_lc_metrics[0, 0, :])][argmin_lc, 0, :],
							lcs['true_lcs'][np.argmin(fit_lc_metrics[0, 0, :])][argmin_lc, 1, :],
							'-',
							color = '#502db3',
							label = 'ID=%i Predicted by true maps'%ID_pred_true)
				argmin_AD_lc = fit_lc_metrics_argmins2[np.argmin(fit_lc_metrics[m+1, 0, :])]
				axs[0].plot(AD_lcs['AD_lcs'][np.argmin(fit_lc_metrics[m+1, 0, :])][argmin_AD_lc, 0, :],
							AD_lcs['AD_lcs'][np.argmin(fit_lc_metrics[m+1, 0, :])][argmin_AD_lc, 1, :],
							'-',
							color = '#e51f00',
							label = 'ID=%i Predicted by AD maps'%ID_pred_AD)
				axs[0].legend()
				axs[0].set_title('Selected ID=%i'%list_ID[ID_])

			true_params = read_map_meta(ID_picked0)[1:4]
			pred_true_params = read_map_meta(ID_pred_true)[1:4]
			pred_AD_params = read_map_meta(ID_pred_AD)[1:4]
			print('Unconvolved maps:\n'+ 
				  'True ID = %i, k=%.1f, g=%.1f, s=%.1f \n'%(ID_picked0, true_params[0], true_params[1], true_params[2])+
				  'Predicted ID from true maps = %i, k=%.1f, g=%.1f, s=%.1f \n'%(ID_pred_true, pred_true_params[0], pred_true_params[1], pred_true_params[2])+
				  'Predicted ID from AD maps = %i, k=%.1f, g=%.1f, s=%.1f \n'%(ID_pred_AD, pred_AD_params[0], pred_AD_params[1], pred_AD_params[2]))
			for r, rsrc in enumerate(rsrcs):
				ID_pred_true = list_ID[np.argmin(fit_lc_metrics[0, r+1, :])]
				ID_pred_AD = list_ID[np.argmin(fit_lc_metrics[m+1, r+1, :])]
				if fit_lc_metric_plot:
					axs[r+1].plot(lcs_sel['conv_lcs_'+str(rsrc)][0][0, 0, :],
								lcs_sel['conv_lcs_'+str(rsrc)][0][0, 1, :],
								'-',
								color = '#59b359',
								label='ID=%i Picked Light Curves'%lcs_sel['ID'][0])
					# axs[r+1].plot(AD_lcs['conv_lcs_'+str(rsrc)][ID_][0, 0, :],
					# 			AD_lcs['conv_lcs_'+str(rsrc)][ID_][0, 1, :],
					# 			'-',
					# 			color = '#ffaa00',
					# 			label='ID=%i AD version of Picked Light Curves'%AD_lcs['ID'][ID_])
					argmin_lc = fit_lc_metrics_argmins1[np.argmin(fit_lc_metrics[0, r+1, :])]
					axs[r+1].plot(lcs['conv_lcs_'+str(rsrc)][np.argmin(fit_lc_metrics[0, r+1, :])][argmin_lc, 0, :],
								lcs['conv_lcs_'+str(rsrc)][np.argmin(fit_lc_metrics[0, r+1, :])][argmin_lc, 1, :],
								'-',
								color = '#502db3',
								label = 'ID=%i Predicted by true maps'%ID_pred_true)
					argmin_AD_lc = fit_lc_metrics_argmins3[np.argmin(fit_lc_metrics[m+1, r+1, :])]
					axs[r+1].plot(AD_lcs['conv_lcs_'+str(rsrc)][np.argmin(fit_lc_metrics[m+1, r+1, :])][argmin_AD_lc, 0, :],
								AD_lcs['conv_lcs_'+str(rsrc)][np.argmin(fit_lc_metrics[m+1, r+1, :])][argmin_AD_lc, 1, :],
								'-',
								color = '#e51f00',
								label = 'ID=%i Predicted by AD maps'%ID_pred_AD)
					axs[r+1].legend()
				true_params = read_map_meta(ID_picked1)[1:4]
				pred_true_params = read_map_meta(ID_pred_true)[1:4]
				pred_AD_params = read_map_meta(ID_pred_AD)[1:4]
				print('Convolved maps with rsrc=%.1f:\n'%rsrc+ 
					  'True ID = %i, k=%.1f, g=%.1f, s=%.1f \n'%(ID_picked1, true_params[0], true_params[1], true_params[2])+
					  'Predicted ID from true maps = %i, k=%.1f, g=%.1f, s=%.1f \n'%(ID_pred_true, pred_true_params[0], pred_true_params[1], pred_true_params[2])+
					  'Predicted ID from AD maps = %i, k=%.1f, g=%.1f, s=%.1f \n'%(ID_pred_AD, pred_AD_params[0], pred_AD_params[1], pred_AD_params[2]))
			if fit_lc_metric_plot:
				fig.savefig(output_direc+'fitted_lc_%s.png'%model)


	# if read_metrics:

	read_me_creator(input_direc,
					output_direc, 
					len(list_ID),
					num_lc,
					list_ID,
					rsrcs,
					args)

	end_time = time.time()  # Record the end time
	total_runtime = end_time - start_time 
	print ('Total time to read and convolve %i maps at %i source sizes and generate %i lightcurves for each of them is %.1f seconds'%(len(list_ID), len(rsrcs), num_lc, total_runtime))
	




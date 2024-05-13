from maps_util import *
from convolver import convolver
import multiprocessing as mp
import time
import pickle as pkl
from best_models_info import models_info
import os
from conv_utils import *
from plotting_utils import plot_example_lc, plot_example_conv
from best_models_info import models_info


def process_read(ID):
	map_ = read_binary_map(ID)
	mag_convertor = read_map_meta(ID)[0]
	# print(mag_convertor)
	map_ = norm_maps(map_, 
					 mag_convertor)

	return map_

def process_conv(ls):
	map_ = ls[0]
	rsrc = ls[1]
	map_conv = convolve_map(map_, 
							rsrc)

	return map_conv

def process_lc(ls):
	map_ = ls[0]
	n_lc = ls[1]
	(lc_set, origin_end) = lc_gen_set(n_lc,
								   map_,
								   nxpix=1000,
								   nypix=1000,
								   factor=1,
								   tot_r_e=12.5,
								   nsamp = 100,
								   distance = 10, # in maps units
								   dt = (22*365)/1.5,
								   set_seed = 1)

	return [lc_set, origin_end]


def prepare_maps(list_ID, 
				 rsrcs,
				 model_ID, 
				 model_file, 
				 cost_label,
				 num_lc,
				 output_direc,
				 conv_AD_maps=True,
				 plot_exp_conv=False,
				 gen_lc=True,
				 plot_exp_lc=False,
				 save_maps = True,
				 save_lcs = True,
				 verbose=False):

	num_cores = mp.cpu_count()
	num_to_process = min(num_cores, len(list_ID))
	pool = mp.Pool(processes=num_to_process) 
	print('Number of cores found: ', mp.cpu_count())

	maps_dict = {}
	maps_dict['ID'] = np.asarray(list_ID)
	AD_maps_dict = {}
	AD_maps_dict['ID'] = np.asarray(list_ID)

	lcs_dict = {}
	lcs_dict['ID'] = np.asarray(list_ID)
	AD_lcs_dict = {}
	AD_lcs_dict['ID'] = np.asarray(list_ID)
	if verbose:
		print('Reading all the true maps...')
	temp0 = pool.map(process_read, 
					[ID for ID in list_ID])
	maps_dict['true_maps'] = np.asarray(temp0)

	if len(rsrcs)!=0:
		for r, rsrc in enumerate(rsrcs):
			if verbose:
				print('Convolving the true maps with a source size of %.1f R_E.'%rsrc)

			temp = pool.map(process_conv, 
							[[maps_dict['true_maps'][i], rsrc] for i in range(len(list_ID))])
			maps_dict[str(rsrc)] = np.asarray(temp)
							

			if plot_exp_conv:
				ID1 = list_ID[0]
				map1 = maps_dict['true_maps'][maps_dict['ID']==ID1][0]
				map1_conv = maps_dict[str(rsrc)][maps_dict['ID']==ID1][0]

				plot_example_conv(map1, map1_conv, 
								  output_direc+'%i_%.1f_%s_exp.png'%(list_ID[0], 
																	 rsrc, 
																	 model_ID))
			if gen_lc:
				if verbose:
					print('Generating %i lightcurves for the true maps convolved with a source size of %.1f R_E.'%(num_lc, rsrc))
				start_time = time.time()
				lc_temp = pool.map(process_lc, 
								[[maps_dict[str(rsrc)][i], num_lc] for i in range(len(list_ID))]) 

				if r == 0:
					if verbose:
						print('Generating %i lightcurves for the unconvolved true maps'%(num_lc))
					lc_temp_true = pool.map(process_lc, 
											[[maps_dict['true_maps'][i], num_lc] for i in range(len(list_ID))])
					lcs_dict['true_lcs'] = np.asarray([lc_temp_true[i][0] for i in range(len(list_ID))])
					lcs_dict['true_lcs_orig'] = np.asarray([lc_temp_true[i][1] for i in range(len(list_ID))])

				lcs_dict['conv_lcs_'+str(rsrc)] = np.asarray([lc_temp[i][0] for i in range(len(list_ID))])
				lcs_dict['conv_lcs_orig_'+str(rsrc)] = np.asarray([lc_temp[i][0] for i in range(len(list_ID))])

				end_time = time.time()  # Record the end time
				total_runtime = end_time - start_time
				if verbose: 
					print ('time for %s lightcurves at source size %.1f is %.1f seconds'%(num_lc, rsrc, total_runtime))

				

				if plot_exp_lc:
					if verbose:
						print('Generating example plots of the produced lightcurves for rsrc=%.1f'%rsrc)
					ID1 = list_ID[0]
					map1 = maps_dict['true_maps'][maps_dict['ID']==ID1][0]
					map1_conv = maps_dict[str(rsrc)][maps_dict['ID']==ID1][0]
					lc_set1 = lcs_dict['true_lcs'][lcs_dict['ID']==ID1][0][0]
					lc_set2 = lcs_dict['conv_lcs_'+str(rsrc)][lcs_dict['ID']==ID1][0][0]
					origin1 = lcs_dict['true_lcs_orig'][lcs_dict['ID']==ID1][0][0]
					origin2 = lcs_dict['conv_lcs_orig_'+str(rsrc)][lcs_dict['ID']==ID1][0][0]
					lc_direc = output_direc+'map_%i_%.1f_%s_exp_lc.png'%(list_ID[0], rsrc, model_ID)
					plot_example_lc(list_ID[0],
									[map1, map1_conv], 
									[lc_set1, lc_set2], 
									[origin1, origin2], 
									rsrc, 
									lc_direc)
	else:

		if gen_lc:
			if verbose:
				print('Generating %i lightcurves for the unconvolved true maps'%(num_lc))

			lc_temp_true = pool.map(process_lc, 
									[[maps_dict['true_maps'][i], num_lc] for i in range(len(list_ID))])
			lcs_dict['true_lcs'] = np.asarray([lc_temp_true[i][0] for i in range(len(list_ID))])
			lcs_dict['true_lcs_orig'] = np.asarray([lc_temp_true[i][1] for i in range(len(list_ID))])

	if save_maps:
		if verbose:
			print('Saving the maps...')
		pkl.dump(maps_dict, 
			 open(output_direc+'true_maps.pkl', 'wb'))
	if gen_lc and save_lcs:
		if verbose:
			print('Saving the lightcurves...')
		pkl.dump(lcs_dict, 
		 open(output_direc+'true_lcs.pkl', 'wb'))


	if conv_AD_maps:
		if verbose:
			print('Predicting AD maps for model %s.'%model_ID)
		normed_maps = maps_dict['true_maps']
		# print(normed_maps.shape)
		shape_ = normed_maps[0].shape[0]
		autoencoder = read_AD_model(model_ID, model_file, cost_label)
		AD_maps = autoencoder.predict(normed_maps.reshape((len(list_ID), shape_, shape_, 1)))
		AD_maps = AD_maps.reshape((len(list_ID), shape_, shape_))

		AD_maps_dict['AD_maps'] = AD_maps

		if len(rsrcs)!=0:

			for r, rsrc in enumerate(rsrcs):

				if verbose:
					print('Convolving AD maps for model %s with a source size of %.1f.'%(model_ID, rsrc))
				
				temp = pool.map(process_conv, 
								[[AD_maps[i], rsrc] for i in range(len(list_ID))])
				AD_maps_dict[str(rsrc)] = np.asarray(temp)
				

				if plot_exp_conv:
					ID1 = list_ID[0]
					map1 = AD_maps_dict['AD_maps'][AD_maps_dict['ID']==ID1][0]
					map1_conv = AD_maps_dict[str(rsrc)][AD_maps_dict['ID']==ID1][0]

					plot_example_conv(map1, map1_conv, 
									  output_direc+'%i_%.1f_%s_AD_exp.png'%(list_ID[0], 
																		 rsrc, 
																		 model_ID))

				if gen_lc:
					if verbose:
						print('Generating %i lightcurves for the convolved AD maps with rsrc=%0.1f'%(num_lc, rsrc))

					
					AD_lc_temp = pool.map(process_lc, 
									[[AD_maps_dict[str(rsrc)][i], num_lc] for i in range(len(list_ID))]) 

					if r == 0:
						if verbose:
							print('Generating %i lightcurves for the unconvolved AD maps'%(num_lc))
						AD_lc_temp_true = pool.map(process_lc, 
											[[AD_maps_dict['AD_maps'][i], num_lc] for i in range(len(list_ID))])
						AD_lcs_dict['AD_lcs'] = np.asarray([AD_lc_temp_true[i][0] for i in range(len(list_ID))])
						AD_lcs_dict['AD_lcs_orig'] = np.asarray([AD_lc_temp_true[i][1] for i in range(len(list_ID))])

					AD_lcs_dict['conv_lcs_'+str(rsrc)] = np.asarray([AD_lc_temp_true[i][0] for i in range(len(list_ID))])
					AD_lcs_dict['conv_lcs_orig_'+str(rsrc)] = np.asarray([AD_lc_temp_true[i][0] for i in range(len(list_ID))])

					
					if plot_exp_lc:
						ID1 = list_ID[0]
						map1 = AD_maps_dict['AD_maps'][AD_maps_dict['ID']==ID1][0]
						map1_conv = AD_maps_dict[str(rsrc)][AD_maps_dict['ID']==ID1][0]
						lc_set1 = AD_lcs_dict['AD_lcs'][AD_lcs_dict['ID']==ID1][0][0]
						lc_set2 = AD_lcs_dict['conv_lcs_'+str(rsrc)][AD_lcs_dict['ID']==ID1][0][0]
						origin1 = AD_lcs_dict['AD_lcs_orig'][AD_lcs_dict['ID']==ID1][0][0]
						origin2 = AD_lcs_dict['conv_lcs_orig_'+str(rsrc)][AD_lcs_dict['ID']==ID1][0][0]
						lc_direc = output_direc+'AD_map_%i_%.1f_%s_exp_lc.png'%(list_ID[0], rsrc, model_ID)

						plot_example_lc(list_ID[0],
										[map1, map1_conv], 
										[lc_set1, lc_set2], 
										[origin1, origin2], 
										rsrc, 
										lc_direc)
		else:
			if gen_lc:
				if verbose:
					print('Generating %i lightcurves for the unconvolved AD maps'%(num_lc))
				AD_lc_temp_true = pool.map(process_lc, 
										[[AD_maps_dict['true_maps'][i], num_lc] for i in range(len(list_ID))])
				AD_lcs_dict['AD_lcs'] = np.asarray([AD_lc_temp_true[i][0] for i in range(len(list_ID))])
				AD_lcs_dict['AD_lcs_orig'] = np.asarray([AD_lc_temp_true[i][1] for i in range(len(list_ID))])

		if save_maps:
			pkl.dump(AD_maps_dict, 
			 open(output_direc+'AD_maps_%s.pkl'%(model_ID), 'wb'))
		if save_maps and save_lcs:
			pkl.dump(AD_lcs_dict, 
			open(output_direc+'AD_lcs_%s.pkl'%(model_ID), 'wb'))


	return (maps_dict,
			AD_maps_dict,
			lcs_dict,
			AD_lcs_dict)
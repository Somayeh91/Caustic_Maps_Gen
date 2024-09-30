import time
import numpy as np

config_ger_1000 = {}
config_ger_1000['dataset'] = 'ger1000'
config_ger_1000['verbose'] = True
config_ger_1000['dim'] = 10000
config_ger_1000['date'] = None
config_ger_1000['save_test_every_epoch'] = 20
config_ger_1000['print_every_n_epoch'] = 1
config_ger_1000['output_dir'] = './../results/'
config_ger_1000['metadata_directory'] = './../data/all_maps_meta_kgs.csv'
config_ger_1000['list_IDs_directory'] = './../data/ID_maps_selected_kappa_equal_gamma.dat'
config_ger_1000['shuffle'] = True
config_ger_1000['random_seed'] = 10
config_ger_1000['data_size'] = 3828
config_ger_1000['val_frac'] = 0.1
config_ger_1000['test_frac'] = 0.05
config_ger_1000['n_channel'] = 1
config_ger_1000['res_scale'] = 10
config_ger_1000['crop_scale'] = 1
config_ger_1000['output_format'] = 'xx'
config_ger_1000['include_lens_pos'] = False
config_ger_1000['include_map_units'] = False
config_ger_1000['recons_loss_abv'] = 'bce'
config_ger_1000['data_path'] = './../../../fred/oz108/GERLUMPH_project/DATABASES/gerlumph_db/'
config_ger_1000['data_path_bulk'] = './../../../fred/oz108/skhakpas/'

config_ger_1000['input_normalize_sym'] = True

config_ger_1000['optimizer'] = 'adam' # adam, sgd
config_ger_1000['adam_beta1'] = 0.5
config_ger_1000['lr'] = 1e-05 #0.001 for WAE-MMD and 0.0003 for WAE-GAN
config_ger_1000['lr_adv'] = 0.001
config_ger_1000['lr_schedule'] = 'manual_smooth' #manual, plateau, or a number
config_ger_1000['batch_size'] = 8
config_ger_1000['epoch_num'] = 50
config_ger_1000['n_workers'] = 4
config_ger_1000['batch_norm'] = False
config_ger_1000['batch_norm_eps'] = 1e-05
config_ger_1000['batch_norm_decay'] = 0.9
config_ger_1000['double_conv_blocks_num'] = 4
config_ger_1000['double_conv_n_channels'] = np.array([32, 64, 128, 128])
config_ger_1000['double_conv_kernel_dim'] = np.array([3, 3, 3, 3])
config_ger_1000['double_conv_maxpool_size'] = np.array([2, 2, 2, 5])
config_ger_1000['double_conv_upsample_size'] = np.array([5, 2, 2, 2])
config_ger_1000['double_conv_kernel_stride'] = np.array([1, 1, 1, 1])
config_ger_1000['double_conv_kernel_padding'] = np.array([1, 1, 1, 1])
config_ger_1000['2d_zshape'] = (25, 25)
config_ger_1000['1d_zshape'] = 10
config_ger_1000['sigma'] = 1
config_ger_1000['flatten_z'] = False

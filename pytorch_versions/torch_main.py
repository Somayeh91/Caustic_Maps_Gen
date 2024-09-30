import argparse
import torch
import torch_config
import torch_dataloader
import torch_models
import torch_operations
import os
import numpy as np

# from torchsummary import summary

torch.manual_seed(123)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Process input parameters.')
parser.add_argument('-exp', action='store', default='gerlumph',
                    help='What dataset is uesd.')
parser.add_argument('-dim', action='store', default=10000,
                    help='What is the dimension of the maps.')
parser.add_argument('-directory', action='store',
                    default='./../../../fred/oz108/GERLUMPH_project/DATABASES/gerlumph_db/',
                    help='Specify the directory where the maps are stored on the supercomputer.')
parser.add_argument('-batch_size', action='store', default=8, help='Batch size for the autoencoder.')
parser.add_argument('-n_epochs', action='store', default=50, help='Number of epochs.')
parser.add_argument('-n_channels', action='store', default=1, help='Number of channels.')
parser.add_argument('-lr_rate', action='store', default=0.00001, help='Learning rate.')
parser.add_argument('-lr_schedule', action='store', default='manual_smooth', help='Learning rate schedule.')
parser.add_argument('-res_scale', action='store', default=10,
                    help='With what scale should the original maps resoultion be reduced.')
parser.add_argument('-crop_scale', action='store', default=1,
                    help='With what scale should the original maps be cropped and multipled by 4.')
parser.add_argument('-date', action='store',
                    default='',
                    help='Specify the directory where the results should be stored.')
parser.add_argument('-list_IDs_directory', action='store',
                    default='./../data/ID_maps_selected_kappa_equal_gamma.dat',
                    help='Specify the directory where the list of map IDs is stored.')
parser.add_argument('-conversion_file_directory', action='store',
                    default='./../data/maps_selected_kappa_equal_gamma.csv',
                    help='Specify the directory where the conversion parameters.')
parser.add_argument('-test_set_size', action='store',
                    default=10,
                    help='Size of the test set.'),
parser.add_argument('-sample_size', action='store',
                    default=3828,
                    help='Size of the sample data set.'
                    )
parser.add_argument('-saved_model', action='store',
                    default=False,
                    help='Whether or not you want to train an already-saved model.'
                    )
parser.add_argument('-model_design', action='store',
                    default='2l',
                    help='Which model design you want.'
                    )
parser.add_argument('-saved_model_path', action='store',
                    default='./../results/23-03-09-01-39-47/model_10000_8_1e-05',
                    help='Whether or not you want to train an already-saved model.'
                    )
parser.add_argument('-loss_function', action='store',
                    default='binary_crossentropy',
                    help='What is the loss function?'
                    )
parser.add_argument('-lc_loss_function_metric', action='store',
                    default='mse',
                    help='What is the metric to calculate statistics in the lc loss function?'
                    )
parser.add_argument('-lc_loss_function_coeff', action='store',
                    default=1,
                    help='What is the metric to calculate statistics in the lc loss function?'
                    )
parser.add_argument('-optimizer', action='store',
                    default='adam',
                    help='What is the optimizer?'
                    )
parser.add_argument('-flow_label', action='store',
                    default=None,
                    help='What flow model you want to use. Use this with Unet_NF model design ONLY.'
                    )
parser.add_argument('-n_flows', action='store',
                    default=4,
                    help='Number of flows in the Unet_NF model design.'
                    )
parser.add_argument('-bottleneck_size', action='store',
                    default=625,
                    help='Size of the bottleneck when NF is added. ONLY with Unet_NF model design.'
                    )
parser.add_argument('-test_set_selection', action='store',
                    default='random',
                    help='Do you want to choose your test set randomly or are you looking for a chosen set of IDs')
parser.add_argument('-early_callback', action='store',
                    default='False',
                    help='Do you want to add early calback when fitting the model?')
parser.add_argument('-early_callback_type', action='store',
                    default=1,
                    help='What type of early callback you want?')
parser.add_argument('-add_params', action='store',
                    default=False,
                    help='Do you want to add the 3 map params as new channels?')
parser.add_argument('-add_lens_pos', action='store',
                    default=False,
                    help='Do you want to add the lens positions?')
parser.add_argument('-add_map_units', action='store',
                    default=False,
                    help='Do you want to add the map units in RE? That is more useful when lens positions are '
                         'included')
parser.add_argument('-model_input_format', action='store',
                    default='xx',
                    help='Does the model get x as data and y as targets? xy, other options are xx, x')
parser.add_argument('-ngpu', action='store',
                    default=1,
                    help='Number of GPUs you are selecting.')
parser.add_argument('-n_workers', action='store',
                    default=4,
                    help='Number of processes for loading the data.')
parser.add_argument('-training_plans', action='store',
                    default='simple',
                    help='If you want to change the optimizer during the training.')
FLAGS = parser.parse_args()

# def main(FLAGS):
if FLAGS.exp == 'gerlumph':
    opts = torch_config.config_ger_1000
else:
    assert False, 'Unknown experiment configuration'

if FLAGS.date is not None:
    opts['date'] = FLAGS.date
if FLAGS.bottleneck_size is not None:
    opts['1d_zshape'] = int(FLAGS.bottleneck_size)
if FLAGS.lr_rate is not None:
    opts['lr'] = float(FLAGS.lr_rate)
if FLAGS.batch_size is not None:
    opts['batch_size'] = int(FLAGS.batch_size)
if FLAGS.n_epochs is not None:
    opts['epoch_num'] = int(FLAGS.n_epochs)
if FLAGS.n_channels is not None:
    opts['n_channel'] = int(FLAGS.n_channels)
if FLAGS.n_workers is not None:
    opts['n_workers'] = int(FLAGS.n_workers)
if FLAGS.loss_function is not None:
    opts['recons_loss_abv'] = FLAGS.loss_function
if FLAGS.lr_schedule is not None:
    opts['lr_schedule'] = FLAGS.lr_schedule

output_dir = opts['output_dir'] + opts['date'] + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('Loading data...')
data_dir = opts['data_path_bulk']
data_dict = {}
true_params = np.array([], dtype=np.float32).reshape(0, 5)

for i in range(7):
    data_dict_tmp, true_params_tmp = torch_dataloader.datareader(i, opts)
    data_dict = {**data_dict, **data_dict_tmp}
    true_params = np.vstack([true_params, true_params_tmp])
train_loader, test_loader = torch_dataloader.Data_Loader(opts, data_dict, true_params, verbose=True)

print('Defining the model...')
autoencoder = torch_models.Autoncoder(opts)  # , torch_models.Decoder(opts)

print('Sending the model to gpu/s...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# autoencoder = nn.DataParallel(autoencoder, device_ids=[0, 1])
autoencoder.to(device)

# if torch.cuda.is_available():
#     encoder, decoder = encoder.to(device), decoder.to(device)

# dim = int((opts['dim'] / opts['res_scale']) * opts['crop_scale'])
# summary(autoencoder, (1, dim, dim))
# summary(decoder, (1, opts['1d_zshape']))

print('Training data...')
autoencoder, loss = torch_operations.train(autoencoder, train_loader, opts, device, verbose=True)

print('Saving the loss plot...')
torch_operations.fig_loss(loss, output_dir)

print('Testing the data...')
torch_operations.test(autoencoder, train_loader, opts, device)

# main(FLAGS_)

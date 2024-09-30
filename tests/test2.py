import argparse
import torch
import torch_config
import torch_dataloader
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch_models
import torch_operations
import torch.nn as nn
import os
import pickle as pkl
import multiprocessing
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from torch.autograd import Variable
import matplotlib.pyplot as plt

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
if FLAGS.loss_function is not None:
    opts['recons_loss_abv'] = FLAGS.loss_function

print('Sending the model to gpu/s...')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is None:
            # pass
            print(p.shape, p.requires_grad)
        else:
            print(p.grad.shape, p.grad.data)
            # param_norm = p.grad.data.norm(2)
            # total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


print('Loading data...')
# dataset = torch_dataloader.prepare_pytorch_Dataset(opts)
data_dir = opts['data_path_bulk']
data_dict = {}
true_params = np.array([], dtype=np.float32).reshape(0,5)

for i in range(7):
    data_dict_tmp, true_params_tmp = torch_dataloader.datareader(i, opts)
    data_dict = {**data_dict, **data_dict_tmp}
    true_params = np.vstack([true_params, true_params_tmp])


dataset = torch_dataloader.prepare_pytorch_Dataset2(data_dict, true_params)
n = len(true_params)
batch = opts['batch_size']
frac = opts['test_frac']
train_set_size = ((int((1 - frac) * n)) - ((int((1 - frac) * n)) % batch))

print('Total sample size: ', n)
print('Training sample size: ', train_set_size)
print('Testing sample size: ', n - train_set_size)

# trainset = torch.stack([torch.Tensor(i) for i in maps[:train_set_size]])
# testset = torch.stack([torch.Tensor(i) for i in maps[train_set_size:]])

train_loader = DataLoader(dataset=dataset[:train_set_size][0],
                          batch_size=opts['batch_size'],
                          shuffle=True, num_workers=1)

test_loader = DataLoader(dataset=dataset[train_set_size:][0],
                         batch_size=opts['batch_size'],
                         shuffle=False, num_workers=1)

# encoder, decoder = Encoder(config_ger_1000), Decoder(config_ger_1000)
autoencoder = torch_models.Autoencoder(opts)
criterion = nn.MSELoss()
# nn.BCELoss() #
if torch.cuda.is_available():
    autoencoder = autoencoder.to(device)
    # encoder, decoder = encoder.to(device), decoder.to(device)

# summary(encoder, (1, 1000, 1000))
# summary(decoder, (1, 25, 25))
# summary(autoencoder, (1, 1000, 1000))

# def train(encoder, decoder, opts):
# encoder.train()
# decoder.train()
autoencoder.train()

output_dir = opts['output_dir'] + opts['date'] + '/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

lr = opts['lr']
epochs = opts['epoch_num']  # opts['epoch_num']
n_z = opts['1d_zshape']
sigma = opts['sigma']

# one = torch.Tensor([1])
# mone = one * -1

# if torch.cuda.is_available():
#     one = one.cuda()
#     mone = mone.cuda()

# Optimizers
# enc_optim = optim.Adam(encoder.parameters(), lr=0.01)
# dec_optim = optim.Adam(decoder.parameters(), lr=0.0001)

auto_optim = optim.Adam(autoencoder.parameters(), lr=lr)

# enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
# dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)
auto_scheduler = StepLR(auto_optim, step_size=30, gamma=0.2)
loss = []
# print(torch.cuda.memory_reserved(0))
for epoch in range(epochs):

    step = 0
    for images in train_loader:
        # print (images)
        images = images.to(torch.float32)

        if torch.cuda.is_available():
            images = images.cuda()
        torch.cuda.memory_reserved(0)

        # enc_optim.zero_grad()
        # dec_optim.zero_grad()
        auto_optim.zero_grad()

        torch.cuda.memory_reserved(0)
        # ======== Train Generator ======== #

        batch_size = images.size()[0]

        # z = encoder(images)
        # print(torch.cuda.memory_reserved(0))
        # x_recon = decoder(z)

        x_recon = autoencoder(images)

        recon_loss = criterion(x_recon, images)
        # recon_loss = F.binary_cross_entropy(x_recon, images)

        # ======== MMD Kernel Loss ======== #

        # z_fake = Variable(torch.randn(images.size()[0], n_z) * sigma)
        # if torch.cuda.is_available():
        #     z_fake = z_fake.cuda()

        # z_real = encoder(images)

        # mmd_loss = imq_kernel(z, z_fake, h_dim=encoder.n_z_1d)
        # mmd_loss = mmd_loss / batch_size

        # if step<10:
        #   coeff = 0
        # elif step<20 and step>10:
        #   coeff = 0.1
        # else:
        #   coeff = 1
        total_loss = recon_loss  # + coeff *mmd_loss
        total_loss.retain_grad()

        total_loss.backward()

        # decoder_grad_norm = get_grad_norm(decoder)
        # encoder_grad_norm = get_grad_norm(encoder)
        # decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(),
        #                                                    10000, norm_type=2)
        # encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(),
        #                                                    10000, norm_type=2)
        # autoencoder_grad_norm = torch.nn.utils.clip_grad_norm_(autoencoder.parameters(),
        #                                                        10000, norm_type=2)

        # enc_optim.step()
        # dec_optim.step()

        auto_optim.step()

        # print('Encoder grad norm: ', encoder_grad_norm)
        # print('Decoder grad norm: ', decoder_grad_norm)
        # print('Autoencoder grad norm: ', autoencoder_grad_norm)

        # enc_scheduler.step()
        # dec_scheduler.step()
        auto_scheduler.step()
        step += 1

    if (epoch + 1) % 1 == 0:
        # print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f, MMD Loss %.4f" %
        #       (epoch + 1, epochs, step + 1, len(train_loader), recon_loss.data.item(),
        #         mmd_loss.item()))
        loss.append(recon_loss.data.item())
        print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f" %
              (epoch + 1, epochs, step + 1, len(train_loader),
               recon_loss.data.item()))
        autoencoder_grad_norm = torch.nn.utils.clip_grad_norm_(autoencoder.parameters(),
                                                               10000, norm_type=2)
        print('Autoencoder grad norm: ', autoencoder_grad_norm)
    # torch.save(encoder.state_dict(), opts['output_dir']+'/encoder_1.pt')
    # torch.save(decoder.state_dict(), opts['output_dir']+'/decoder_1.pt')
    if (epoch + 1) % 10 == 0:
        torch.save(autoencoder.state_dict(), opts['output_dir'] + '/autoencoder_1.pt')

# return encoder, decoder, recon_loss, total_loss

autoencoder.eval()
batch_size = 8
test_iter = iter(test_loader)
test_data = next(test_iter)
test_data = test_data.to(torch.float32)
if torch.cuda.is_available():
    test_data = test_data.cuda()
# z_real = encoder(Variable(test_data))
# reconst = decoder(z_real).cpu()


reconst = autoencoder(Variable(test_data)).cpu()


for i in range(8):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(test_data[i].view(1000, 1000).cpu().detach().numpy())
    axs[1].imshow(reconst[i].view(1000, 1000).detach().numpy())
    # print(z_real[i])
    fig.savefig(opts['output_dir'] + '/result_' + str(i) + '.png')

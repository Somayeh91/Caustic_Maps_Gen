from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import pickle as pkl
import pandas as pd
from os.path import join
import numpy as np
from torch.utils.data import Dataset, DataLoader, \
    BatchSampler, SequentialSampler, \
    RandomSampler
from PIL import Image
import time


def NormalizeData(data, data_max=6,
                  data_min=-3):
    data_new = (data - data_min) / (data_max - data_min)
    return data_new


import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def double_conv2d(in_channels_, out_channels_, kernel_size=3, stride=1, padding=1,
                  pool_size=2, pool_stride=2,
                  dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels_, out_channels_, kernel_size=kernel_size,
                  stride=stride, padding=padding,
                  bias=bias),
        nn.ReLU(),
        nn.Conv2d(out_channels_, out_channels_, kernel_size=kernel_size,
                  stride=stride, padding=padding,
                  bias=bias),
        # nn.BatchNorm2d(out_channels_),
        nn.ReLU(),
        nn.MaxPool2d(pool_size, pool_stride))


def double_conv2dtranspos(in_channels_, out_channels_, kernel_size=3, stride=1,
                          scale_factor=2, mode='nearest',
                          padding=1, output_padding=0, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels_, out_channels_, kernel_size=kernel_size,
                           stride=stride, padding=padding,
                           output_padding=output_padding, groups=groups,
                           bias=bias, dilation=dilation),
        nn.ReLU(),
        nn.ConvTranspose2d(out_channels_, out_channels_, kernel_size=kernel_size,
                           stride=stride, padding=padding,
                           output_padding=output_padding, groups=groups,
                           bias=bias, dilation=dilation),
        # nn.BatchNorm2d(out_channels_),
        nn.ReLU(),
        nn.Upsample(scale_factor=scale_factor, mode=mode))


def single_conv2d(in_channels_, out_channels_, kernel_size_=1, stride_=1, padding_=1,
                  dilation_=1, groups=1, bias=True):
    return nn.Conv2d(in_channels_, out_channels_, kernel_size_, stride_,
                     padding_, dilation_, groups, bias)


def single_conv2dtranspos(in_channels_, out_channels_, kernel_size_=1, stride_=1,
                          padding_=0, output_padding_=0, dilation_=1, groups=1, bias=True):
    return nn.ConvTranspose2d(in_channels_, out_channels_, kernel_size_, stride_,
                              padding_, output_padding_, groups, bias, dilation_)


def _linear(in_features_, out_features_, bias=True):
    return nn.Linear(in_features_, out_features_, bias)


def _dropout(p=0.5, inplace=False):
    return nn.Dropout(p, inplace)


config_ger_1000 = {}
config_ger_1000['dataset'] = 'ger1000'
config_ger_1000['dim'] = 10000
config_ger_1000['verbose'] = True
config_ger_1000['date'] = time.strftime("%y-%m-%d-%H-%M-%S")
config_ger_1000['save_test_every_epoch'] = 20
config_ger_1000['print_every_n_epoch'] = 1
config_ger_1000['output_dir'] = '/fred/oz108/skhakpas/logs/' + \
                                config_ger_1000['date']
config_ger_1000['list_IDs_directory'] = '/home/skhakpas/data/all_maps_meta_kgs.csv'
config_ger_1000['shuffle'] = True
config_ger_1000['random_seed'] = 10
config_ger_1000['val_frac'] = 0.1
config_ger_1000['test_frac'] = 0.1
config_ger_1000['n_channel'] = 1
config_ger_1000['res_scale'] = 10
config_ger_1000['crop_scale'] = 1
config_ger_1000['include_lens_pos'] = False
config_ger_1000['include_map_units'] = False
config_ger_1000['data_path'] = '/home/skhakpas/data/single_map.pkl'
# config_ger_1000['data_path'] = '/home/skhakpas/data/all_maps_lens_pos.pkl'

config_ger_1000['input_normalize_sym'] = True

config_ger_1000['optimizer'] = 'adam'  # adam, sgd
config_ger_1000['adam_beta1'] = 0.5
config_ger_1000['lr'] = 1e-5  # 0.001 for WAE-MMD and 0.0003 for WAE-GAN
config_ger_1000['lr_adv'] = 0.001
config_ger_1000['lr_schedule'] = 'manual_smooth'  # manual, plateau, or a number
config_ger_1000['batch_size'] = 8

config_ger_1000['epoch_num'] = 2

config_ger_1000['batch_norm'] = True
config_ger_1000['batch_norm_eps'] = 1e-05
config_ger_1000['batch_norm_decay'] = 0.9
config_ger_1000['double_conv_blocks_num'] = 3
config_ger_1000['double_conv_n_channels'] = np.array([32, 64, 128])
config_ger_1000['double_conv_kernel_dim'] = np.array([3, 3, 3])
config_ger_1000['double_conv_maxpool_size'] = np.array([5, 2, 2])
config_ger_1000['double_conv_upsample_size'] = np.array([2, 2, 5])
config_ger_1000['double_conv_kernel_stride'] = np.array([1, 1, 1])
config_ger_1000['double_conv_kernel_padding'] = np.array([1, 1, 1])
config_ger_1000['2d_zshape'] = (50, 50)
config_ger_1000['1d_zshape'] = 500
config_ger_1000['sigma'] = 1
config_ger_1000['flatten_z'] = False
config_ger_1000['n_channels'] = 1
config_ger_1000['data_path_bulk'] = './../../../fred/oz108/skhakpas/'


# os.listdir("/content/drive/MyDrive/caustic_maps/reduced_maps/kappa_equal_gamma/maps_batch012345.pkl")
def datafinder(index, opts):
    path = opts['data_path_bulk'] + 'all_maps_batch' + str(index) + '.pkl'
    data_dict_batch = pkl.load(open(path, "rb"))

    all_params = pd.read_csv(opts['list_IDs_directory'])

    true_params_batch = []
    # maps = np.zeros((len(data_dict.keys()), 1000, 1000))
    for i, key in enumerate(data_dict_batch.keys()):
        true_params_batch.append([int(key), all_params.k[all_params.ID == key].values[0],
                                  all_params.g[all_params.ID == key].values[0],
                                  all_params.s[all_params.ID == key].values[0],
                                  all_params.const[all_params.ID == key].values[0]])
        # data = data_dict[key].reshape((1000,1000))
        # data = np.log10(data * all_params.const[all_params.ID == key].values[0] + 0.001)
        # maps[i, :, :] = NormalizeData(data, 6, -3)

    return data_dict_batch, np.asarray(true_params_batch)


def datareader(opts):
    # path = opts['data_path_bulk']
    data_dict_all = {}
    true_params_all = np.array([], dtype=np.float32).reshape(0, 5)
    for j in range(2):
        data_dict_tmp, true_params_tmp = datafinder(j, opts)
        data_dict_all = {**data_dict_all, **data_dict_tmp}
        true_params_all = np.vstack([true_params_all, true_params_tmp])

    # data_dict = pkl.load(open(path, "rb"))
    # print(len(data_dict.keys()))
    #
    # all_params = pd.read_csv(opts[ 'list_IDs_directory'])
    #
    #
    # true_params = []
    # # maps = np.zeros((len(data_dict.keys()), 1000, 1000))
    # for i, key in enumerate(data_dict.keys()):
    #   true_params.append([int(key), all_params.k[all_params.ID == key].values[0],
    #                     all_params.g[all_params.ID == key].values[0],
    #                     all_params.s[all_params.ID == key].values[0],
    #                     all_params.const[all_params.ID == key].values[0]])
    # data = data_dict[key].reshape((1000,1000))
    # data = np.log10(data * all_params.const[all_params.ID == key].values[0] + 0.001)
    # maps[i, :, :] = NormalizeData(data, 6, -3)

    return data_dict_all, np.asarray(true_params_all)


# (maps.reshape((len(maps), 1, 1000, 1000))).astype(np.float32)

from pandas.core.construction import is_datetime64_ns_dtype
import os
import pandas as pd


class prepare_pytorch_Dataset(Dataset):
    def __init__(self, data_dict, ids, transform=None, target_transform=None):
        self.ids = ids
        self.data_dict = data_dict
        self.len_data = len(ids)
        self.batch_size = 8
        self.input_keys = list(data_dict.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return (self.len_data % self.batch_size)

    def __getitem__(self, idx):
        keys = self.input_keys[idx]

        return self.__generatedata(keys)

    def __generatedata(self, keys):
        # print(len(keys))
        X = np.zeros((len(keys), 1, 1000, 1000))
        Y = np.zeros((len(keys), 1, 1000, 1000))
        for i, key in enumerate(keys):
            data = np.asarray(self.data_dict[key]).reshape((1000, 1000))
            data = np.log10(data * self.ids[i, 4] + 0.001)
            X[i, 0, :, :] = NormalizeData(data, 6, -3).astype(np.float32)
            Y[i, 0, :, :] = X[i, 0, :, :].astype(np.float32)
            if self.transform:
                X[i, 0, :, :] = self.transform(X[i, 0, :, :])
            if self.target_transform:
                Y[i, 0, :, :] = self.target_transform(Y[i, 0, :, :])
        return X, Y


def dataloader(opts, data_dict, ids):  # transform = T.Resize(28)

    # path = opts['data_path']
    # data_dict = pkl.load(open(path, "rb"))

    dataset = prepare_pytorch_Dataset(data_dict, ids)
    n = len(ids)
    batch = opts['batch_size']
    frac = opts['test_frac']
    train_set_size = ((int((1 - frac) * n)) - ((int((1 - frac) * n)) % batch))

    # trainset = torch.stack([torch.Tensor(i) for i in maps[:train_set_size]])
    # testset = torch.stack([torch.Tensor(i) for i in maps[train_set_size:]])

    train_loader = DataLoader(dataset=dataset[:train_set_size],
                              batch_size=opts['batch_size'],
                              shuffle=True)

    test_loader = DataLoader(dataset=dataset[:train_set_size],
                             batch_size=1,
                             shuffle=False)
    return train_loader, test_loader


data_dict, true_params = datareader(config_ger_1000)

train_loader, test_loader = dataloader(config_ger_1000, data_dict, true_params)

print("len(train_loader.dataset)", len(train_loader.dataset))

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable
# from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR

torch.manual_seed(123)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device", device)


def free_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = True


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


class Encoder(nn.Module):
    def __init__(self, opts):
        super(Encoder, self).__init__()

        self.n_channel = opts['n_channel']
        self.n_z_2d = opts['2d_zshape']
        self.n_z_1d = opts['1d_zshape']
        self.flatten_z = opts['flatten_z']
        self.num_blocks = opts['double_conv_blocks_num']
        self.filters_blocks = opts['double_conv_n_channels']
        self.kernel_blocks = opts['double_conv_kernel_dim']
        self.stride_blocks = opts['double_conv_kernel_stride']
        self.padding_blocks = opts['double_conv_kernel_padding']
        self.maxpool_sizes = opts['double_conv_maxpool_size']

        self.main = []
        for j in range(self.num_blocks):
            if j == 0:
                in_channels = 1
            else:
                in_channels = self.filters_blocks[j - 1]
            self.main.append(double_conv2d(in_channels, self.filters_blocks[j],
                                           kernel_size=self.kernel_blocks[j],
                                           stride=self.stride_blocks[j],
                                           padding=self.padding_blocks[j],
                                           pool_size=self.maxpool_sizes[j],
                                           pool_stride=self.maxpool_sizes[j]))
        self.main = nn.Sequential(*self.main,
                                  single_conv2d(self.filters_blocks[-1],
                                                1, kernel_size_=3,
                                                stride_=1, padding_=1),
                                  nn.ReLU())

        self.linear = nn.Linear(self.n_z_2d[0] * self.n_z_2d[1],
                                self.n_z_1d)

    def forward(self, x):
        x = self.main(x)
        if self.flatten_z:
            x = x.view(-1, self.n_z_2d[0] * self.n_z_2d[1] * 1)
            x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self, opts):
        super(Decoder, self).__init__()

        self.n_channel = opts['n_channel']
        self.n_z_2d = opts['2d_zshape']
        self.n_z_1d = opts['1d_zshape']
        self.flatten_z = opts['flatten_z']
        self.num_blocks = opts['double_conv_blocks_num']
        self.filters_blocks = opts['double_conv_n_channels']
        self.kernel_blocks = opts['double_conv_kernel_dim']
        self.stride_blocks = opts['double_conv_kernel_stride']
        self.padding_blocks = opts['double_conv_kernel_padding']
        self.upsampling_sizes = opts['double_conv_upsample_size']

        self.linear = nn.Sequential(
            nn.Linear(self.n_z_1d, self.n_z_2d[0] * self.n_z_2d[1]),
            nn.ReLU()
        )
        self.main = []
        for j in range(self.num_blocks):
            if j == 0:
                in_channels = 1
            else:
                in_channels = self.filters_blocks[-1 - (j - 1)]
            self.main.append(double_conv2dtranspos(in_channels, self.filters_blocks[-1 - j],
                                                   kernel_size=self.kernel_blocks[-1 - j],
                                                   stride=self.stride_blocks[-1 - j],
                                                   padding=self.padding_blocks[-1 - j],
                                                   scale_factor=self.upsampling_sizes[j]))
        self.main = nn.Sequential(*self.main,
                                  single_conv2dtranspos(self.filters_blocks[0],
                                                        1, 3, stride_=1, padding_=1),
                                  nn.Sigmoid())

    def forward(self, x):
        if self.flatten_z:
            x = self.linear(x)
            x = x.view(-1, 1, self.n_z_2d[0], self.n_z_2d[1])
        x = self.main(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, opts):
        super(Autoencoder, self).__init__()

        self.n_channel = opts['n_channel']
        self.n_z_2d = opts['2d_zshape']
        self.n_z_1d = opts['1d_zshape']
        self.flatten_z = opts['flatten_z']
        self.num_blocks = opts['double_conv_blocks_num']
        self.filters_blocks = opts['double_conv_n_channels']
        self.kernel_blocks = opts['double_conv_kernel_dim']
        self.stride_blocks = opts['double_conv_kernel_stride']
        self.padding_blocks = opts['double_conv_kernel_padding']
        self.maxpool_sizes = opts['double_conv_maxpool_size']
        self.upsampling_sizes = opts['double_conv_upsample_size']

        self.main = []
        for j in range(self.num_blocks):
            if j == 0:
                in_channels = 1
            else:
                in_channels = self.filters_blocks[j - 1]
            self.main.append(double_conv2d(in_channels, self.filters_blocks[j],
                                           kernel_size=self.kernel_blocks[j],
                                           stride=self.stride_blocks[j],
                                           padding=self.padding_blocks[j],
                                           pool_size=self.maxpool_sizes[j],
                                           pool_stride=self.maxpool_sizes[j]))

        self.main.append(single_conv2d(self.filters_blocks[-1],
                                       1, kernel_size_=3,
                                       stride_=1, padding_=1))
        self.main.append(nn.ReLU())

        n_ = self.num_blocks - 1
        for j in range(self.num_blocks):
            if j == 0:
                in_channels = 1
            else:
                in_channels = self.filters_blocks[n_ - (j - 1)]
            self.main.append(double_conv2dtranspos(in_channels, self.filters_blocks[n_ - j],
                                                   kernel_size=self.kernel_blocks[n_ - j],
                                                   stride=self.stride_blocks[n_ - j],
                                                   padding=self.padding_blocks[n_ - j],
                                                   scale_factor=self.upsampling_sizes[j]))

        self.main = nn.Sequential(*self.main,
                                  single_conv2dtranspos(self.filters_blocks[0],
                                                        1, 1, stride_=1, padding_=0),
                                  nn.Sigmoid())

        # self.main = nn.Sequential(*self.main,
        #                           nn.UpsamplingNearest2d(self.filters_blocks[0],
        #                                    1, 3, stride_=1, padding_=1),
        #                           nn.Sigmoid())

    def forward(self, x):

        x = self.main(x)
        return x


def imq_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / (batch_size)
        stats += res1 - res2

    return stats


def rbf_kernel(X: torch.Tensor,
               Y: torch.Tensor,
               h_dim: int):
    batch_size = X.size(0)

    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 / scale
        res1 = torch.exp(-C * dists_x)
        res1 += torch.exp(-C * dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = torch.exp(-C * dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats += res1 - res2

    return stats


def gaussian_likelihood(self, x_hat, logscale, x):
    scale = torch.exp(logscale)
    mean = x_hat
    dist = torch.distributions.Normal(mean, scale)

    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum(dim=(1, 2, 3))


def kl_divergence(self, z, mu, std):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    kl = kl.sum(-1)
    return kl


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


# encoder, decoder = Encoder(config_ger_1000), Decoder(config_ger_1000)
autoencoder = Autoencoder(config_ger_1000)
criterion = nn.MSELoss()
# nn.BCELoss() #
if torch.cuda.is_available():
    autoencoder = autoencoder.to(device)
    # encoder, decoder = encoder.to(device), decoder.to(device)

# summary(encoder, (1, 1000, 1000))
# summary(decoder, (1, 25, 25))
print(summary(autoencoder, (1, 1000, 1000)))

for images in tqdm(train_loader):
    im = images
    break

# def train(encoder, decoder, opts):
opts = config_ger_1000
# encoder.train()
# decoder.train()
autoencoder.train()

output_dir = opts['output_dir']
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

lr = opts['lr']
epochs = 2000  # opts['epoch_num']
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

auto_optim = optim.Adam(autoencoder.parameters(), lr=0.00001)

# enc_scheduler = StepLR(enc_optim, step_size=30, gamma=0.5)
# dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)
auto_scheduler = StepLR(auto_optim, step_size=5, gamma=0.5)
loss = []
# print(torch.cuda.memory_reserved(0))
for epoch in range(epochs):

    step = 0
    for images in tqdm(train_loader):
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

        # recon_loss = 1./torch.abs(torch.log(F.mse_loss(x_recon, images)))
        recon_loss = F.binary_cross_entropy(x_recon, images)
        # recon_loss = 0.5*F.mse_loss(x_recon, images)
        # recon_loss = F.huber_loss(x_recon, images, delta=0.5)

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
        # total_loss.retain_grad()

        total_loss.backward()

        # decoder_grad_norm = get_grad_norm(decoder)
        # encoder_grad_norm = get_grad_norm(encoder)
        # decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(),
        #                                                    10000, norm_type=2)
        # encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(),
        #                                                    10000, norm_type=2)

        # enc_optim.step()
        # dec_optim.step()

        auto_optim.step()

        # print('Encoder grad norm: ', encoder_grad_norm)
        # print('Decoder grad norm: ', decoder_grad_norm)
        # print('Autoencoder grad norm: ', autoencoder_grad_norm)
        #
        # enc_scheduler.step()
        # dec_scheduler.step()
        # auto_scheduler.step()
        step += 1

    if (epoch + 1) % 1 == 0:
        # print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f, MMD Loss %.4f" %
        #       (epoch + 1, epochs, step + 1, len(train_loader), recon_loss.data.item(),
        #         mmd_loss.item()))
        loss.append(recon_loss.data.item())
        print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f" %
              (epoch + 1, epochs, step + 1, len(train_loader),
               recon_loss.data.item()))
        # autoencoder_grad_norm = torch.nn.utils.clip_grad_norm_(autoencoder.parameters(),
        #                                                    10000, norm_type=2)
        # print('Autoencoder grad norm: ', autoencoder_grad_norm)
    # torch.save(encoder.state_dict(), opts['output_dir']+'/encoder_1.pt')
    # torch.save(decoder.state_dict(), opts['output_dir']+'/decoder_1.pt')
    if (epoch + 1) % 10 == 0:
        torch.save(autoencoder.state_dict(), opts['output_dir'] + '/autoencoder_1.pt')

# return encoder, decoder, recon_loss, total_loss

# import matplotlib.pyplot as plt
# plt.plot(loss)

# encoder, decoder, recon_loss, mmd_loss, total_loss = train(encoder,
#                                                            decoder,
#                                                            config_ger_1000)

# encoder.eval()
# decoder.eval()

autoencoder.eval()
batch_size = 1
test_iter = iter(test_loader)
test_data = next(test_iter)
# test_data2 = next(test_iter)
test_data = test_data.to(torch.float32)
# test_data2 = test_data2.to(torch.float32)

if torch.cuda.is_available():
    test_data = test_data.cuda()
    # test_data2 = test_data2.cuda()
# z_real = encoder(Variable(test_data))
# reconst = decoder(z_real).cpu()


reconst = autoencoder(Variable(test_data)).cpu()

print(reconst)

im

import matplotlib.pyplot as plt

# from matplotlib.colors import LogNorm

for i in range(1):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(test_data[i].view(1000, 1000).cpu().detach().numpy())
    axs[1].imshow(reconst[i].view(1000, 1000).detach().numpy())
    # print(z_real[i])
    fig.savefig(config_ger_1000['output_dir'] + '/result_' + str(i) + '.png')

# -(torch.mul(im.squeeze(),  torch.log(im.squeeze())) + torch.mul((1 - im.squeeze()) , torch.log(1 - im.squeeze()))).mean()
# ((reconst - im)**2).mean()

import keras

keras.losses.BinaryCrossentropy(from_logits=True)(test_data.cpu(), test_data2.cpu())

print(F.binary_cross_entropy_with_logits(torch.zeros(8, 1, 1000, 1000).cuda(), test_data2))
print(F.mse_loss(torch.zeros(8, 1, 1000, 1000).cuda(), test_data2))
print(1. / torch.abs(torch.log(F.mse_loss(torch.zeros(8, 1, 1000, 1000).cuda(), test_data2))))
print(keras.losses.BinaryCrossentropy(from_logits=True)(test_data2.cpu(), test_data2.cpu()))

print(F.binary_cross_entropy(test_data, test_data))
print(F.mse_loss(test_data, test_data))
print(1. / torch.abs(torch.log(F.mse_loss(test_data, test_data))))
print(keras.losses.BinaryCrossentropy(from_logits=True)(test_data.cpu(), test_data.cpu()))

print(F.binary_cross_entropy(test_data, test_data2))
print(1. / torch.abs(torch.log(F.mse_loss(test_data, test_data2))))
print(keras.losses.BinaryCrossentropy(from_logits=True)(test_data.cpu(), test_data2.cpu()))

plt.hist(test_data[i].cpu().flatten())
plt.hist(reconst[i].detach().numpy().flatten())

next(iter(test_loader)).shape

49152 / (128 * 128)

552 / 8
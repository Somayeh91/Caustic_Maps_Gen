import argparse
import os
import torch
import torch.nn as nn
import torch_operations


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
            self.main.append(torch_operations.double_conv2d(in_channels, self.filters_blocks[j],
                                                            kernel_size=self.kernel_blocks[j],
                                                            stride=self.stride_blocks[j],
                                                            padding=self.padding_blocks[j],
                                                            pool_size=self.maxpool_sizes[j],
                                                            pool_stride=self.maxpool_sizes[j]))
        self.main = nn.Sequential(*self.main,
                                  torch_operations.single_conv2d(self.filters_blocks[-1],
                                                                 1, kernel_size_=3,
                                                                 stride_=1, padding_=1))
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
            self.main.append(torch_operations.double_conv2dtranspos(in_channels, self.filters_blocks[-1 - j],
                                                                    kernel_size=self.kernel_blocks[-1 - j],
                                                                    stride=self.stride_blocks[-1 - j],
                                                                    padding=self.padding_blocks[-1 - j],
                                                                    scale_factor=self.upsampling_sizes[j]))
        self.main = nn.Sequential(*self.main,
                                  torch_operations.single_conv2dtranspos(self.filters_blocks[0],
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
            self.main.append(torch_operations.double_conv2d(in_channels, self.filters_blocks[j],
                                                            kernel_size=self.kernel_blocks[j],
                                                            stride=self.stride_blocks[j],
                                                            padding=self.padding_blocks[j],
                                                            pool_size=self.maxpool_sizes[j],
                                                            pool_stride=self.maxpool_sizes[j]))

        self.main.append(torch_operations.single_conv2d(self.filters_blocks[-1],
                                                        1, kernel_size_=3,
                                                        stride_=1, padding_=1))
        self.main.append(nn.ReLU())

        for j in range(self.num_blocks):
            if j == 0:
                in_channels = 1
            else:
                in_channels = self.filters_blocks[-1 - (j - 1)]
            self.main.append(torch_operations.double_conv2dtranspos(in_channels, self.filters_blocks[-1 - j],
                                                                    kernel_size=self.kernel_blocks[-1 - j],
                                                                    stride=self.stride_blocks[-1 - j],
                                                                    padding=self.padding_blocks[-1 - j],
                                                                    scale_factor=self.upsampling_sizes[j]))

        self.main = nn.Sequential(*self.main,
                                  torch_operations.single_conv2dtranspos(self.filters_blocks[0],
                                                                         1, 3, stride_=1, padding_=1),
                                  nn.Sigmoid())

    def forward(self, x):

        x = self.main(x)
        return x

import torch
import torch.nn as nn
import os
import torch.optim as optim
import torch_losses
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.autograd import Variable
import torch_losses
from torch.optim.lr_scheduler import StepLR


def double_conv2d(in_channels_, out_channels_, kernel_size=3, stride=1, padding=1,
                  pool_size=2, pool_stride=2,
                  dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels_, out_channels_, kernel_size=kernel_size,
                  stride=stride, padding=padding,
                  bias=bias),
        nn.ReLU(True),
        nn.Conv2d(out_channels_, out_channels_, kernel_size=kernel_size,
                  stride=stride, padding=padding,
                  bias=bias),
        # nn.BatchNorm2d(out_channels_),
        nn.ReLU(True),
        nn.MaxPool2d(pool_size, pool_stride))


def double_conv2dtranspos(in_channels_, out_channels_, kernel_size=3, stride=1,
                          scale_factor=2, mode='nearest',
                          padding=1, output_padding=0, dilation=1, groups=1, bias=True):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels_, out_channels_, kernel_size=kernel_size,
                           stride=stride, padding=padding,
                           output_padding=output_padding, groups=groups,
                           bias=bias, dilation=dilation),
        nn.ReLU(True),
        nn.ConvTranspose2d(out_channels_, out_channels_, kernel_size=kernel_size,
                           stride=stride, padding=padding,
                           output_padding=output_padding, groups=groups,
                           bias=bias, dilation=dilation),
        # nn.BatchNorm2d(out_channels_),
        nn.ReLU(True),
        nn.Upsample(scale_factor=scale_factor, mode=mode))


def single_conv2d(in_channels_, out_channels_, kernel_size_=1, stride_=1, padding_=0,
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


def train(autoencoder, train_loader, opts, device, verbose=False):
    autoencoder.train()
    # decoder.train()

    output_dir = opts['output_dir'] + opts['date'] + '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    lr = opts['lr']
    epochs = opts['epoch_num']
    # n_z = opts['1d_zshape']
    # sigma = opts['sigma']

    # Optimizers
    auto_optim = optim.Adam(autoencoder.parameters(), lr=lr)
    # dec_optim = optim.Adam(decoder.parameters(), lr=lr)

    # enc_scheduler = StepLR(autoenc_optim, step_size=30, gamma=0.5)
    # dec_scheduler = StepLR(dec_optim, step_size=30, gamma=0.5)
    if opts['lr_schedule'] == 'manual_smooth':
        auto_scheduler = StepLR(auto_optim, step_size=30, gamma=0.2)

    # print(torch.cuda.memory_reserved(0))
    loss = []
    for epoch in range(epochs):
        step = 0
        for images in tqdm(train_loader):
            # print (images)
            images = images.to(device)
            # if torch.cuda.is_available():
            #     images = images.cuda()
            torch.cuda.memory_reserved(0)

            auto_optim.zero_grad()
            # dec_optim.zero_grad()

            torch.cuda.memory_reserved(0)
            # ======== Train Generator ======== #

            # batch_size = images.size()[0]

            # z = encoder(images)
            # print(torch.cuda.memory_reserved(0))
            x_recon = autoencoder(images)
            criterion = torch_losses.recon_loss(opts['recons_loss_abv'])
            recon_loss = criterion(x_recon, images)

            # ======== MMD Kernel Loss ======== #

            # z_fake = Variable(torch.randn(images.size()[0], n_z) * sigma)
            # if torch.cuda.is_available():
            #     z_fake = z_fake.cuda()
            #
            # # z_real = encoder(images)
            #
            # mmd_loss = torch_losses.imq_kernel(z, z_fake, h_dim=encoder.n_z_1d)
            # mmd_loss = mmd_loss / batch_size

            # if step < 10:
            #     coeff = 0
            # elif 20 > step > 10:
            #     coeff = 0.1
            # else:
            #     coeff = 1
            total_loss = recon_loss  # + coeff * mmd_loss
            total_loss.backward()

            # enc_optim.step()
            # dec_optim.step()
            auto_optim.step()

            if opts['lr_schedule'] == 'manual_smooth':
                auto_scheduler.step()

            step += 1

        if (epoch + 1) % 1 == 0:
            print("Epoch: [%d/%d], Step: [%d/%d], Reconstruction Loss: %.4f" %
                  (epoch + 1, epochs, step + 1, len(train_loader), recon_loss.data.item()))
            loss.append(recon_loss)
            if verbose:
                autoencoder_grad_norm = torch.nn.utils.clip_grad_norm_(autoencoder.parameters(),
                                                                       10000, norm_type=2)
                print('Autoencoder grad norm: ', autoencoder_grad_norm)
        if (epoch + 1) % 10 == 0:
            torch.save(autoencoder.state_dict(), output_dir + 'encoder_1.pt')
    # torch.save(decoder.state_dict(), opts['output_dir'] + '/decoder_1.pt')
    return autoencoder, loss


# def save_single_prediction(encoder, decoder, map_test, opts):
#     encoder.eval()
#     decoder.eval()
#     map_test = torch.tensor(map_test)
#     if torch.cuda.is_available():
#         map_test = map_test.cuda()
#     z_real = encoder(Variable(map_test))
#     reconst = decoder(z_real).cpu()


def save_plot(x, y, opts, filename):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    dim = int((opts['dim'] / opts['res_scale']) * opts['crop_scale'])
    axs[0].imshow(x.view(dim, dim).cpu().detach().numpy())
    axs[1].imshow(y.view(dim, dim).detach().numpy())
    fig.savefig(opts['output_dir']+ opts['date'] + '/' + filename)


def test(autoencoder, test_loader, opts, device):
    # encoder.eval()
    # decoder.eval()
    autoencoder.eval()
    test_epochs = int(len(test_loader.dataset) / opts['batch_size'])
    test_iter = iter(test_loader)
    # for epoch in tqdm(test_epochs):
    test_data = next(test_iter)
    test_data = test_data.to(device)
    # if torch.cuda.is_available():
    #     test_data = test_data.cuda()
    reconst = autoencoder(Variable(test_data)).cpu()
    for i in range(len(reconst)):
        save_plot(test_data, reconst, opts, 'result_' + str(i) + '.png')


def fig_loss(loss, path):
    fig = plt.figure(figsize=(8, 8))
    # plt.plot(np.array(model_history['val_loss']), label='Validation loss')
    plt.plot(loss, label='loss')
    plt.ylabel('loss')
    plt.xlabel('iteration')
    plt.legend()
    fig.savefig(path + 'loss.png')

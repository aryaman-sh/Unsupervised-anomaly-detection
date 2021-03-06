"""
VAE model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class ConvVAE(nn.Module):
    def __init__(self, img_size, name):
        super(ConvVAE, self).__init__()
        # Parameters
        self.img_size = img_size
        self.input_size = 1 # channels
        self.name = name
        self.gf_dim = 16

        # Encoder
        self.encoder, self.res_encoder = encoder_layer(self.input_size, self.gf_dim)

        # hidden => mu
        self.fc1 = latent_layer_1(self.gf_dim)

        # hidden => logvar
        self.fc2 = latent_layer_2(self.gf_dim)

        ## Decoder
        self.decoder = decoder_layer(self.gf_dim)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc1(h)
        logvar = self.fc2(h)
        res = self.res_encoder(x)
        return mu, logvar, res

    def decode(self, z):
        z = self.decoder(z)
        return z

    def reparameterize(self, mu, logvar):
        # Repametrization trick to make backprop. possible
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar, res = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, res

    def sample(self, batch_size, device):
        # Saple latent variables z
        # latent_size = self.gf_dim*4 # Change when other latent layer
        sample = torch.randn(batch_size, 512, 2, 2, dtype=torch.double).to(device)  # batch_size, latent_size)
        return self.decode(sample)


def loss_function(recon_x, x, res, mu, logvar, weight = 1):
    # Autoencoder loss - reconstruction loss
    l2_loss = torch.sum((recon_x.view(-1, recon_x.numel()) - x.view(-1, x.numel())).pow(2)) # * 0.5

    # Latent loss
    kl_divergence_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    #0.5 * torch.sum(mu.pow(2) + logvar.pow(2) - 2.*torch.log(torch.abs(logvar)) - 1)
    # Residual loss
    true_residuals = torch.abs(x-recon_x)
    autoencoder_res_loss = torch.sum((res - true_residuals).pow(2))

    # Total loss sum of all
    return torch.sum(l2_loss + weight*kl_divergence_loss + autoencoder_res_loss), kl_divergence_loss, l2_loss, autoencoder_res_loss

def train_vae(model, train_loader, device, optimizer, epoch):
    # Params
    model.train()
    train_loss = 0
    train_lat_loss = 0
    train_gen_loss = 0
    train_res_loss = 0

    weight = 1 #(epoch+1)/25 if epoch < 25 else 1

    for batch_idx, (data, _) in enumerate(train_loader): #tqdm(enumerate(train_loader), total=len(train_loader), desc='train'):
        data = data.to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar, res = model(data.double())
        loss, lat_loss, gen_loss, res_loss  = loss_function(recon_batch, data, res, mu, logvar, weight)

        train_loss += loss.item()
        train_lat_loss += lat_loss.item()
        train_gen_loss += gen_loss.item()
        train_gen_loss += res_loss.item()

        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    train_lat_loss /= len(train_loader.dataset)
    train_gen_loss /= len(train_loader.dataset)
    train_res_loss /= len(train_loader.dataset)

    return train_loss, train_lat_loss, train_gen_loss, train_res_loss


def valid_vae(model, test_loader, device, epoch):
    model.eval()
    test_loss = 0
    test_lat_loss = 0
    test_gen_loss = 0
    test_res_loss = 0

    weight = 1#(epoch%25+1)/25 #epoch/50 if epoch < 50 else 1

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader): #tqdm(enumerate(test_loader), total=len(test_loader), desc='test'):
            data = data.to(device)

            recon_batch, mu, logvar, res = model(data.double())
            loss, lat_loss, gen_loss, res_loss = loss_function(recon_batch, data, res, mu, logvar, weight)

            test_loss += loss.item()
            test_lat_loss += lat_loss.item()
            test_gen_loss += gen_loss.item()
            test_res_loss += res_loss.item()

    test_loss /= len(test_loader.dataset)
    test_lat_loss /= len(test_loader.dataset)
    test_gen_loss /= len(test_loader.dataset)
    test_res_loss /= len(test_loader.dataset)

    return test_loss, test_lat_loss, test_gen_loss, test_res_loss

def plot_restored(path, img_batch, batch_size,img_nbr = 0, img_size=128):
    plt.imsave(path,img_batch.view(batch_size, 1, img_size, img_size)[img_nbr,0].detach().numpy())


class ResBlock_Down(nn.Module):
    """
    Res block for convolution.
    1) One Conv2d() layer and a following Conv2d(stride=1) with in_layer between.
    2) One a Conv2d() layer with input and output directly.
    Output is these two summarized
    """

    def __init__(self, input_size, out_layers, stride=2, padding=1, act=True):
        super(ResBlock_Down, self).__init__()
        self.input_size = input_size
        self.in_layers = input_size
        self.out_layers = out_layers
        self.stride = stride
        self.pad = padding
        self.act = act

        self.conv1 = nn.Conv2d(self.input_size, self.in_layers, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.in_layers)
        self.conv2 = nn.Conv2d(self.in_layers, self.out_layers, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.out_layers)

        if self.act:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.input_size, self.out_layers, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(self.out_layers),
                nn.LeakyReLU(0.2)
            )
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(self.input_size, self.out_layers, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(self.out_layers)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        if self.act:
            out = F.leaky_relu(self.bn2(self.conv2(out)), 0.2)
        else:
            out = self.bn2(self.conv2(out))

        out += self.shortcut(x)
        return out


class ResBlock_Up(nn.Module):
    """
    Res block for transpose convolution.
    1) One transpose Conv2d() layer and a following transpose Conv2d(stride=1) with in_layer between.
    2) One a transpose Conv2d() layer with input and output directly.
    Output is these two summarized
    """

    def __init__(self, input_size, out_layers, padding=1):
        super(ResBlock_Up, self).__init__()
        self.input_size = input_size
        self.in_layers = input_size
        self.out_layers = out_layers
        self.pad = padding

        # self.conv1 = nn.ConvTranspose2d(self.input_size, self.in_layers, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.input_size, self.in_layers, kernel_size=3, stride=2, padding=1)
        )
        self.bn1 = nn.BatchNorm2d(self.in_layers)
        # self.conv2 = nn.ConvTranspose2d(self.in_layers, self.out_layers, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(self.in_layers, self.out_layers, kernel_size=3, stride=1, padding=1)
        )
        self.bn2 = nn.BatchNorm2d(self.out_layers)

        self.shortcut = nn.Sequential(
            # nn.ConvTranspose2d(self.input_size, self.out_layers, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(self.input_size, self.out_layers, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_layers)
        )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = F.leaky_relu(self.bn2(self.conv2(out)), 0.2)
        out += F.leaky_relu(self.shortcut(x), 0.2)
        return out


def encoder_layer(input_size, gf_dim):
    """
    Encoder function returning encoder layer
    """

    fist_encoder_layer = nn.Sequential(
        nn.Conv2d(input_size, gf_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(gf_dim),
        nn.LeakyReLU(0.2)
    )

    encoder = nn.Sequential(
        fist_encoder_layer,
        ResBlock_Down(gf_dim, gf_dim),
        ResBlock_Down(gf_dim, gf_dim * 2),
        ResBlock_Down(gf_dim * 2, gf_dim * 4),
        ResBlock_Down(gf_dim * 4, gf_dim * 8),
        ResBlock_Down(gf_dim * 8, gf_dim * 16)
    )

    res_encoder = nn.Sequential(
        fist_encoder_layer,
        nn.Conv2d(gf_dim, gf_dim, kernel_size=3, padding=2, dilation=2),
        nn.BatchNorm2d(gf_dim),
        nn.LeakyReLU(0.2),
        nn.Conv2d(gf_dim, gf_dim * 2, kernel_size=3, padding=2, dilation=2),
        nn.BatchNorm2d(gf_dim * 2),
        nn.LeakyReLU(0.2),
        nn.Conv2d(gf_dim * 2, gf_dim, kernel_size=3, padding=2, dilation=2),
        nn.BatchNorm2d(gf_dim),
        nn.LeakyReLU(0.2),
        nn.Conv2d(gf_dim, 1, kernel_size=3, padding=2, dilation=2)
    )

    return encoder, res_encoder


def latent_layer_1(gf_dim):
    """
    Function used for returning mu
    """
    return ResBlock_Down(gf_dim * 16, gf_dim * 32, act=False)


def latent_layer_2(gf_dim):
    """
    Function used for returning std
    """
    return ResBlock_Down(gf_dim * 16, gf_dim * 32, act=False)


def decoder_layer(gf_dim):
    """
    Decoder function returning decoder layer
    """
    decoder = nn.Sequential(
        ResBlock_Up(gf_dim * 32, gf_dim * 16),
        ResBlock_Up(gf_dim * 16, gf_dim * 8),
        ResBlock_Up(gf_dim * 8, gf_dim * 4),
        ResBlock_Up(gf_dim * 4, gf_dim * 2),
        ResBlock_Up(gf_dim * 2, gf_dim),
        ResBlock_Up(gf_dim, gf_dim),
        nn.Conv2d(gf_dim, gf_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(gf_dim),
        nn.LeakyReLU(0.2),
        nn.Conv2d(gf_dim, 1, kernel_size=3, stride=1, padding=1)
    )
    return decoder



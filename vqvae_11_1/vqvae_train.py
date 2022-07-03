import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision as vision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


class Decoder(nn.Module):
    '''
    Decoder network
    params:
        in_channels: input channels (from vq)
        num_hidden: hidden convolution channels
        residual_inter: intermediary residual block channels
    '''

    def __init__(self, in_channels, num_hidden, residual_inter):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hidden,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.residual1 = Residual_block(
            in_channels=num_hidden,
            hidden_inter=residual_inter,
            hidden_final=num_hidden
        )
        self.residual2 = Residual_block(
            in_channels=num_hidden,
            hidden_inter=residual_inter,
            hidden_final=num_hidden
        )
        self.transpose1 = nn.ConvTranspose2d(
            in_channels=num_hidden,
            out_channels=num_hidden // 2,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.transpose2 = nn.ConvTranspose2d(
            in_channels=num_hidden // 2,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.residual1(x)
        x = self.residual2(x)
        x = self.transpose1(x)
        x = self.transpose2(x)
        return x


class Encoder(nn.Module):
    '''
    Encoder block
    params:
        in_channels = input channels
        num_hidden = hidden blocks for encoder convolution
        residual_inter = intermediary residual block channels
    '''

    def __init__(self, in_channels, num_hidden, residual_inter):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hidden // 2,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_hidden // 2,
            out_channels=num_hidden,
            kernel_size=4,
            stride=2,
            padding=1
        )
        self.residual1 = Residual_block(
            in_channels=num_hidden,
            hidden_inter=residual_inter,
            hidden_final=num_hidden
        )
        self.residual2 = Residual_block(
            in_channels=num_hidden,
            hidden_inter=residual_inter,
            hidden_final=num_hidden
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.residual1(x)
        x = self.residual2(x)
        return x


class Residual_block(nn.Module):
    '''
    Create new Residual block
    Params:
        in_channels: Input channels
        hidden_inter: hidden channels for intermediate convolution
        hidden_final: Number of channels for output convolution
    '''

    def __init__(self, in_channels, hidden_inter, hidden_final):
        super(Residual_block, self).__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_inter,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_inter,
                out_channels=hidden_final,
                kernel_size=1,
                stride=1,
                bias=False
            )
        )

    def forward(self, x):
        # Skip connection
        return x + self.net(x)


import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Create a Vector Quantizer module
    Params:
        num_embeddings: Number of embeddings in embeddings codebook
        embedding_dim: dim of each embedding in embeddings codebook
        commitment_cost: commitment term of the loss (beta in loss function)
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Create embedding table
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # the learnable weights of the module of shape (num_embeddings, embedding_dim) initialized from \mathcal{N}(0, 1)N(0,1)
        self.embedding.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, x):
        # convert inputs from BCHW to BHWC
        x = x.permute(0, 2, 3, 1).contiguous()  # keep memory contiguous
        x_shape = x.shape
        # Flatten
        # Each flattened layer is individually quantized
        flat_x = x.view(-1, self.embedding_dim)
        # Calculate distances
        # Find closest codebook vectors
        # find distance of encoded vector to all coded vectors
        # shape (#,num encodings)
        distances = (torch.sum(flat_x ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_x, self.embedding.weight.t()))

        # Encoding

        # return val for training
        train_indices_return = torch.argmin(distances, dim=1)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # min d

        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)  # place in encodings (eq to keras one-hot)

        # Quantize and unflatten
        # Multiply encodings table with embeddings
        quantized = torch.matmul(encodings, self.embedding.weight).view(x_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x)  # stop gradient propogation on quantized
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss  # loss fn (paper)

        quantized = x + (quantized - x).detach()  # when backprop end up with x (no gradient flow for other term)

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encodings, train_indices_return

    """
    Returns embedding corresponding to encoding index
    For one index
    """

    def get_quantized(self, x):
        encoding_indices = x.unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.embedding.weight).view(1, 64, 64, 64)
        return quantized.permute(0, 3, 1, 2).contiguous()


class VQVAE(nn.Module):
    """
    VQVAE model
    params:
        num_hiddens: Hidden blocks for encoder convolutions
        residual_inter: Intermediary residual block channels
        num_embeddings: Number of codebook embeddings
        embedding_dim: Dimensions of each embedding
        commitment_cost: loss function beta value
    """

    def __init__(self, num_hiddens, residual_inter,
                 num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()

        # Create the encoder
        self.encoder = Encoder(
            in_channels=1,
            num_hidden=num_hiddens,
            residual_inter=residual_inter
        )

        # initial conv Convert input dimensions to embedding dimension
        self.conv1 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1
        )

        # Create vector qunatizer
        self.vq = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost
        )

        # Create decoder
        self.decoder = Decoder(
            in_channels=embedding_dim,
            num_hidden=num_hiddens,
            residual_inter=residual_inter)

    def forward(self, x):
        # encode
        z = self.encoder(x)
        # change channel dim
        z = self.conv1(z)
        loss, quantized, _, _ = self.vq(z)
        # decode
        x_recon = self.decoder(quantized)  # reconstructed

        return loss, x_recon

BATCH_SIZE = 32
EPOCHS = 51
LR = 1e-3
DEVICE = 'cuda'
NUM_HIDDENS = 128 # hidden blocks for encoder convolution
RESIDUAL_INTER = 32 # intermediary residual block channels
NUM_EMBEDDINGS = 512 # number of embeddings for codebook
EMBEDDING_DIM = 64 # dimension of each embedding
COMMITMENT_COST = 0.25 # beta term in loss function
TRAIN_DATA_PATH = '/home/Student/s4606685/summer_research/oasis-3/png_data' # path to training data
TEST_DATA_PATH = '/home/Student/s4606685/summer_research/oasis-3/png_data' # path to test data
DATA_VARIANCE = 0.0338 # evaluated seperately on training data

transform = transforms.Compose(
    [
        transforms.Resize((256,256)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(
            0.5, 0.5
        ),
    ]
)

train_ds = vision.datasets.ImageFolder(TRAIN_DATA_PATH, transform=transform)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_ds = vision.datasets.ImageFolder(TEST_DATA_PATH, transform=transform)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

model = VQVAE(NUM_HIDDENS, RESIDUAL_INTER, NUM_EMBEDDINGS, EMBEDDING_DIM, COMMITMENT_COST)
model.to(DEVICE)

# Init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
training_reconstruction_loss = []


def train_function(dataloader, model, optimizer):
    size = len(dataloader.dataset)
    losses = []
    for batch, (X, _) in enumerate(dataloader):
        X = X.to(device=DEVICE)

        optimizer.zero_grad()
        vq_loss, data_recon = model(X)

        recon_error = F.mse_loss(data_recon, X) / DATA_VARIANCE
        loss = recon_error + vq_loss
        loss.backward()
        optimizer.step()
        losses.append(recon_error.item())

        if batch % 1200 == 0:
            print(sum(losses) / len(losses))

    losses = sum(losses) / len(losses)
    return losses


for i in range(EPOCHS):
    print(f"EPOCH = {i + 1}  ")
    lo = train_function(train_dl, model, optimizer)
    training_reconstruction_loss.append(lo)
    print(f"Reconstruction loss: {lo}")
    if i%10 == 0:
        filename = 'model-vqvae-epoch_' + str(i+1) + '.pth.tar'
        checkpoint = {'save_dic': model.state_dict()}
        torch.save(checkpoint, filename)


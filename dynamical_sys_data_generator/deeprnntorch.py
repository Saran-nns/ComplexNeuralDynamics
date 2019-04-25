from __future__ import print_function
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

seed= 10001
cuda = True
batch_size = 16

# Random seed 
torch.manual_seed(seed)

# CUDA if avaialble
device = torch.device("cuda" if cuda else "cpu")

# Load the data to GPU if available
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# Training data loader
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)

# Test data loader
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True, **kwargs)


# ENCODER MODEL
class RVAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,latent_size, output_size, dropout = 0):
        super(RVAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.dropout = dropout

        # ENCODER
        self.encoder = nn.Sequential(
            nn.LSTM(self.latent_size,self.hidden_size,self.num_layers,dropout = self.dropout,batch_first = True), 
            nn.ReLU())

        # Sample z ~ Q(z|X) : REPARAMETERIZATION: 
        self.z_mean = nn.Linear(self.hidden_size,self.latent_size)
        self.z_var = nn.Linear(self.hidden_size,self.latent_size)
        self.z_vector = self.sample_z(self.z_mean,self.z_var)

        # Q(X|Z) - DECODER
        self.decoder = nn.Sequential(
            nn.LSTM(self.latent_size,self.hidden_size,self.num_layers,dropout = self.dropout,batch_first = True),
            nn.ReLU())

        # DECODER TO FC LINEAR OUTPUT LAYER
        self.output = nn.Linear(self.hidden_size, self.output_size)  # Output size must equal the flatenned input size
      
    def sample_z(self, mean, logvar):
        stddev = torch.exp(0.5 * logvar)
        noise = Variable(torch.randn(stddev.size())) # Gaussian Noise
        return (noise * stddev) + mean

    # Q(z|X) - 
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Only hidden states;Ignored cell states
        mean = self.z_mean(x)
        var = self.z_var(x)
        return mean, var

    # Q(X|Z) -
    def decode(self, z):
        out = self.z_vector(z)
        out = out.view(z.size(0))
        out = self.decoder(out)
        return out

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar


def vae_loss(output, input, mean, logvar, loss_func):
    recon_loss = loss_func(output, input)
    kl_loss = torch.mean(0.5 * torch.sum(
        torch.exp(logvar) + mean**2 - 1. - logvar, 1))
    return recon_loss + kl_loss


def train(model, loader, loss_func, optimizer):
    model.train()
    for inputs, _ in loader:
        inputs = Variable(inputs)

        output, mean, logvar = model(inputs)
        loss = vae_loss(output, inputs, mean, logvar, loss_func)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# parameters 
N_STEPS = 28
N_INPUTS = 28
N_NEURONS = 150
N_OUTPUTS = 10
N_EPHOCS = 10

# Initiate model instance
model = RVAE(input_size = N_INPUTS, hidden_size = N_NEURONS, num_layers = 1,latent_size = 12, output_size = 10, dropout = 0).cuda()

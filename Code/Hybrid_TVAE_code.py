## CODE FROM: https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/tvae.py

## Changes in the original code will be segnalated with proper comments and the symbol (*)

"""
This file is a modified version of the original TVAE implementation:
https://github.com/sdv-dev/CTGAN/blob/main/ctgan/synthesizers/tvae.py

The only substantive change is that the encoder and decoder—
which were originally MLPs—have been replaced with Kolmogorov–Arnold Networks.
All other code and documentation remain unchanged.

For the original CTGAN design, see:
    Xu, L., Nightingale, A., & Krishnan, R. (2019).
    Modeling Tabular Data Using Conditional GAN.
    https://arxiv.org/abs/1907.00503
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state

# Import KAN (*)
from KAN_code import KAN, KANLinear

# (*) KAN ENCODER
class HybridEncoder(Module):
    """TVAE Ecoder with a single KAN block at layer 'kan_layer_idx',
    the rest remain as standard TVAE Linear Layers.

    Args:
        data_dim (int):
            Dimensions of the data.
        compress_dims (tuple or list of ints):
            Size of each hidden layer.
        embedding_dim (int):
            Size of the output vector.
        kan_layer_idx (int): index of KAN layer.
        grid_size, spline_order, etc. (int): 
            Hyperparameters for the KAN layers. 
    """

    def __init__(self, data_dim, compress_dims, embedding_dim, 
                 kan_layer_idx=0, grid_size=5, spline_order=3, scale_noise=0.1, 
                 scale_base=1.0, scale_spline=1.0, base_activation=torch.nn.SiLU,
                 grid_eps=0.02, grid_range=[-1, 1]):
        super(HybridEncoder, self).__init__()
        self.kan_layer_idx = kan_layer_idx

        # Swapping KAN layer when needed
        layers = []
        dim = data_dim
        for i, out_dim in enumerate(compress_dims):
            if i == kan_layer_idx:
                layers += [
                    KANLinear(dim, out_dim,
                              grid_size=grid_size,
                              spline_order=spline_order,
                              scale_noise=scale_noise,
                              scale_base=scale_base,
                              scale_spline=scale_spline,
                              base_activation=base_activation,
                              grid_eps=grid_eps,
                              grid_range=grid_range),
                    nn.SiLU() # SiLU for KAN
                ]
            else:
                layers += [
                    Linear(dim, out_dim),
                    ReLU()
                ]
            dim = out_dim
        
        self.seq = Sequential(*layers)
        # Final VAE heads unchanged
        self.fc_mu = Linear(dim, embedding_dim)
        self.fc_logvar = Linear(dim, embedding_dim)

    def forward(self, x):
        """Encode the passed input x."""
        h = self.seq(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5*logvar)
        return mu, std, logvar


# (*) KAN DECODER
class HybridDecoder(Module):
    """TVAE Decoder with a single KAN block at layer 'kan_layer_idx',
    the rest remain as standard TVAE Linear layers.

    Args:
        embedding_dim (int):
            Size of the input vector.
        decompress_dims (tuple or list of ints):
            Size of each hidden layer.
        data_dim (int):
            Dimensions of the data.
        kan_layer_idx (int): index of KAN layer.
        grid_size, spline_order, etc. (int): 
            Hyperparameters for the KAN layers. 
    """

    def __init__(self, embedding_dim, decompress_dims, data_dim,
                 kan_layer_idx=0, grid_size=5, spline_order=3, scale_noise=0.1, 
                 scale_base=1.0, scale_spline=1.0, base_activation=torch.nn.SiLU,
                 grid_eps=0.02, grid_range=[-1, 1]):
        super(HybridDecoder, self).__init__()
        self.kan_layer_idx = kan_layer_idx
        # Swapping KAN layer when needed
        layers = []
        dim = embedding_dim
        for i, out_dim in enumerate(decompress_dims):
            if i == kan_layer_idx:
                layers += [
                    KANLinear(dim, out_dim,
                              grid_size=grid_size,
                              spline_order=spline_order,
                              scale_noise=scale_noise,
                              scale_base=scale_base,
                              scale_spline=scale_spline,
                              base_activation=base_activation,
                              grid_eps=grid_eps,
                              grid_range=grid_range),
                    nn.SiLU() # SiLU for KAN
                ]
            else:
                layers += [
                    Linear(dim, out_dim),
                    ReLU()
                ]
            dim = out_dim
        
        # Final projection to data_dim
        layers.append(Linear(dim, data_dim))

        self.seq = Sequential(*layers)
        self.sigma = Parameter(torch.ones(data_dim) * 0.1)

    def forward(self, z):
        """Decode the passed input z"""
        x_recon = self.seq(z)
        return  x_recon, self.sigma
    

def _loss_function(recon_x, x, sigmas, mu, logvar, output_info, factor):
    st = 0
    loss = []
    for column_info in output_info:
        for span_info in column_info:
            if span_info.activation_fn != 'softmax':
                ed = st + span_info.dim
                std = sigmas[st]
                eq = x[:, st] - torch.tanh(recon_x[:, st])
                loss.append((eq**2 / 2 / (std**2)).sum())
                loss.append(torch.log(std) * x.size()[0])
                st = ed

            else:
                ed = st + span_info.dim
                loss.append(
                    cross_entropy(
                        recon_x[:, st:ed], torch.argmax(x[:, st:ed], dim=-1), reduction='sum'
                    )
                )
                st = ed

    assert st == recon_x.size()[1]
    KLD = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
    return sum(loss) * factor / x.size()[0], KLD / x.size()[0]

# (*) KAN_TVAE
class HYBRID_KAN_TVAE(BaseSynthesizer):
    """HYBRID_KAN_TVAE. Instead of MLPs, the Encoder and Decoder
       use Kolmogorov-Arnold Networks (KANs), only in one layer"""

    def __init__(
        self,
        embedding_dim=128,
        compress_dims=(128, 128),
        decompress_dims=(128, 128),
        grid_size_enc=5, 
        spline_order_enc=3,
        grid_size_dec=5,
        spline_order_dec=5,
        l2scale=1e-5,
        batch_size=500,
        epochs=300,
        loss_factor=2,
        cuda=True,
        verbose=False,
    ):
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims

        # (*) Hyperparametrs for the KANs.
        self.grid_size_enc = grid_size_enc
        self.spline_order_enc = spline_order_enc
        self.grid_size_dec = grid_size_dec
        self.spline_order_dec = spline_order_dec

        self.l2scale = l2scale
        self.batch_size = batch_size
        self.loss_factor = loss_factor
        self.epochs = epochs
        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        self.verbose = verbose

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

    # (*) HYBRID_KAN_TVAE Fit
    @random_state
    def fit(self, train_data, discrete_columns=()):
        """Fit the HYBRID_KAN_TVAE Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self.transformer = DataTransformer()
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)
        dataset = TensorDataset(torch.from_numpy(train_data.astype('float32')).to(self._device))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        data_dim = self.transformer.output_dimensions
        # (*) Use HybridEncder and HybridDecoder
        encoder = HybridEncoder(data_dim, self.compress_dims, self.embedding_dim).to(self._device)
        self.decoder = HybridDecoder(self.embedding_dim, self.decompress_dims, data_dim).to(self._device)
        optimizerAE = Adam(
            list(encoder.parameters()) + list(self.decoder.parameters()), weight_decay=self.l2scale
        )

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Batch', 'Loss'])
        iterator = tqdm(range(self.epochs), disable=(not self.verbose))
        if self.verbose:
            iterator_description = 'Loss: {loss:.3f}'
            iterator.set_description(iterator_description.format(loss=0))

        for i in iterator:
            loss_values = []
            batch = []
            for id_, data in enumerate(loader):
                optimizerAE.zero_grad()
                real = data[0].to(self._device)
                mu, std, logvar = encoder(real)
                eps = torch.randn_like(std)
                emb = eps * std + mu
                rec, sigmas = self.decoder(emb)
                loss_1, loss_2 = _loss_function(
                    rec,
                    real,
                    sigmas,
                    mu,
                    logvar,
                    self.transformer.output_info_list,
                    self.loss_factor,
                )
                loss = loss_1 + loss_2
                loss.backward()
                optimizerAE.step()
                self.decoder.sigma.data.clamp_(0.01, 1.0)

                batch.append(id_)
                loss_values.append(loss.detach().cpu().item())

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i] * len(batch),
                'Batch': batch,
                'Loss': loss_values,
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat([self.loss_values, epoch_loss_df]).reset_index(
                    drop=True
                )
            else:
                self.loss_values = epoch_loss_df

            if self.verbose:
                iterator.set_description(
                    iterator_description.format(loss=loss.detach().cpu().item())
                )

    @random_state
    def sample(self, samples):
        """Sample data similar to the training data.

        Args:
            samples (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        self.decoder.eval()

        steps = samples // self.batch_size + 1
        data = []
        for _ in range(steps):
            mean = torch.zeros(self.batch_size, self.embedding_dim)
            std = mean + 1
            noise = torch.normal(mean=mean, std=std).to(self._device)
            fake, sigmas = self.decoder(noise)
            fake = torch.tanh(fake)
            data.append(fake.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:samples]
        return self.transformer.inverse_transform(data, sigmas.detach().cpu().numpy())

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        self.decoder.to(self._device)
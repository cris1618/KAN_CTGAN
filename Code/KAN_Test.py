from KAN_code import KAN
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

# Testing parameters
noise_dim = 100
condition_dim = 100
input_dim = noise_dim + condition_dim
output_dim = 50

# Define the hidden layer of the model
hidden_layers = [input_dim, 128, 128, output_dim]

# Model
generator_KAN = KAN(
    layers_hidden=hidden_layers,
    grid_size=5,
    spline_order=3,
    scale_noise=0.1,
    scale_base=1.0,
    scale_spline=1.0,
    base_activation=torch.nn.SiLU,
    grid_eps=0.02,
    grid_range=[-1,1]
)

generator_KAN.eval()

batch_size = 32
noise = torch.randn(batch_size, noise_dim, dtype=torch.float64)
condition = torch.randn(batch_size, condition_dim, dtype=torch.float64)
input_data = torch.cat([noise, condition], dim=1)

# Pass to the model
output_data = generator_KAN(input_data)

# Print some information to check correct dimensions
print(f"Input shape: {input_data.shape}") # Expected (32, 200) noise_dim + condition_dim = 200
print(f"Output shape: {output_data.shape}") # Expected (32, 50) output_dim = 50
print(f"Sample output (first instance):", output_data[0])
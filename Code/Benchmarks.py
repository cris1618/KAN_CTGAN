import sdgym
from sdgym import create_single_table_synthesizer
import warnings
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional
from tqdm import tqdm
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.errors import InvalidDataError
from ctgan.synthesizers.base import BaseSynthesizer, random_state
from KAN_CTGAN_code import KAN_CTGAN, Generator_KAN, Discriminator_KAN
from KAN_TVAE_code import KAN_TVAE, KAN_Encoder, KAN_Decoder
from Hybrid_CTGAN_code import KAN_HYBRID_CTGAN
from Disc_KAN_CTGAN_code import Disc_KAN_CTGAN
from ctgan import CTGAN, TVAE


# Models to test
synthetizers = ["GaussianCopulaSynthesizer", "CTGANSynthesizer", "TVAESynthesizer"]

# ORIGINAL MODELS BUT MANUALLY IMPORTED
def get_trained_synthesizer_ORIGINAL_CTGAN(data, metadata):
    discrete = [
        col_name
        for col_name, col_info in metadata["columns"].items()
        if col_info["sdtype"] == "categorical"
    ]

    # Initialize the KAN_CTGAN model
    synth = CTGAN()

    # Train on the provided testing datasets
    synth.fit(data, discrete_columns=discrete)

    return synth

def sample_from_synthesizer_ORIGINAL_CTGAN(synthesizer, n_rows):
    return synthesizer.sample(n_rows)

ORIGINAL_CTGAN_synth = create_single_table_synthesizer(
    get_trained_synthesizer_fn=get_trained_synthesizer_ORIGINAL_CTGAN,
    sample_from_synthesizer_fn=sample_from_synthesizer_ORIGINAL_CTGAN,
    display_name="ORIGINAL-CTGAN"
)

print("ORIGINAL-CTGAN CREATED")

# TVAE ORIGINAL
def get_trained_synthesizer_ORIGINAL_TVAE(data, metadata):
    discrete = [
        col_name
        for col_name, col_info in metadata["columns"].items()
        if col_info["sdtype"] == "categorical"
    ]

    # Initialize the KAN_CTGAN model
    synth = TVAE()

    # Train on the provided testing datasets
    synth.fit(data, discrete_columns=discrete)

    return synth

def sample_from_synthesizer_ORIGINAL_TVAE(synthesizer, n_rows):
    return synthesizer.sample(n_rows)

ORIGINAL_TVAE_synth = create_single_table_synthesizer(
    get_trained_synthesizer_fn=get_trained_synthesizer_ORIGINAL_TVAE,
    sample_from_synthesizer_fn=sample_from_synthesizer_ORIGINAL_TVAE,
    display_name="ORIGINAL-TVAE"
)

print("ORIGINAL-TVAE CREATED")

# Create Custom Synthesizer: KAN_CTGAN
def get_trained_synthesizer_KAN_CTGAN(data, metadata):
    print("METADATA KEYS:", metadata)
    discrete = [
        col_name
        for col_name, col_info in metadata["columns"].items()
        if col_info["sdtype"] == "categorical"
    ]

    # Initialize the KAN_CTGAN model
    synth = KAN_CTGAN()

    # Train on the provided testing datasets
    synth.fit(data, discrete_columns=discrete)

    return synth

def sample_from_synthesizer_KAN_CTGAN(synthesizer, n_rows):
    return synthesizer.sample(n_rows)

KAN_CTGAN_synth = create_single_table_synthesizer(
    get_trained_synthesizer_fn=get_trained_synthesizer_KAN_CTGAN,
    sample_from_synthesizer_fn=sample_from_synthesizer_KAN_CTGAN,
    display_name="KAN-CTGAN"
)

print("KAN-CTGAN CREATED")

# Create Custom Synthesizer: Disc KAN CTGAN
def get_trained_synthesizer_Disc_KAN_CTGAN(data, metadata):
    
    discrete = [
        col_name
        for col_name, col_info in metadata["columns"].items()
        if col_info["sdtype"] == "categorical"
    ]

    # Initialize Hybrid_CTGAN
    synth = Disc_KAN_CTGAN()

    # Training
    synth.fit(data, discrete_columns=discrete)

    return synth

def sample_from_synthesizer_Disc_KAN_CTGAN(synthesizer, n_rows):
    return synthesizer.sample(n_rows)

KAN_HYBRID_CTGAN_synth = create_single_table_synthesizer(
    get_trained_synthesizer_fn=get_trained_synthesizer_Disc_KAN_CTGAN,
    sample_from_synthesizer_fn=sample_from_synthesizer_Disc_KAN_CTGAN,
    display_name="Disc_KAN_CTGAN"
)

print("Disc-KAN-CTGAN CREATED")


# Create Custom Synthesizer: Hybrid_CTGAN
def get_trained_synthesizer_Hybrid_CTGAN(data, metadata):
    
    discrete = [
        col_name
        for col_name, col_info in metadata["columns"].items()
        if col_info["sdtype"] == "categorical"
    ]

    # Initialize Hybrid_CTGAN
    synth = KAN_HYBRID_CTGAN()

    # Training
    synth.fit(data, discrete_columns=discrete)

    return synth

def sample_from_synthesizer_Hybrid_CTGAN(synthesizer, n_rows):
    return synthesizer.sample(n_rows)

KAN_HYBRID_CTGAN_synth = create_single_table_synthesizer(
    get_trained_synthesizer_fn=get_trained_synthesizer_Hybrid_CTGAN,
    sample_from_synthesizer_fn=sample_from_synthesizer_Hybrid_CTGAN,
    display_name="KAN_HYBRID_CTGAN"
)

print("HYBRID-KAN-CTGAN CREATED")


# Create Custom Synthesizer: KAN_TAVE
def get_trained_synthesizer_KAN_TVAE(data, metadata):
    
    discrete = [
        col_name
        for col_name, col_info in metadata["columns"].items()
        if col_info["sdtype"] == "categorical"
    ]

    # Initialize KAN_TVAE
    synth = KAN_TVAE()

    # Training
    synth.fit(data, discrete_columns=discrete)

    return synth

def sample_from_synthesizer_KAN_TVAE(synthesizer, n_rows):
    return synthesizer.sample(n_rows)

KAN_TVAE_synth = create_single_table_synthesizer(
    get_trained_synthesizer_fn=get_trained_synthesizer_KAN_TVAE,
    sample_from_synthesizer_fn=sample_from_synthesizer_KAN_TVAE,
    display_name="KAN_TVAE"
)

print("KAN-TVAE CREATED")

# Output file path
output_filepath = r"C:\Users\Utente\OneDrive\Desktop\Thesis\Benchmarks\SDGym_comparison_TEST_HYBRID_With_ORIGINALS.csv"

results = sdgym.benchmark_single_table(
    synthesizers=synthetizers,
    custom_synthesizers=[ORIGINAL_CTGAN_synth, ORIGINAL_TVAE_synth, KAN_CTGAN_synth, KAN_TVAE_synth, KAN_HYBRID_CTGAN_synth],
    sdv_datasets=["adult"],
    output_filepath=output_filepath,
    limit_dataset_size=True,
    show_progress=True
)
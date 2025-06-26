import sdgym
from sdgym import create_single_table_synthesizer


# Models to test
synthetizers = ["GaussianCopulaSynthesizer", "CTGANSynthesizer", "TVAESynthesizer"]

# Create Custom Synthesizer: KAN_CTGAN
def get_trained_synthesizer_KAN_CTGAN(data, metadata):
    return

def sample_from_synthesizer_KAN_CTGAN(synthesizer, n_rows):
    return 

KAN_CTGAN = create_single_table_synthesizer(
    get_trained_synthesizer_fn=get_trained_synthesizer_KAN_CTGAN,
    sample_from_synthesizer_fn=sample_from_synthesizer_KAN_CTGAN,
    display_name="KAN_CTGAN"
)

# Create Custom Synthesizer: KAN_TAVE
def get_trained_synthesizer_KAN_TVAE(data, metadata):
    return

def sample_from_synthesizer_KAN_TVAE(synthesizer, n_rows):
    return 

KAN_TVAE = create_single_table_synthesizer(
    get_trained_synthesizer_fn=get_trained_synthesizer_KAN_TVAE,
    sample_from_synthesizer_fn=sample_from_synthesizer_KAN_TVAE,
    display_name="KAN_TVAE"
)

#results = sdgym.benchmark_single_table()
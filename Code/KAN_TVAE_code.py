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

# Import KAN (*)
from KAN_code import KAN, KANLinear
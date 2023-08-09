import os
import sys

from itertools import product

import scanpy as sc
import scvi
import numpy as np

# Necessary to import models as below
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.inVAE import FinVAE, NFinVAE
from src.inVAE.utils import sparse_shift, mcc, get_linear_score, prepare_params_decoder, decoder, synthetic_data

import argparse

print(f'The current directory is: {os.getcwd()}')

### Read in all hyperparameters/settings we specify in advance
parser = argparse.ArgumentParser(description='Single-cell HP for simulation')

## Data arguments -----------------------------------------------------------------
parser.add_argument('--latent_dim', type=int, default=10)
parser.add_argument('--latent_dim_inv', type=int, default=8)
parser.add_argument('--n_genes', type=int, default=100)
parser.add_argument('--n_epochs', type=int, default=2000)
parser.add_argument('--N_RUNS', type=int, default=1)
parser.add_argument('--device', default='cpu')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--correlate_latents', action='store_true')

args = parser.parse_args()

latent_dim = args.latent_dim
latent_dim_inv = args.latent_dim_inv
latent_dim_spur = latent_dim - latent_dim_inv

# dict for results
beta_list = [0.5, 1., 2., 4., 10.]

results_dict = {}

for beta in beta_list:
    results_dict[f'mcc_pear_f_invae_{beta}'] = []
    results_dict[f'mcc_spear_f_invae_{beta}'] = []
    results_dict[f'r2_f_invae_{beta}'] = []

# Simulate data
adata = synthetic_data(
    n_cells_per_comb = args.n_samples,
    n_cell_types = 2,
    n_conditions = 3,
    n_latent_inv = latent_dim_inv,
    n_latent_spur = latent_dim_spur,
    n_genes = args.n_genes,
    shift_cell_type = [-20, 20],
    shift_conditions = [0, 5, 5],
    mean_batch = [1],
    var_batch = None,
    verbose = False,
    correlate = args.correlate_latents
)

gt_latent = adata.obsm['groundtruth_latent']

# Settings for FinVAE
inv_covar_keys = {
    'cont': [],
    'cat': ['cell_type','disease'] #set to the keys in the adata
}

spur_covar_keys = {
    'cont': [],
    'cat': ['batch'] #set to the keys in the adata
}

for beta in beta_list:
    for i in range(args.N_RUNS):
        # FinVAE
        model = FinVAE(
            adata = adata,
            layer = 'raw', # The layer where the raw counts are stored in adata (None for adata.X: default)
            latent_dim_inv=latent_dim_inv,
            latent_dim_spur=latent_dim_spur,
            hidden_dim = 128,
            inv_covar_keys = inv_covar_keys,
            spur_covar_keys = spur_covar_keys,
            kl_rate=beta,
            elbo_version='sample',
            device=args.device,
            batch_size=args.batch_size
        )

        model.train(n_epochs = 2000, lr_train=0.001, weight_decay=0.0001)

        latent = model.get_latent_representation(latent_type='full')

        #adata.obsm[f'X_FinVAE_full_{i}'] = latent

        results_dict[f'mcc_pear_f_invae_{beta}'].append(mcc(gt_latent, latent, method='pearson'))

        results_dict[f'mcc_spear_f_invae_{beta}'].append(mcc(gt_latent, latent, method='spearman'))

        results_dict[f'r2_f_invae_{beta}'].append(get_linear_score(gt_latent, latent))

for key, score_list in results_dict.items():
    print(f'{key}: {np.mean(score_list):.3f} +- {np.std(score_list):.3f}')
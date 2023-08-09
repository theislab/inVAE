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

parser.add_argument('--N_RUNS', type=int, default=1)
parser.add_argument('--device', default='cpu')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--n_samples', type=int, default=100)
parser.add_argument('--beta', default=10)

parser.add_argument('--correlate_latents', action='store_true')

args = parser.parse_args()

latent_dim = args.latent_dim
latent_dim_inv = args.latent_dim_inv
latent_dim_spur = latent_dim - latent_dim_inv

# dict for results
results_dict = {
    'mcc_pear_scvi': [],
    'mcc_spear_scvi': [],
    'r2_scvi': [],
    'mcc_pear_scanvi': [],
    'mcc_spear_scanvi': [],
    'r2_scanvi': [],
    'mcc_pear_f_invae': [],
    'mcc_spear_f_invae': [],
    'r2_f_invae': [],
    'mcc_pear_nf_invae': [],
    'mcc_spear_nf_invae': [],
    'r2_nf_invae': []
}

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

# Settings for FinVAE and NFinVAE
inv_covar_keys = {
    'cont': [],
    'cat': ['cell_type','disease'] #set to the keys in the adata
}

spur_covar_keys = {
    'cont': [],
    'cat': ['batch'] #set to the keys in the adata
}

for i in range(args.N_RUNS):
    # SCVI
    scvi.model.SCVI.setup_anndata(adata, layer="raw", batch_key="batch")

    vae = scvi.model.SCVI(adata, n_layers=2, n_latent=latent_dim, gene_likelihood="nb")

    vae.train()

    latent_scvi = vae.get_latent_representation(adata)

    #adata.obsm[f'X_scvi_{i}'] = latent_scvi

    # Save results
    results_dict['mcc_pear_scvi'].append(mcc(gt_latent, latent_scvi, method='pearson'))

    results_dict['mcc_spear_scvi'].append(mcc(gt_latent, latent_scvi, method='spearman'))

    results_dict['r2_scvi'].append(get_linear_score(gt_latent, latent_scvi))

    # SCANVI
    lvae = scvi.model.SCANVI.from_scvi_model(
        vae,
        adata=adata,
        labels_key="cell_type",
        unlabeled_category="Unknown",
    )

    lvae.train(max_epochs=20, n_samples_per_label=100)

    latent_scanvi = lvae.get_latent_representation(adata)

    #adata.obsm[f'X_scanvi_{i}'] = latent_scanvi

    # Save results
    results_dict['mcc_pear_scanvi'].append(mcc(gt_latent, latent_scanvi, method='pearson'))

    results_dict['mcc_spear_scanvi'].append(mcc(gt_latent, latent_scanvi, method='spearman'))

    results_dict['r2_scanvi'].append(get_linear_score(gt_latent, latent_scanvi))

    # FinVAE
    model = FinVAE(
        adata = adata,
        layer = 'raw', # The layer where the raw counts are stored in adata (None for adata.X: default)
        latent_dim_inv=latent_dim_inv,
        latent_dim_spur=latent_dim_spur,
        hidden_dim = 128,
        inv_covar_keys = inv_covar_keys,
        spur_covar_keys = spur_covar_keys,
        kl_rate = args.beta,
        elbo_version='sample',
        device=args.device,
        batch_size=args.batch_size
    )

    model.train(n_epochs = 2000, lr_train=0.001, weight_decay=0.0001)

    latent = model.get_latent_representation(latent_type='full')

    #adata.obsm[f'X_FinVAE_full_{i}'] = latent

    results_dict['mcc_pear_f_invae'].append(mcc(gt_latent, latent, method='pearson'))

    results_dict['mcc_spear_f_invae'].append(mcc(gt_latent, latent, method='spearman'))

    results_dict['r2_f_invae'].append(get_linear_score(gt_latent, latent))

    # NFinVAE
    model = NFinVAE(
        adata = adata,
        layer = 'raw', # The layer where the raw counts are stored in adata (None for adata.X: default)
        latent_dim_inv=latent_dim_inv,
        latent_dim_spur=latent_dim_spur,
        hidden_dim = 128,
        inv_covar_keys = inv_covar_keys,
        spur_covar_keys = spur_covar_keys,
        device=args.device,
        batch_size=args.batch_size
    )

    model.train(n_epochs = 2000, lr_train=0.001, weight_decay=0.0001)

    latent = model.get_latent_representation(latent_type='full')

    #adata.obsm[f'X_NFinVAE_full_{i}'] = latent

    results_dict['mcc_pear_nf_invae'].append(mcc(gt_latent, latent, method='pearson'))

    results_dict['mcc_spear_nf_invae'].append(mcc(gt_latent, latent, method='spearman'))

    results_dict['r2_nf_invae'].append(get_linear_score(gt_latent, latent))

for key, score_list in results_dict.items():
    print(f'{key}: {np.mean(score_list):.3f} +- {np.std(score_list):.3f}')
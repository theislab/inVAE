import os
import sys

from itertools import product

import scanpy as sc

# Necessary to import models as below
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.inVAE import FinVAE, NFinVAE

# Change to False to only run one function call
FULL_TEST = True

def debug_models(model_name = 'NFinVAE', decoder_dist = 'nb', device = 'cpu'):
    adata = sc.read(f'./data/biological/multiome_gex_processed_training.h5ad')

    sc.pp.highly_variable_genes(adata, layer='log_norm', n_top_genes=2000)

    if decoder_dist == 'normal':
        sc.pp.scale(adata, layer='counts', zero_center=True, copy=False)

    # Split data into train and test environments (with batch names)

    # First save site and donor info sep

    # site:
    adata.obs['site'] = [s[1] for s in adata.obs['batch']]

    # donor:
    adata.obs['donor'] = [s[3] for s in adata.obs['batch']]

    train_batch_names = ['s1d1', 's1d2', 's2d1']
    val_batch_names = ['s2d4']
    test_batch_names = ['s3d6']

    adata_train = adata[
        adata.obs.batch.isin(train_batch_names), 
        adata.var.highly_variable
    ]

    adata_val = adata[
        adata.obs.batch.isin(val_batch_names),
        adata.var.highly_variable
    ]

    adata_test = adata[
        adata.obs.batch.isin(test_batch_names), 
        adata.var.highly_variable
    ]

    inv_covar_keys = {
        'cont': [],
        'cat': ['cell_type', 'donor']
    }

    spur_covar_keys = {
        'cont': [],
        'cat': ['site']
    }

    if model_name == 'FinVAE':
        model = FinVAE(
            adata = adata_train,
            layer = 'counts',
            inv_covar_keys = inv_covar_keys,
            spur_covar_keys = spur_covar_keys,
            latent_dim = 10, 
            latent_dim_spur = 1,
            n_layers = 2, 
            hidden_dim = 128,
            activation = 'relu', 
            slope = 0.1, # only needed when activation is Leaky ReLU ('lrelu')
            device = device,
            normalize_constant = 1.0, 
            fix_mean_prior = True,
            fix_var_prior = False,
            decoder_dist = decoder_dist,
            batch_norm = True,
            kl_rate = 1.0,
            batch_size = 256,
            elbo_version = 'sample',
        )
    else:
        model = NFinVAE(
            adata = adata_train,
            layer = 'counts',
            inv_covar_keys = inv_covar_keys,
            spur_covar_keys = spur_covar_keys,
            latent_dim = 10, 
            latent_dim_spur = 1,
            n_layers = 2, 
            hidden_dim = 128,
            activation = 'relu', 
            slope = 0.1, # only needed when activation is Leaky ReLU ('lrelu')
            device = device,
            normalize_constant = 1.0, 
            fix_mean_spur_prior = True,
            fix_var_spur_prior = False,
            decoder_dist = decoder_dist,
            batch_norm = True,
            batch_size = 256,
            reg_sm = 0.01,
            output_dim_prior_nn = None,
            hidden_dim_prior = None,
            n_layers_prior = None,
        )

    model.train(n_epochs=1, lr_train=0.001, weight_decay=0.0001)

    latent = model.get_latent_representation(type='inv')

    latent_val = model.get_latent_representation(adata_val, type='inv')

    neg_elbo = model.get_negative_elbo()

    neg_elbo_val = model.get_negative_elbo(adata_val)
    
    print(f'The shape of the latents is: "train"-{latent.shape}, "val"-{latent_val.shape}')
    print(f'The negative elbo is: "train"-{neg_elbo:.2f}, "val"-{neg_elbo_val:.2f}')

def debug_saving_loading(model_name = 'NFinVAE', decoder_dist = 'nb', device_from = 'cpu', device_to = 'cpu', only_loading = False):
    adata = sc.read(f'./data/biological/multiome_gex_processed_training.h5ad')

    sc.pp.highly_variable_genes(adata, layer='log_norm', n_top_genes=2000)

    if decoder_dist == 'normal':
        sc.pp.scale(adata, layer='counts', zero_center=True, copy=False)

    # Split data into train and test environments (with batch names)

    # First save site and donor info sep

    # site:
    adata.obs['site'] = [s[1] for s in adata.obs['batch']]

    # donor:
    adata.obs['donor'] = [s[3] for s in adata.obs['batch']]

    train_batch_names = ['s1d1', 's1d2', 's2d1']
    val_batch_names = ['s2d4']
    test_batch_names = ['s3d6']

    adata_train = adata[
        adata.obs.batch.isin(train_batch_names), 
        adata.var.highly_variable
    ].copy()

    adata_val = adata[
        adata.obs.batch.isin(val_batch_names),
        adata.var.highly_variable
    ].copy()

    adata_test = adata[
        adata.obs.batch.isin(test_batch_names), 
        adata.var.highly_variable
    ].copy()

    inv_covar_keys = {
        'cont': [],
        'cat': ['cell_type', 'donor']
    }

    spur_covar_keys = {
        'cont': [],
        'cat': ['site']
    }

    if model_name == 'FinVAE':
        hp_dict = dict(
            layer = 'counts',
            inv_covar_keys = inv_covar_keys,
            spur_covar_keys = spur_covar_keys,
            latent_dim = 10, 
            latent_dim_spur = 1,
            n_layers = 2, 
            hidden_dim = 128,
            activation = 'relu', 
            slope = 0.1, # only needed when activation is Leaky ReLU ('lrelu')
            normalize_constant = 1.0, 
            fix_mean_prior = True,
            fix_var_prior = False,
            decoder_dist = decoder_dist,
            batch_norm = True,
            kl_rate = 1.0,
            batch_size = 256,
            elbo_version = 'sample',
        )

        model = FinVAE(
            adata = adata_train,
            device = device_from,
            **hp_dict
        )
    else:
        hp_dict = dict(
            layer = 'counts',
            inv_covar_keys = inv_covar_keys,
            spur_covar_keys = spur_covar_keys,
            latent_dim = 10, 
            latent_dim_spur = 1,
            n_layers = 2, 
            hidden_dim = 128,
            activation = 'relu', 
            slope = 0.1, # only needed when activation is Leaky ReLU ('lrelu')
            normalize_constant = 1.0, 
            fix_mean_spur_prior = True,
            fix_var_spur_prior = False,
            decoder_dist = decoder_dist,
            batch_norm = True,
            batch_size = 256,
            reg_sm = 0.01,
            output_dim_prior_nn = None,
            hidden_dim_prior = None,
            n_layers_prior = None,
        )

        model = NFinVAE(
            adata = adata_train,
            device = device_from,
            **hp_dict
        )
    
    if not only_loading:
        model.train(n_epochs=5, lr_train=0.001, weight_decay=0.0001, print_every_n_epochs=1)

        model.save(f'./checkpoints/test_general_invae/debuger_{model_name}.pt')

    if model_name == 'FinVAE':
        model_new = FinVAE(
            adata_train,
            device=device_to,
            **hp_dict
        )
    else:
        model_new = NFinVAE(
            adata_train,
            device=device_to,
            **hp_dict
        )

    model_new.load(f'./checkpoints/test_general_invae/debuger_{model_name}.pt')

    latent = model_new.get_latent_representation()
    latent_val = model_new.get_latent_representation(adata_val)

    neg_elbo = model_new.get_negative_elbo()
    neg_elbo_val = model_new.get_negative_elbo(adata_val)
    
    print(f'The shape of the latents is: "train"-{latent.shape}, "val"-{latent_val.shape}')
    print(f'The negative elbo is: "train"-{neg_elbo:.2f}, "val"-{neg_elbo_val:.2f}')

def debug_train_classifier(
        adata_train,
        adata_val,
        adata_test,
        model_name = 'NFinVAE', 
        decoder_dist = 'nb', 
        layer = 'raw_counts',
        device = 'cuda', 
        inject_covar_in_latent = False, 
        elbo_version = 'sample',
        fix_mean_prior = True,
        fix_var_prior = False,
        inv_covars = ['cell_type', 'donor'],
        spur_covars = ['site']
    ):
    inv_covar_keys = {
        'cont': [],
        'cat': inv_covars
    }

    spur_covar_keys = {
        'cont': [],
        'cat': spur_covars
    }

    if model_name == 'FinVAE':
        model = FinVAE(
            adata = adata_train,
            layer = layer,
            inv_covar_keys = inv_covar_keys,
            spur_covar_keys = spur_covar_keys,
            latent_dim_inv = 9, 
            latent_dim_spur = 1,
            n_layers = 2, 
            hidden_dim = 128,
            activation = 'relu', 
            slope = 0.1, # only needed when activation is Leaky ReLU ('lrelu')
            device = device,
            normalize_constant = 1.0, 
            fix_mean_prior = fix_mean_prior,
            fix_var_prior = fix_var_prior,
            decoder_dist = decoder_dist,
            batch_norm = True,
            dropout = 0.1,
            kl_rate = 1.0,
            batch_size = 256,
            elbo_version = elbo_version,
            inject_covar_in_latent = inject_covar_in_latent
        )
    else:
        model = NFinVAE(
            adata = adata_train,
            layer = layer,
            inv_covar_keys = inv_covar_keys,
            spur_covar_keys = spur_covar_keys,
            latent_dim_inv = 9, 
            latent_dim_spur = 1,
            n_layers = 2, 
            hidden_dim = 128,
            activation = 'relu', 
            slope = 0.1, # only needed when activation is Leaky ReLU ('lrelu')
            device = device,
            normalize_constant = 1.0, 
            fix_mean_spur_prior = fix_mean_prior,
            fix_var_spur_prior = fix_var_prior,
            decoder_dist = decoder_dist,
            batch_norm = True,
            dropout = 0.1,
            batch_size = 256,
            reg_sm = 0.01,
            output_dim_prior_nn = None,
            hidden_dim_prior = None,
            n_layers_prior = None,
            inject_covar_in_latent = inject_covar_in_latent
        )

    model.train(n_epochs=1, lr_train=0.001, weight_decay=0.0001)

    model.train_classifier(
        adata_val,
        batch_key = 'batch',
        label_key = 'cell_type',
        n_epochs_train_class = 100,
        n_epochs_opt_val = 1,
        nr_samples = 2,
        hidden_dim_class = 50,
        n_layers_class = 1,
        act_class = 'relu',
        lr_train_class = 0.01,
        lr_opt_val = 0.001,
        class_print_every_n_epochs = 100,
        opt_val_print_every_n_epochs = 10
    )

    pred_train = model.predict(adata_train, dataset_type='train')
    pred_val = model.predict(adata_val, dataset_type='val')
    pred_train = model.predict(adata_test, dataset_type='test')

if __name__ == '__main__':
    #debug_models(model_name = 'NFinVAE', decoder_dist = 'nb', device = 'cuda')
    #debug_saving_loading(model_name = 'NFinVAE', decoder_dist = 'nb', device_from = 'cpu', device_to = 'cpu', only_loading=True)
    print(os.getcwd())
    adata = sc.read(f'./data/multiome_gex_processed_training.h5ad')

    sc.pp.highly_variable_genes(adata, layer='log_norm', n_top_genes=2000)

    adata.layers['raw_counts'] = adata.layers['counts'].copy()

    sc.pp.scale(adata, layer='counts', zero_center=True, copy=False)

    # Split data into train and test environments (with batch names)

    # First save site and donor info sep

    # site:
    adata.obs['site'] = [s[1] for s in adata.obs['batch']]

    # donor:
    adata.obs['donor'] = [s[3] for s in adata.obs['batch']]

    train_batch_names = ['s1d1', 's1d2', 's2d1']
    val_batch_names = ['s2d4']
    test_batch_names = ['s3d6']

    adata_train = adata[
        adata.obs.batch.isin(train_batch_names), 
        adata.var.highly_variable
    ].copy()

    adata_val = adata[
        adata.obs.batch.isin(val_batch_names),
        adata.var.highly_variable
    ][:100].copy()

    adata_test = adata[
        adata.obs.batch.isin(test_batch_names), 
        adata.var.highly_variable
    ][:100].copy()

    if FULL_TEST:
        # Set-up combinations to test
        model_names = ['NFinVAE', 'FinVAE']
        decoder_dists = ['nb', 'normal'] 
        
        inject_covar_in_latents = [False, True] 
        elbo_versions = ['sample', 'kl_div']
        fix_mean_priors = [True, False]
        fix_var_priors = [False, True]
        inv_covarss = [['cell_type', 'donor'], []]
        spur_covarss = [['site'], []]

        combinations = list(product(model_names, decoder_dists, inject_covar_in_latents, elbo_versions, fix_mean_priors, fix_var_priors, inv_covarss, spur_covarss))
        n_comb = len(combinations)
        #raise ValueError('Debug setup')

        for i, comb in enumerate(combinations):
            if i <= -1:
                continue

            print(f'Trying combination {i}/{n_comb}:\n\t{comb}')

            if (comb[0] == 'NFinVAE') and (comb[3] == 'kl_div' or len(comb[6]) == 0):
                continue

            if (comb[2]) and (len(comb[7]) == 0):
                continue

            debug_train_classifier(
                adata_train,
                adata_val,
                adata_test,
                model_name = comb[0], 
                decoder_dist = comb[1],  
                layer = 'raw_counts' if comb[1] == 'nb' else 'counts',
                device = 'cpu', 
                inject_covar_in_latent = comb[2], 
                elbo_version = comb[3],
                fix_mean_prior = comb[4],
                fix_var_prior = comb[5],
                inv_covars = comb[6],
                spur_covars = comb[7]
            )
    else:
        comb = ('FinVAE', 'nb', False, 'kl_div', True, False, ['cell_type', 'donor'], [])

        debug_train_classifier(
            adata_train,
            adata_val,
            adata_test,
            model_name = comb[0], 
            decoder_dist = comb[1],  
            layer = 'raw_counts' if comb[1] == 'nb' else 'counts',
            device = 'cpu', 
            inject_covar_in_latent = comb[2], 
            elbo_version = comb[3],
            fix_mean_prior = comb[4],
            fix_var_prior = comb[5],
            inv_covars = comb[6],
            spur_covars = comb[7]
        )

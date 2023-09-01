import os
import sys
import argparse

import scanpy as sc

import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import pandas as pd

# Necessary to import models as below
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.inVAE import FinVAE, NFinVAE

sc.set_figure_params(dpi=200,figsize=(6,4),fontsize=10,frameon=False)

if os.path.basename(os.getcwd()) == 'tests':
    os.chdir('../')

print(f'The current directory is: {os.getcwd()}')

### Read in all hyperparameters/settings we specify in advance
parser = argparse.ArgumentParser(description='Single-cell HP for generative models to experiment with different TC beta hyperparameters')

## Which model? -> ['f_invae', 'nf_invae'] ----------------------
parser.add_argument('--model', default='f_invae')

## Data arguments -----------------------------------------------------------------
parser.add_argument('--dataset', default='multiome')
parser.add_argument('--no_preprocessing', action='store_true')#, default=True)
parser.add_argument('--use_layer', default='counts')
parser.add_argument('--n_top_genes', type=int, default=2000)

## Experiments args ---------------------------------------------------------------
parser.add_argument('--n_experiments', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--set_seed', action='store_true')
parser.add_argument('--load_checkpoint_path', default='')
parser.add_argument('--load_hps_path', default='')#'./outputs/f_invae/multiome/1692952668_0_checkpoint_end_training.pth')
parser.add_argument('--debug', action='store_true')#, default=True)
parser.add_argument('--track_execution_time', action='store_true')

## Model HP -----------------------------------------------------------------------
parser.add_argument('--n_epochs_phase_1', type=int, default=50)
# If -1 -> random latent dim; otherwise fixed to given positive value
parser.add_argument('--latent_dim_fixed', type=int, default=-1)
# Changes distribution of the decoder: ['nb', 'normal']
parser.add_argument('--decoder_dist', default='nb')
parser.add_argument('--no_norm_const', action='store_true')#, default=True)
parser.add_argument('--norm_constant', type=float, default=-1.0)
#parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--hidden_dim', type=int, default=-1)
parser.add_argument('--n_layers', type=int, default=-1)
# Options for activation function of the model: ['lrelu' (Leaky ReLU), 'relu']
parser.add_argument('--activation', default='')
# Use batch norm?
parser.add_argument('--use_bn', action='store_true', default=True)

# Dimension of the latent noise space
parser.add_argument('--latent_dim_noise_fixed', type=int, default=-1)
# Hyperparameter for the weight of the KL-divergence
parser.add_argument('--beta', type=float, default=1.0)
# Which ELBO version? (['kl_div', 'sample'])
parser.add_argument('--elbo_version', default='sample')

# Arguments for the prior distribution for iVAE or idVAE
parser.add_argument('--fix_mean_prior', action='store_true', default=True)
parser.add_argument('--fix_var_prior', action='store_true')

# Classifier args
parser.add_argument('--n_samples_val_label_match', type=int, default=1000)
parser.add_argument('--n_epochs_opt_val', type=int, default=50)
parser.add_argument('--test_acc', action='store_true')

# Which tc beta to start with?
parser.add_argument('--start_tc_beta', type=int, default=0)
parser.add_argument('--end_tc_beta', type=int, default=10)

args = parser.parse_args()

if args.debug:
    args.n_experiments = 1
    args.n_epochs_phase_1 = 1
    args.n_samples_val_label_match = 1
    args.n_epochs_opt_val = 1

os.makedirs(f'./outputs/{args.model}/multiome/', exist_ok=True)

# Define ID:
experiment_id = int(time.time())

# Define device to use
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda = torch.cuda.is_available()

print(f'Using device: {device}')

# Set random seed
if args.set_seed:
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

## Load data and preprocess it according to data args
# Read data
if args.dataset == 'multiome':
    adata = sc.read(f'./data/multiome_gex_processed_training.h5ad')

    label_key = 'cell_type'
    batch_key = 'batch'

    inv_covar_keys = {
        'cont': [],
        'cat': ['cell_type', 'donor']
    }

    spur_covar_keys = {
        'cont': [],
        'cat': ['site']
    }

    if not args.no_preprocessing:
        if args.decoder_dist == 'normal':
            sc.pp.scale(adata, layer='counts', zero_center=True, copy=False)
        elif args.decoder_dist == 'nb':
            print('The log transformation is done by the model; raw counts are passed to it!')

    sc.pp.highly_variable_genes(adata, layer='log_norm', n_top_genes=args.n_top_genes)

    # Split data into train and test environments (with batch names)

    # for idVAE model first save site and donor info sep

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

    if args.debug:
        adata_val = adata[
        adata.obs.batch.isin(val_batch_names),
        adata.var.highly_variable
        ][:100].copy()    

        adata_test = adata[
            adata.obs.batch.isin(test_batch_names), 
            adata.var.highly_variable
        ][:100].copy()
    else:
        adata_val = adata[
            adata.obs.batch.isin(val_batch_names),
            adata.var.highly_variable
        ].copy()

        adata_test = adata[
            adata.obs.batch.isin(test_batch_names), 
            adata.var.highly_variable
        ].copy()

experiments_dt = pd.DataFrame()

if (args.load_hps_path == '') and (args.load_checkpoint_path == ''):
    tc_list = list(range(args.start_tc_beta, args.end_tc_beta + 1, 2)) if not args.debug else [1]
else:
    if args.load_hps_path != '':
        checkpoint_model = torch.load(args.load_hps_path, map_location=device)
    else:
        checkpoint_model = torch.load(args.load_checkpoint_path, map_location=device)

    hp_dict = checkpoint_model['hyperparameters']

    tc_list = [hp_dict['tc_beta']]

print(f'Trying out the following tc beta HPs: {tc_list}')

print('>>> Results:')

for tc_beta in tc_list:
    for exp_id in range(args.n_experiments):
        print(f'\nRunning experiment {exp_id}/{args.n_experiments-1} ...')
        
        if (args.load_hps_path == '') and (args.load_checkpoint_path == ''):
            # Init model with random hyperparameters
            lr_train = 10**np.random.uniform(-5, -2)

            weight_decay = 10**np.random.uniform(-6, -3)

            # Generate combinations of hyperparameters
            if args.latent_dim_fixed < 0:
                latent_dim = np.random.choice([10, 20, 30, 40, 50])
            else:
                latent_dim = args.latent_dim_fixed
            
            if args.hidden_dim < 0:
                hidden_dim_x = int(2**np.random.uniform(6, 10))
            else:
                hidden_dim_x = args.hidden_dim

            if args.n_layers < 0:
                n_layers_x = np.random.randint(2, 5)
            else:
                n_layers_x  = args.n_layers

            if args.activation == '':
                activation = np.random.choice(['lrelu', 'relu'])
            else:
                activation = args.activation

            # Think of normalizing constant as a HP for prior importance, i.e. upweighting the Score Matching part
            # and then dividing by it to keep in reasonable range
            if args.no_norm_const:
                normalize_constant = 1
            elif args.norm_constant <= 0:
                normalize_constant = 10**np.random.uniform(1, 5)
            else:
                normalize_constant = args.norm_constant
            
            if args.model == 'nf_invae':
                if args.latent_dim_noise_fixed < 0:
                    latent_dim_noise = np.random.randint(1, int(latent_dim/5))
                else:
                    latent_dim_noise = args.latent_dim_noise_fixed

                # Weight for regularized score matching (50% of the times = 0)
                sm_non_zero_weight = 10**np.random.uniform(-4, -2)
                score_matching_weight = np.random.choice([0, sm_non_zero_weight])

                output_dim_prior_nn = np.random.randint(4*latent_dim, 6*latent_dim) #np.random.choice([40, 50, 60])
                hidden_dim_prior = output_dim_prior_nn + np.random.randint(1, 2*latent_dim) #np.random.choice([80, 100, 120])
                n_layers_prior = np.random.choice([2, 3])
                
                # Init model with random hyperparameters
                hp_dict = dict(
                    latent_dim_inv = latent_dim - latent_dim_noise, 
                    latent_dim_spur = latent_dim_noise,
                    n_layers = n_layers_x,
                    hidden_dim = hidden_dim_x,
                    activation = activation, 
                    device = device, 
                    normalize_constant = normalize_constant,
                    fix_mean_spur_prior = args.fix_mean_prior,
                    fix_var_spur_prior = args.fix_var_prior,
                    decoder_dist = args.decoder_dist,
                    batch_norm = args.use_bn,
                    tc_beta = tc_beta,
                    dropout_rate = 0.1,
                    batch_size = args.batch_size,
                    reg_sm = score_matching_weight,
                    output_dim_prior_nn = output_dim_prior_nn,
                    hidden_dim_prior =  hidden_dim_prior,
                    n_layers_prior =  n_layers_prior,
                    inject_covar_in_latent = False,
                )

                model = NFinVAE(
                    adata_train,
                    layer = args.use_layer,
                    inv_covar_keys = inv_covar_keys,
                    spur_covar_keys = spur_covar_keys,
                    **hp_dict
                )
            elif args.model == 'f_invae':
                if args.latent_dim_noise_fixed < 0:
                    latent_dim_noise = np.random.randint(1, int(latent_dim/5))
                else:
                    latent_dim_noise = args.latent_dim_noise_fixed

                hp_dict = dict(
                    latent_dim_inv = latent_dim - latent_dim_noise, 
                    latent_dim_spur = latent_dim_noise,
                    n_layers = n_layers_x,
                    hidden_dim = hidden_dim_x,
                    activation = activation, 
                    device = device,  
                    fix_mean_spur_prior = args.fix_mean_prior,
                    fix_var_spur_prior = args.fix_var_prior,
                    decoder_dist = args.decoder_dist,
                    batch_norm = args.use_bn,
                    tc_beta = tc_beta,
                    kl_rate = args.beta,
                    dropout_rate = 0.1,
                    batch_size = args.batch_size,
                    inject_covar_in_latent = False,
                    elbo_version = 'sample',
                )

                model = FinVAE(
                    adata_train,
                    layer = args.use_layer,
                    inv_covar_keys = inv_covar_keys,
                    spur_covar_keys = spur_covar_keys,
                    **hp_dict
                )
        elif args.load_checkpoint_path != '':
            if args.model == 'nf_invae':
                checkpoint_model = torch.load(args.load_checkpoint_path, map_location=device)

                model = NFinVAE(
                    adata_train,
                    layer = args.use_layer,
                    inv_covar_keys = inv_covar_keys,
                    spur_covar_keys = spur_covar_keys,
                    **checkpoint_model['hyperparameters']
                )

                model.module.load_state_dict(checkpoint_model['model_state_dict'])

                hp_dict = checkpoint_model['hyperparameters']
            elif args.model == 'f_invae':
                checkpoint_model = torch.load(args.load_checkpoint_path, map_location=device)

                model = FinVAE(
                    adata_train,
                    layer = args.use_layer,
                    inv_covar_keys = inv_covar_keys,
                    spur_covar_keys = spur_covar_keys,
                    **checkpoint_model['hyperparameters']
                )

                model.module.load_state_dict(checkpoint_model['model_state_dict'])

                hp_dict = checkpoint_model['hyperparameters']
            else:
                raise ValueError(f'{args.model} is not a valid model!')
        elif args.load_hps_path != '':
            checkpoint_model = torch.load(args.load_hps_path, map_location=device)
            hp_dict = checkpoint_model['hyperparameters']

            if args.model == 'nf_invae':
                model = NFinVAE(
                    adata_train,
                    layer = args.use_layer,
                    inv_covar_keys = inv_covar_keys,
                    spur_covar_keys = spur_covar_keys,
                    **hp_dict
                )
            elif args.model == 'f_invae':
                model = FinVAE(
                    adata_train,
                    layer = args.use_layer,
                    inv_covar_keys = inv_covar_keys,
                    spur_covar_keys = spur_covar_keys,
                    **hp_dict
                )
            else:
                raise ValueError(f'{args.model} is not a valid model!')
             
        if (args.load_checkpoint_path != '') or (args.load_hps_path != ''):
            lr_train = hp_dict['lr_train']
            weight_decay = hp_dict['weight_decay']
            args.n_epochs_phase_1 = hp_dict['n_epochs_phase_1']

            args.n_epochs_opt_val = hp_dict['n_epochs_opt_val']
            args.n_samples_val_label_match = hp_dict['n_samples_val_label_match']

        n_epochs = args.n_epochs_phase_1

        model.train(
            n_epochs = n_epochs,
            lr_train = lr_train,
            weight_decay = weight_decay,
            use_lr_schedule = True,
            lr_scheduler_patience = 30,
        )
        
        ## Save a figure of the latent space colored by batch or cell type
        # For train environments
        latent_full = model.get_latent_representation(latent_type='full')
        latent_inv = model.get_latent_representation(latent_type='invariant')
        latent_spur = model.get_latent_representation(latent_type='spur')  

        adata_train.obsm[f'X_{args.model}_full'] = latent_full
        adata_train.obsm[f'X_{args.model}_inv'] = latent_inv
        adata_train.obsm[f'X_{args.model}_spur'] = latent_spur

        # Full latent space
        if not np.isnan(latent_full).any():
            sc.pp.neighbors(adata_train, use_rep=f'X_{args.model}_full')
            sc.tl.umap(adata_train)
            fig = sc.pl.umap(adata_train, color=['batch', 'cell_type'], return_fig=True, show=False)
            fig.savefig(f'./outputs/{args.model}/multiome/{experiment_id}_{exp_id}_{args.decoder_dist}_train_latent_full_tc_beta_{tc_beta}.png', bbox_inches='tight')
            plt.close(fig)

        # Invariant latent space
        if not np.isnan(latent_inv).any():
            sc.pp.neighbors(adata_train, use_rep=f'X_{args.model}_inv')
            sc.tl.umap(adata_train)
            fig = sc.pl.umap(adata_train, color=['batch', 'cell_type'], return_fig=True, show=False)
            fig.savefig(f'./outputs/{args.model}/multiome/{experiment_id}_{exp_id}_train_latent_inv_tc_beta_{tc_beta}.png', bbox_inches='tight')
            plt.close(fig)

        # Spurious latent space
        if not np.isnan(latent_spur).any():
            sc.pp.neighbors(adata_train, use_rep=f'X_{args.model}_spur')
            sc.tl.umap(adata_train)
            fig = sc.pl.umap(adata_train, color=['batch', 'cell_type'], return_fig=True, show=False)
            fig.savefig(f'./outputs/{args.model}/multiome/{experiment_id}_{exp_id}_train_latent_spur_tc_beta_{tc_beta}.png', bbox_inches='tight')
            plt.close(fig)

        # Train classifier
        train_loss = model.get_negative_elbo()

        if not np.isnan(train_loss):
            model.train_classifier(
                adata_val,
                batch_key = 'batch',
                label_key = 'cell_type',
                n_epochs_train_class = 500,
                n_epochs_opt_val = args.n_epochs_opt_val,
                nr_samples = args.n_samples_val_label_match,
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

            # Calculate accuracies
            train_acc = (pred_train == adata_train.obs[label_key]).sum() / adata_train.n_obs
            val_acc = (pred_val == adata_val.obs[label_key]).sum() / adata_val.n_obs

            if args.test_acc:
                pred_test = model.predict(adata_test, dataset_type='test')
                test_acc = (pred_test == adata_test.obs[label_key]).sum() / adata_test.n_obs

        # TODO: test calculate accuracies and implement MCC?

        ## Save HP in dataframe for investigating best choices
        hp_dict['experiment_id'] = experiment_id
        hp_dict['exp_id'] = exp_id

        hp_dict['lr_train'] = lr_train
        hp_dict['weight_decay'] = weight_decay

        hp_dict['n_epochs_phase_1'] = args.n_epochs_phase_1

        hp_dict['loss'] = train_loss
        hp_dict['val_loss'] = model.get_negative_elbo(adata_val)

        #TODO: save MCC
        hp_dict['n_epochs_opt_val'] = args.n_epochs_opt_val
        hp_dict['n_samples_val_label_match'] = args.n_samples_val_label_match

        hp_dict['train_acc'] = train_acc if not np.isnan(train_loss) else 0
        hp_dict['val_acc'] = val_acc if not np.isnan(train_loss) else 0
        
        if args.test_acc:
            hp_dict['test_acc'] = test_acc if not np.isnan(train_loss) else 0
        
        save_path = f'./outputs/{args.model}/multiome/{experiment_id}_{exp_id}_tc_beta_{tc_beta}_checkpoint_end_training.pth'
                        
        torch.save({
            'epoch': args.n_epochs_phase_1,
            'model_state_dict': model.module.state_dict(),
            'hyperparameters': hp_dict
        }, save_path)

        output_dt = pd.DataFrame([hp_dict])
        experiments_dt = pd.concat([experiments_dt, output_dt], ignore_index=True)
        experiments_dt.to_csv(f'./outputs/{args.model}/multiome/{experiment_id}_nexp_{args.n_experiments}_{args.n_top_genes}_genes.csv')

print("The experiment run is done!")
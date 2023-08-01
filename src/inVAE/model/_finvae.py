from typing import List, Literal, Optional, Dict, Union
from itertools import chain

from anndata import AnnData

from ._abstract_invae import inVAE
from ..module import FinVAEmodule

class FinVAE(inVAE):

    def __init__(
        self, 
        adata: AnnData,
        layer: Optional[str] = None,
        inv_covar_keys: Dict[str, List[str]] = None,
        spur_covar_keys: Dict[str, List[str]] = None,
        latent_dim_inv: int = 9, 
        latent_dim_spur: int = 1,
        n_layers: int = 2, 
        hidden_dim: int = 128,
        activation: Literal['relu', 'lrelu'] = 'relu', 
        slope: float = 0.1, # only needed when activation is Leaky ReLU ('lrelu')
        device: Literal['cpu', 'cuda'] = 'cpu', 
        fix_mean_prior: bool = True,
        fix_var_prior: bool = False,
        decoder_dist: Literal['normal', 'nb'] = 'nb',
        batch_norm: bool = True,
        dropout_rate: float = 0.1,
        kl_rate: float = 1.0,
        batch_size: int = 256,
        elbo_version: Literal['kl_div', 'sample'] = 'sample',
        inject_covar_in_latent: bool = False,
        **kwargs,
    ):
        super().__init__()

        if adata is None:
            raise ValueError('Adata is None, check if you passed the data to the model!')
        
        self.adata = adata

        # Dict to save latents used in prediction later
        self.saved_latent = {}

        if layer is None:
            print('Layer is None, check if you want to specify the layer of adata!')
        
        # Assign all keys
        self.layer = layer

        self.inv_covar_keys = inv_covar_keys
        self.spur_covar_keys = spur_covar_keys

        # Number of genes -> Rename?
        self.data_dim = adata.shape[1]

        self.batch_size = batch_size

        self.device = device

        # Latent dimensions
        if inject_covar_in_latent:
            print('Injecting spurious covariates in the latent space! The latent_dim_spur are ignored and set to zero!')
            latent_dim_spur = 0

        self.latent_dim = latent_dim_inv + latent_dim_spur
        self.latent_dim_spur = latent_dim_spur
        self.latent_dim_inv = self.latent_dim - latent_dim_spur

        # Set-up data
        
        # Check if the layer of the data contains the right data:
        self.check_data(adata, layer, decoder_dist)
        
        self.dict_encoders, self.data_loading_encoders, self.data_loader, self.transformed_data = self.setup_adata(adata, inv_covar_keys, spur_covar_keys, device, batch_size)
        
        # Flatten list of covariates
        self.list_spur_covar = [value for key, value in self.spur_covar_keys.items() if key in ['cat', 'cont']]
        self.list_spur_covar = list(chain.from_iterable(self.list_spur_covar))

        self.list_inv_covar = [value for key, value in self.inv_covar_keys.items() if key in ['cat', 'cont']]
        self.list_inv_covar = list(chain.from_iterable(self.list_inv_covar))

        # Need to setup data first; then add dim of covariates:
        self.spur_covar_dim = sum([self.transformed_data.obs[covar].shape[1] for covar in self.list_spur_covar])
        self.inv_covar_dim = sum([self.transformed_data.obs[covar].shape[1] for covar in self.list_inv_covar])

        if inject_covar_in_latent and self.spur_covar_dim == 0:
            raise ValueError('The spurious covariates are None, can not inject them into the latent space.' + 
                             'Check if you specified spurious covariates or set "inject_covar_in_latent" to False!')

        self.module = FinVAEmodule(
            latent_dim = self.latent_dim, 
            latent_dim_spur = latent_dim_spur,
            n_layers = n_layers, 
            hidden_dim = hidden_dim,
            activation = activation, 
            slope = slope,
            device = device,
            fix_mean_prior = fix_mean_prior,
            fix_var_prior =  fix_var_prior,
            decoder_dist = decoder_dist,
            batch_norm = batch_norm,
            dropout_rate=dropout_rate,
            kl_rate = kl_rate,
            batch_size = batch_size,
            elbo_version = elbo_version,
            data_dim = self.data_dim,
            inv_covar_dim = self.inv_covar_dim,
            spur_covar_dim = self.spur_covar_dim,
            inject_covar_in_latent = inject_covar_in_latent
        )
    
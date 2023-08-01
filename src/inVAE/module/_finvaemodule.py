from typing import Literal

import torch
from torch import distributions as dist
from torch import nn

from ..utils import Normal, MLP, log_nb_positive, weights_init

class FinVAEmodule(nn.Module):

    def __init__(
        self,
        latent_dim: int = 10, 
        latent_dim_spur: int = 1,
        n_layers: int = 2, 
        hidden_dim: int = 128,
        activation: Literal['relu', 'lrelu'] = 'relu', 
        slope: float = 0.1, # only needed when activation is Leaky ReLU ('lrelu')
        device: Literal['cpu', 'cuda'] = 'cpu', 
        normalize_constant: float = 1.0,
        fix_mean_prior: bool = True,
        fix_var_prior: bool = False,
        decoder_dist: Literal['normal', 'nb'] = 'nb',
        batch_norm: bool = True,
        dropout_rate: float = 0.0,
        kl_rate: float = 1.0,
        batch_size: int = 256,
        elbo_version: Literal['kl_div', 'sample'] = 'sample',
        data_dim: int = None,
        inv_covar_dim: int = None,
        spur_covar_dim: int = None,
        inject_covar_in_latent: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.latent_dim_spur = latent_dim_spur
        self.latent_dim_inv = latent_dim - latent_dim_spur
        
        # Params for MLP
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.normalize_constant = normalize_constant

        # General model params
        self.fix_mean_prior = fix_mean_prior
        self.fix_var_prior = fix_var_prior
        self.decoder_dist = decoder_dist
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.inject_covar_in_latent = inject_covar_in_latent

        # Dimension for decoder input
        if inject_covar_in_latent:
            # As default only inject the covariates of the spurious prior
            input_dim_decoder = latent_dim + spur_covar_dim
        else:
            input_dim_decoder = latent_dim

        # Training HPs
        self.beta = kl_rate
        self.elbo_version = elbo_version
        self.batch_size = batch_size

        self.device = device

        # Data related values

        # Number of genes
        self.data_dim = data_dim

        # Need to setup data first; then add dim of covariates:
        self.inv_covar_dim = inv_covar_dim
        self.spur_covar_dim = spur_covar_dim

        ## Two priors:
        # One prior for invariant latent space conditioned on invariant covariates
        self.prior_dist_inv = Normal(device=device)

        # One prior for spurious latent space conditioned on spurious covariates
        if not inject_covar_in_latent:
            self.prior_dist_spur = Normal(device=device)

        if decoder_dist == 'normal':
            self.decoder_dist_fct = Normal(device=device)
        
        self.encoder_dist = Normal(device=device)

        # If invariant or spurious covariates are None: use N(0,1) prior
        if self.inv_covar_dim == 0:
            print('The covariates for the invariant prior are None. Defaulting to N(0,1) for that prior.')

        # If we inject covariates in latent do not use spurious prior at all
        if (self.spur_covar_dim == 0) and (not inject_covar_in_latent):
            print('The covariates for the spurious prior are None. Defaulting to N(0,1) for that prior.')
        
        ## prior_params
        # Logic: both priors are modeled as a multivariate normal distribution with diagonal covariance (i.e. independent components) for this model
        if self.fix_mean_prior and self.fix_var_prior:
            self.prior_mean_inv = torch.zeros(1).to(device)
            self.logl_inv = torch.ones(1).to(device)
            
            if not inject_covar_in_latent:
                self.logl_spur = torch.ones(1).to(device)
                self.prior_mean_spur = torch.zeros(1).to(device)
            else:
                self.logl_spur = None
                self.prior_mean_spur = None
        elif self.fix_mean_prior:
            self.prior_mean_inv = torch.zeros(1).to(device)

            if not inject_covar_in_latent:
                self.prior_mean_spur = torch.zeros(1).to(device)
            else:
                self.prior_mean_spur = None

            if self.inv_covar_dim != 0:
                self.logl_inv = MLP(
                    self.inv_covar_dim, 
                    self.latent_dim_inv, 
                    hidden_dim, 
                    n_layers, 
                    activation=activation, 
                    slope=slope, 
                    device=device,
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate
                ).to(device)
            else:
                self.logl_inv = torch.ones(1).to(device)

            if self.spur_covar_dim != 0 and not inject_covar_in_latent:
                self.logl_spur = MLP(
                    self.spur_covar_dim, 
                    self.latent_dim_spur, 
                    hidden_dim, n_layers, 
                    activation=activation, 
                    slope=slope, 
                    device=device, 
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate
                ).to(device)
            elif not inject_covar_in_latent:
                self.logl_spur = torch.ones(1).to(device)
            else:
                self.logl_spur = None
        elif self.fix_var_prior:
            if self.inv_covar_dim != 0:
                self.prior_mean_inv = MLP(
                    self.inv_covar_dim, 
                    self.latent_dim_inv, 
                    hidden_dim, 
                    n_layers, 
                    activation=activation, 
                    slope=slope, 
                    device=device,
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate
                ).to(device)
            else:
                self.prior_mean_inv = torch.zeros(1).to(device)

            if self.spur_covar_dim != 0 and not inject_covar_in_latent:
                self.prior_mean_spur = MLP(
                    self.spur_covar_dim, 
                    self.latent_dim_spur, 
                    hidden_dim, 
                    n_layers, 
                    activation=activation, 
                    slope=slope, 
                    device=device, 
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate
                ).to(device)
            elif not inject_covar_in_latent:
                self.prior_mean_spur = torch.zeros(1).to(device)
            else:
                self.prior_mean_spur = None

            self.logl_inv = torch.ones(1).to(device)
            if not inject_covar_in_latent:
                self.logl_spur = torch.ones(1).to(device)
            else:
                self.logl_spur = None
        elif (not self.fix_mean_prior) and (not self.fix_var_prior):
            # use one layer less for shared NN such that the whole prior NN has n_layers (end with act for first part)
            
            ## Invariant latent prior
            if self.inv_covar_dim != 0:
                self.prior_nn_inv = MLP(
                    self.inv_covar_dim, 
                    hidden_dim, 
                    hidden_dim, 
                    n_layers-1, 
                    activation=activation, 
                    slope=slope, 
                    device=device, 
                    end_with_act=True, 
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate
                ).to(device)

                self.prior_mean_inv = nn.Linear(hidden_dim, self.latent_dim_inv, device=device)
                self.logl_inv = nn.Linear(hidden_dim, self.latent_dim_inv, device=device)
            else:
                self.prior_mean_inv = torch.zeros(1).to(device)
                self.logl_inv = torch.ones(1).to(device)

            ## Noise latent prior
            if self.spur_covar_dim != 0 and not inject_covar_in_latent:
                self.prior_nn_spur = MLP(
                    self.spur_covar_dim, 
                    hidden_dim, 
                    hidden_dim, 
                    n_layers-1, 
                    activation=activation, 
                    slope=slope, 
                    device=device, 
                    end_with_act=True, 
                    batch_norm=batch_norm,
                    dropout_rate=dropout_rate
                ).to(device)

                self.prior_mean_spur = nn.Linear(hidden_dim, self.latent_dim_spur, device=device)
                self.logl_spur = nn.Linear(hidden_dim, self.latent_dim_spur, device=device)
            elif not inject_covar_in_latent:
                self.prior_mean_spur = torch.zeros(1).to(device)
                self.logl_spur = torch.ones(1).to(device)
            else:
                self.prior_mean_spur = None
                self.logl_spur = None

        # decoder params
        if self.decoder_dist == 'normal':
            self.f = MLP(
                input_dim_decoder,
                self.data_dim, 
                hidden_dim, 
                n_layers, 
                activation=activation,
                slope=slope,
                device=device,
                batch_norm=batch_norm,
                dropout_rate=dropout_rate
            ).to(device)
            
            self.decoder_var = .01 * torch.ones(1).to(device)
        elif self.decoder_dist == 'nb':
            ## Decoder for NB distribution consists of:
            #  mean -> (raw mean values + Softmax) * library_size
            #  theta -> parameter shared across cells (so variability per gene)

            self.decoder_raw_mean = MLP(
                input_dim=input_dim_decoder, 
                output_dim=hidden_dim, 
                hidden_dim=hidden_dim, 
                n_layers=n_layers, 
                activation=activation, 
                device=device,
                batch_norm=self.batch_norm,
                dropout_rate=dropout_rate
            ).to(self.device)

            self.decoder_freq  = nn.Sequential(
                nn.Linear(hidden_dim, self.data_dim),
                nn.Softmax(dim=-1),
            ).to(device)
            
            # Raw theta has pos&neg values -> call exp(.) later
            self.decoder_raw_theta = nn.Parameter(torch.randn(self.data_dim, device = self.device))
        else:
            raise ValueError(f'The argument of decoder_dist "{decoder_dist}" is not one of ["normal", "nb"]!')

        # encoder params
        self.shared_encoder = MLP(
            self.data_dim + self.inv_covar_dim + self.spur_covar_dim, 
            hidden_dim, 
            hidden_dim, 
            n_layers-1, 
            activation=activation, 
            slope=slope,
            device=device,
            batch_norm=self.batch_norm,
            dropout_rate=dropout_rate,
            end_with_act=True
        ).to(device)

        self.g = nn.Linear(hidden_dim, self.latent_dim, device=self.device)

        self.logv = nn.Linear(hidden_dim, self.latent_dim, device=self.device)

        self.apply(weights_init)

    def encode(self, x, inv_covar, spur_covar):
        if self.decoder_dist == 'normal':
            # Assumes x is scaled to zero mean and one var
            if (inv_covar is None) and (spur_covar is None):
                xde = x
            elif inv_covar is None:
                xde = torch.cat((x, spur_covar.view(-1, self.spur_covar_dim)), 1)
            elif spur_covar is None:
                xde = torch.cat((x, inv_covar.view(-1, self.inv_covar_dim)), 1)
            else: 
                xde = torch.cat((x, inv_covar.view(-1, self.inv_covar_dim), spur_covar.view(-1, self.spur_covar_dim)), 1)
        elif self.decoder_dist == 'nb':
            # Assumes x is raw count data
            x_log = torch.log(x+1)

            if (inv_covar is None) and (spur_covar is None):
                xde = x_log
            elif inv_covar is None:
                xde = torch.cat((x_log, spur_covar.view(-1, self.spur_covar_dim)), 1)
            elif spur_covar is None:
                xde = torch.cat((x_log, inv_covar.view(-1, self.inv_covar_dim)), 1)
            else: 
                xde = torch.cat((x_log, inv_covar.view(-1, self.inv_covar_dim), spur_covar.view(-1, self.spur_covar_dim)), 1)

        # Shared encoder
        h = self.shared_encoder(xde)

        g = self.g(h)
        logv = self.logv(h)
        
        return g, (logv.exp() + 1e-4)

    def decode(self, z, inv_covar=None, spur_covar=None):
        if (inv_covar is None) and (spur_covar is None):
            zde = z
        elif spur_covar is not None:
            zde = torch.cat((z, spur_covar.view(-1, self.spur_covar_dim)), 1)
        
        # Disable other options for now -> Use only covariates of spurious prior, when "inject_covar_in_latent" is True
        #elif spur_covar is None:
        #    zde = torch.cat((z, inv_covar.view(-1, self.inv_covar_dim)), 1)
        #else: 
        #    zde = torch.cat((z, inv_covar.view(-1, self.inv_covar_dim), spur_covar.view(-1, self.spur_covar_dim)), 1)

        if self.decoder_dist == 'normal':
            f = self.f(zde)
            return f, self.decoder_var
        elif self.decoder_dist == 'nb':
            decoder_raw_mean = self.decoder_raw_mean(zde)
            decoder_mean = self.decoder_freq(decoder_raw_mean)
            
            return decoder_mean, torch.exp(self.decoder_raw_theta)

    def prior(self, inv_covar, spur_covar):
        if self.fix_mean_prior and self.fix_var_prior:
            return (self.prior_mean_inv, self.prior_mean_spur), (self.logl_inv, self.logl_spur)
        elif self.fix_mean_prior:
            logl_inv = (self.logl_inv(inv_covar).exp() + 1e-4) if (inv_covar is not None) else self.logl_inv

            if not self.inject_covar_in_latent:
                if (spur_covar is not None):
                    logl_spur = (self.logl_spur(spur_covar).exp() + 1e-4)   
                else:
                    logl_spur = self.logl_spur
            else:
                logl_spur = None

            return (self.prior_mean_inv, self.prior_mean_spur), (logl_inv, logl_spur)
        elif self.fix_var_prior:
            prior_mean_inv = self.prior_mean_inv(inv_covar) if (inv_covar is not None) else self.prior_mean_inv

            if (spur_covar is not None) and (not self.inject_covar_in_latent):
                prior_mean_spur = self.prior_mean_spur(spur_covar) 
            else:
                prior_mean_spur = self.prior_mean_spur

            return (prior_mean_inv, prior_mean_spur), (self.logl_inv, self.logl_spur)
        elif (not self.fix_mean_prior) and (not self.fix_var_prior):
            if inv_covar is not None:
                prior_shared_inv = self.prior_nn_inv(inv_covar)

                prior_mean_inv = self.prior_mean_inv(prior_shared_inv)
                logl_inv = self.logl_inv(prior_shared_inv).exp() + 1e-4
            else:
                prior_mean_inv = self.prior_mean_inv
                logl_inv = self.logl_inv

            if (spur_covar is not None) and (not self.inject_covar_in_latent):
                prior_shared_spur = self.prior_nn_spur(spur_covar)

                prior_mean_spur = self.prior_mean_spur(prior_shared_spur)
                logl_spur = self.logl_spur(prior_shared_spur).exp() + 1e-4
            else:
                prior_mean_spur = self.prior_mean_spur
                logl_spur = self.logl_spur

            return (prior_mean_inv, prior_mean_spur), (logl_inv, logl_spur)

    def sample_latent(self, x, inv_covar, spur_covar):
        encoder_params = self.encode(x, inv_covar, spur_covar)
        z = self.encoder_dist.sample(*encoder_params)
        return z
    
    def get_log_decoder_density(self, x, z, inv_covar=None, spur_covar=None):
        if spur_covar is None:
            zde = z
        else:
            zde = torch.cat((z, spur_covar.view(-1, self.spur_covar_dim)), 1)

        if self.decoder_dist == 'normal':
            decoder_params = self.decode(zde)
            log_px_z = self.decoder_dist_fct.log_pdf(x, *decoder_params)
        elif self.decoder_dist == 'nb':
            # decoder_mean are only the freq
            decoder_mean, decoder_theta = self.decode(zde)

            # multiply with library size to get "correct" means
            library = x.sum(dim=-1, keepdim=True)
            decoder_mean = decoder_mean * library

            log_px_z = log_nb_positive(x, decoder_mean, decoder_theta)

        return log_px_z

    def forward(self, x, inv_covar, spur_covar):
        prior_params_mean, prior_params_var = self.prior(inv_covar, spur_covar)
        encoder_params = self.encode(x, inv_covar, spur_covar)
        z = self.encoder_dist.sample(*encoder_params)
        
        if self.decoder_dist == 'normal':
            if self.inject_covar_in_latent:
                decoder_params = self.decode(z, inv_covar, spur_covar)
            else:
                decoder_params = self.decode(z)

            return decoder_params, encoder_params, z, prior_params_mean, prior_params_var
        elif self.decoder_dist == 'nb':
            if self.inject_covar_in_latent:
                decoder_mean, decoder_theta = self.decode(z, inv_covar, spur_covar)
            else:
                # decoder_mean are only the freq
                decoder_mean, decoder_theta = self.decode(z)

            # multiply with library size to get "correct" means
            library = x.sum(dim=-1, keepdim=True)
            decoder_mean = decoder_mean * library

            return decoder_mean, decoder_theta, encoder_params, z, prior_params_mean, prior_params_var
    
    def elbo(self, x, inv_covar, spur_covar):
        if self.decoder_dist == 'normal':
            decoder_params, (g, v), z, prior_params_mean, prior_params_var = self.forward(x, inv_covar, spur_covar)
            log_px_z = self.decoder_dist_fct.log_pdf(x, *decoder_params)
        elif self.decoder_dist == 'nb':
            decoder_mean, decoder_theta, (g, v), z, prior_params_mean, prior_params_var = self.forward(x, inv_covar, spur_covar)
            log_px_z = log_nb_positive(x, decoder_mean, decoder_theta)
            
        if self.elbo_version == 'sample':
            log_qz_xde = self.encoder_dist.log_pdf(z, g, v)

            # Prior parts
            # invariant latent space
            log_pzi_d = self.prior_dist_inv.log_pdf(z[:, :self.latent_dim_inv], prior_params_mean[0], prior_params_var[0])
            
            # Spurious latent space
            if not self.inject_covar_in_latent:
                log_pzs_e = self.prior_dist_spur.log_pdf(z[:, self.latent_dim_inv:], prior_params_mean[1], prior_params_var[1])
            else:
                log_pzs_e = 0

            return (log_px_z + self.beta * (log_pzi_d + log_pzs_e - log_qz_xde)).mean(), z
        elif self.elbo_version == 'kl_div':
            if torch.any(torch.isnan(g)) or torch.any(torch.isnan(v)):
                return torch.tensor(float('nan')), z
            
            # Encoder distribution
            qz_xde = dist.Normal(g, v.sqrt())

            #raise ValueError('Debug elbo version kl_div...')

            # Prior distribution (independent components -> diagonal cov)
            # Ugly fix for size match; e.g. inv var size [128, 9] and spur var size [1]
            prior_params_mean_list = [prior_params_mean[0], prior_params_mean[1]]
            prior_params_var_list = [prior_params_var[0], prior_params_var[1]]

            for i, tens in enumerate(prior_params_mean_list):
                if tens is not None and tens.shape[0] == 1:
                    tmp_dim = self.latent_dim_inv if i == 0 else self.latent_dim_spur
                    prior_params_mean_list[i] = tens.expand(z.shape[0], tmp_dim)

            prior_params_mean_list = [p for p in prior_params_mean_list if p is not None]
            
            for i, tens in enumerate(prior_params_var_list):
                if tens is not None and tens.shape[0] == 1:
                    tmp_dim = self.latent_dim_inv if i == 0 else self.latent_dim_spur
                    prior_params_var_list[i] = tens.expand(z.shape[0], tmp_dim)

            prior_params_var_list = [p for p in prior_params_var_list if p is not None]

            full_prior_mean = torch.cat(prior_params_mean_list, dim=1) if not self.fix_mean_prior else prior_params_mean[0]
            full_prior_var = torch.cat(prior_params_var_list, dim=1) if not self.fix_var_prior else prior_params_var[0]

            pz_de = dist.Normal(full_prior_mean, full_prior_var.sqrt())

            # KL-divergence between encoder and prior
            kl_div = dist.kl_divergence(qz_xde, pz_de).sum(dim=1)

            # loss
            return (log_px_z - self.beta * kl_div).mean(), z

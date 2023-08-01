from typing import Literal

import torch
from torch import nn
from torch.autograd import grad

from ..utils import log_normal, MLP, log_nb_positive, weights_init

class NFinVAEmodule(nn.Module):

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
        fix_mean_spur_prior: bool = True,
        fix_var_spur_prior: bool = False,
        decoder_dist: Literal['normal', 'nb'] = 'nb',
        batch_norm: bool = True,
        dropout_rate: float = 0.0,
        batch_size: int = 256,
        data_dim: int = None,
        inv_covar_dim: int = None,
        spur_covar_dim: int = None,
        reg_sm: float = 0,
        output_dim_prior_nn: int = None,
        hidden_dim_prior: int = None,
        n_layers_prior: int = None,
        inject_covar_in_latent: bool = False,
        **kwargs
    ):
        super().__init__()

        # Latent dimensions
        self.latent_dim = latent_dim
        self.latent_dim_inv = latent_dim - latent_dim_spur
        self.latent_dim_spur = latent_dim_spur

        # Data dimensions
        self.data_dim = data_dim
        self.inv_covar_dim = inv_covar_dim
        self.spur_covar_dim = spur_covar_dim

        # Params for MLP
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        # General model params
        self.fix_mean_spur_prior = fix_mean_spur_prior
        self.fix_var_spur_prior = fix_var_spur_prior
        self.decoder_dist = decoder_dist
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate
        self.reg_sm = reg_sm
        self.normalize_constant = normalize_constant

        self.inject_covar_in_latent = inject_covar_in_latent

        # Dimension for decoder input
        if inject_covar_in_latent:
            # As default only inject the covariates of the spurious prior
            input_dim_decoder = latent_dim + spur_covar_dim
        else:
            input_dim_decoder = latent_dim

        # Training HPs
        self.batch_size = batch_size

        self.device = device

        # Prior MLPS
        output_dim_prior_nn = output_dim_prior_nn if (output_dim_prior_nn is not None) else (2*latent_dim)
        hidden_dim_prior = hidden_dim_prior if (hidden_dim_prior is not None) else hidden_dim
        n_layers_prior = n_layers_prior if (n_layers_prior is not None) else n_layers

        self.output_dim_prior_nn = output_dim_prior_nn
        self.hidden_dim_prior = hidden_dim_prior
        self.n_layers_prior = n_layers_prior

        # x_y_e_encoder
        self.x_y_e_encoder = MLP(
            input_dim = data_dim + inv_covar_dim + spur_covar_dim, 
            output_dim = hidden_dim,
            hidden_dim = hidden_dim, 
            n_layers = n_layers, 
            activation = activation, 
            device = device, 
            end_with_act = True,
            batch_norm = batch_norm,
            dropout_rate=dropout_rate
        ).to(self.device)

        # Latent mean and log variance
        self.latent_mean = nn.Linear(hidden_dim, self.latent_dim)
        self.latent_log_var = nn.Linear(hidden_dim, self.latent_dim)

        # decoder
        if decoder_dist == 'normal':
            self.decoder_mean = MLP(
                input_dim=input_dim_decoder, 
                output_dim=data_dim, 
                hidden_dim=hidden_dim, 
                n_layers=n_layers, 
                activation=activation, 
                device=device,
                batch_norm=batch_norm,
                dropout_rate=dropout_rate
            ).to(self.device)

            self.decoder_var = torch.mul(0.01, torch.ones(data_dim, device = self.device))
        elif decoder_dist == 'nb':
            ## Decoder for NB distribution consists of:
            #  mean -> (raw mean values + Softmax) * library_size
            #  theta -> parameter shared across cells

            self.decoder_raw_mean = MLP(
                input_dim=input_dim_decoder, 
                output_dim=hidden_dim, 
                hidden_dim=hidden_dim, 
                n_layers=n_layers, 
                activation=activation, 
                device=device,
                batch_norm=batch_norm,
                dropout_rate=dropout_rate
            ).to(self.device)

            self.decoder_freq  = nn.Sequential(
                nn.Linear(hidden_dim, data_dim),
                nn.Softmax(dim=-1),
            ).to(self.device)
            
            # Raw theta has pos&neg values -> call exp(.) later
            self.decoder_raw_theta = nn.Parameter(torch.randn(data_dim, device = self.device))
        else:
            raise ValueError(f'The argument of decoder_dist "{decoder_dist}" is not one of ["normal", "nb"]!')
    
        ## prior:

        # If invariant covariates are None: display warning to use FinVAE model
        if self.inv_covar_dim == 0:
            raise ValueError('The covariates for the invariant prior are None. Use FinVAE, the Factorized inVAE model, if this was intentional!')

        # If spurious covariates are None: use N(0,1) prior
        if (self.spur_covar_dim == 0) and (not inject_covar_in_latent):
            print('The covariates for the spurious prior are None. Defaulting to N(0,1) for that prior.')
        
        ## The biological/invariant prior is non-factorized with
        ## input: inv_covar; output: prior of size latent_dim_inv
        
        ## The noise/spurious prior is factorized gaussian with
        ## input: spur_covar; output: prior of size latent_dim_spur

        # Every prior part has the same number of hidden dims, layers and activation functions

        # Non-factorized prior for invariant latent space
        self.t_nn =  MLP(
            input_dim=self.latent_dim_inv, 
            output_dim=output_dim_prior_nn, 
            hidden_dim=hidden_dim_prior, 
            n_layers=n_layers_prior, 
            activation=activation, 
            device=device,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate
        ).to(self.device)

        self.params_t_nn = MLP(
            input_dim=self.inv_covar_dim, 
            output_dim=output_dim_prior_nn, 
            hidden_dim=hidden_dim_prior, 
            n_layers=n_layers_prior, 
            activation=activation, 
            device=device,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        self.params_t_suff = MLP(
            input_dim=self.inv_covar_dim, 
            output_dim=2*self.latent_dim_inv, 
            hidden_dim=hidden_dim_prior, 
            n_layers=n_layers_prior, 
            activation=activation, 
            device=device,
            batch_norm=batch_norm,
            dropout_rate=dropout_rate
        ).to(self.device)

        # Factorized prior for spurious latent space
        # Logic: factorized prior is modeled as a multivariate normal distribution with diagonal covariance (i.e. independent components) for this model
        if not self.inject_covar_in_latent:
            if self.fix_mean_spur_prior and self.fix_var_spur_prior:
                self.prior_mean_spur = torch.zeros(1).to(device)

                self.logl_spur = torch.ones(1).to(device)
            elif self.fix_mean_spur_prior:
                self.prior_mean_spur = torch.zeros(1).to(device)

                if self.spur_covar_dim != 0:
                    self.logl_spur = MLP(
                        self.spur_covar_dim, 
                        self.latent_dim_spur, 
                        hidden_dim_prior, 
                        n_layers_prior, 
                        activation=activation, 
                        slope=slope, 
                        device=device,
                        batch_norm=batch_norm,
                        dropout_rate=dropout_rate
                    ).to(device)
                else:
                    self.logl_spur = torch.ones(1).to(device)
            elif self.fix_var_spur_prior:
                if self.spur_covar_dim != 0:
                    self.prior_mean_spur = MLP(
                        self.spur_covar_dim, 
                        self.latent_dim_spur, 
                        hidden_dim_prior, 
                        n_layers_prior, 
                        activation=activation, 
                        slope=slope, 
                        device=device,        
                        batch_norm=batch_norm,
                        dropout_rate=dropout_rate
                    ).to(device)
                else:
                    self.prior_mean_spur = torch.zeros(1).to(device)

                self.logl_spur = torch.ones(1).to(device)
            elif (not self.fix_mean_spur_prior) and (not self.fix_var_spur_prior):
                # use one layer less for shared NN such that the whole prior NN has n_layers (end with act for first part)
                
                ## Noise latent prior
                if self.spur_covar_dim != 0:
                    self.prior_nn_spur = MLP(
                        self.spur_covar_dim, 
                        hidden_dim_prior, 
                        hidden_dim_prior, 
                        n_layers_prior-1, 
                        activation=activation, 
                        slope=slope, 
                        device=device, 
                        end_with_act=True,
                        batch_norm=batch_norm,
                        dropout_rate=dropout_rate
                    ).to(device)

                    self.prior_mean_spur = nn.Linear(hidden_dim_prior, self.latent_dim_spur, device=device)
                    self.logl_spur = nn.Linear(hidden_dim_prior, self.latent_dim_spur, device=device)
                else:
                    self.prior_mean_spur = torch.zeros(1).to(device)
                    self.logl_spur = torch.ones(1).to(device)
        else:
            self.prior_mean_spur = None
            self.logl_spur = None
        # Init every linear layer with xavier uniform
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
        h = self.x_y_e_encoder(xde)

        # Return latent mean and var
        return self.latent_mean(h), self.latent_log_var(h)

    def decode(self, z, inv_covar=None, spur_covar=None):
        if (spur_covar is None):
            zde = z
        elif spur_covar is not None:
            zde = torch.cat((z, spur_covar.view(-1, self.spur_covar_dim)), 1)

        # Disable other options for now -> Use only covariates of spurious prior, when "inject_covar_in_latent" is True
        #elif spur_covar is None:
        #    zde = torch.cat((z, inv_covar.view(-1, self.inv_covar_dim)), 1)
        #else: 
        #    zde = torch.cat((z, inv_covar.view(-1, self.inv_covar_dim), spur_covar.view(-1, self.spur_covar_dim)), 1)

        if self.decoder_dist == 'normal':
            decoder_mean = self.decoder_mean(zde)

            return decoder_mean
        elif self.decoder_dist == 'nb':
            decoder_raw_mean = self.decoder_raw_mean(zde)
            decoder_mean = self.decoder_freq(decoder_raw_mean)
            
            return decoder_mean, torch.exp(self.decoder_raw_theta)

    def prior_inv(self, z, inv_covar):
        t_nn = self.t_nn(z)
        params_t_nn = self.params_t_nn(inv_covar)

        t_suff = torch.cat((z, z**2), dim = 1)
        params_t_suff = self.params_t_suff(inv_covar)

        return t_nn, params_t_nn, t_suff, params_t_suff
    
    def prior_spur(self, spur_covar):
        if self.fix_mean_spur_prior and self.fix_var_spur_prior:
            return self.prior_mean_spur, self.logl_spur
        elif self.fix_mean_spur_prior:
            if (spur_covar is not None) and (not self.inject_covar_in_latent):
                logl_spur = (self.logl_spur(spur_covar).exp() + 1e-4)
            else:
                logl_spur = self.logl_spur

            return self.prior_mean_spur, logl_spur
        elif self.fix_var_spur_prior:
            if (spur_covar is not None) and (not self.inject_covar_in_latent):
                prior_mean_spur = self.prior_mean_spur(spur_covar)
            else:
                prior_mean_spur = self.prior_mean_spur

            return prior_mean_spur, self.logl_spur
        elif (not self.fix_mean_spur_prior) and (not self.fix_var_spur_prior):
            if (spur_covar is not None)and (not self.inject_covar_in_latent):
                prior_shared_spur = self.prior_nn_spur(spur_covar)

                prior_mean_spur = self.prior_mean_spur(prior_shared_spur)
                logl_spur = self.logl_spur(prior_shared_spur).exp() + 1e-4
            else:
                prior_mean_spur = self.prior_mean_spur
                logl_spur = self.logl_spur

            return prior_mean_spur, logl_spur
        
    def reparameterize(self, mean, logvar):
        std = (torch.exp(logvar) + 1e-4).sqrt()
        eps = torch.randn_like(std)
        return mean + eps * std

    def get_log_decoder_density(self, x, z, inv_covar=None, spur_covar=None):
        if spur_covar is None:
            zde = z
        else:
            zde = torch.cat((z, spur_covar.view(-1, self.spur_covar_dim)), 1)

        if self.decoder_dist == 'normal':
            decoder_mean = self.decode(zde)

            log_px_z = log_normal(x, decoder_mean, self.decoder_var)
        elif self.decoder_dist == 'nb':
            # decoder_mean are only the freq
            decoder_freq, decoder_theta = self.decode(zde)

            # multiply with library size to get "correct" means
            library = x.sum(dim=-1, keepdim=True)
            decoder_mean = decoder_freq * library

            log_px_z = log_nb_positive(x, decoder_mean, decoder_theta)
        
        return log_px_z

    def forward(self, x, inv_covar, spur_covar):
        # Encoder
        latent_mean, latent_logvar = self.encode(x, inv_covar, spur_covar)

        # Latent variable
        z = self.reparameterize(latent_mean, latent_logvar)

        # Decoder
        if self.decoder_dist == 'normal':
            if self.inject_covar_in_latent:
                decoder_mean = self.decode(z, inv_covar, spur_covar)
            else:
                decoder_mean = self.decode(z)

            return decoder_mean, latent_mean, latent_logvar, z
        elif self.decoder_dist == 'nb':
            if self.inject_covar_in_latent:
                decoder_freq, decoder_theta = self.decode(z, inv_covar, spur_covar)
            else:
                # decoder_mean are only the freq
                decoder_freq, decoder_theta = self.decode(z)
            
            # multiply with library size to get "correct" means
            library = x.sum(dim=-1, keepdim=True)
            decoder_mean = decoder_freq * library

            return decoder_mean, decoder_theta, latent_mean, latent_logvar, z

    def sample_latent(self, x, inv_covar, spur_covar):
        # Encoder
        latent_mean, latent_logvar = self.encode(x, inv_covar, spur_covar)

        # Latent variable
        z = self.reparameterize(latent_mean, latent_logvar)

        return z

    def elbo(self, x, inv_covar, spur_covar):
        if self.decoder_dist == 'normal':
            decoder_mean, latent_mean, latent_logvar, z = self.forward(x, inv_covar, spur_covar)

            log_px_z = log_normal(x, decoder_mean, self.decoder_var)
        elif self.decoder_dist == 'nb':
            decoder_mean, decoder_theta, latent_mean, latent_logvar, z = self.forward(x, inv_covar, spur_covar)

            log_px_z = log_nb_positive(x, decoder_mean, decoder_theta)

        log_qz_xde = log_normal(z, latent_mean, (latent_logvar.exp() + 1e-4))

        # Clone z first then calculate parts of the prior with derivative wrt cloned z
        # prior
        z_inv_copy = z[:, :self.latent_dim_inv].detach().requires_grad_(True)
        
        # Only use the latent invariant space
        t_nn, params_t_nn, t_suff, params_t_suff = self.prior_inv(z_inv_copy, inv_covar)

       # Batched dot product for unnormalized prior probability
        log_pz_d_inv = (
            torch.bmm(t_nn.view((-1, 1, self.output_dim_prior_nn)), params_t_nn.view((-1, self.output_dim_prior_nn, 1))).view(-1) + 
            torch.bmm(t_suff.view((-1, 1, self.latent_dim_inv*2)), params_t_suff.view((-1, self.latent_dim_inv*2, 1))).view(-1)
        )

        # Implement constant log prior so prior params are not updated but grads are backpropagated for the encoder
        self.t_nn.requires_grad_(False)
        self.params_t_nn.requires_grad_(False)
        self.params_t_suff.requires_grad_(False)

        t_nn_copy = self.t_nn(z[:, :self.latent_dim_inv])
        params_t_nn_copy = self.params_t_nn(inv_covar)

        t_suff_copy = torch.cat((z[:, :self.latent_dim_inv], (z[:, :self.latent_dim_inv])**2), dim = 1)
        params_t_suff_copy = self.params_t_suff(inv_covar)

        log_pz_d_inv_copy = (
            torch.bmm(t_nn_copy.view((-1, 1, self.output_dim_prior_nn)), params_t_nn_copy.view((-1, self.output_dim_prior_nn, 1))).view(-1) + 
            torch.bmm(t_suff_copy.view((-1, 1, self.latent_dim_inv*2)), params_t_suff_copy.view((-1, self.latent_dim_inv*2, 1))).view(-1)
        )

        self.t_nn.requires_grad_(True)
        self.params_t_nn.requires_grad_(True)
        self.params_t_suff.requires_grad_(True)

        # Calculate derivatives of prior automatically
        dprior_dz = grad(log_pz_d_inv, z_inv_copy, grad_outputs = torch.ones(log_pz_d_inv.shape, device=self.device), create_graph=True)[0]
        d2prior_d2z = grad(dprior_dz, z_inv_copy, grad_outputs = torch.ones(dprior_dz.shape, device=self.device), create_graph = True)[0]

        # Spurious prior
        prior_mean_spur, prior_var_spur = self.prior_spur(spur_covar)

        if not self.inject_covar_in_latent:
            log_pz_e_spur = log_normal(
                z[:, self.latent_dim_inv:], 
                prior_mean_spur * torch.ones(prior_var_spur.shape, device = self.device), # need for shape match (could change check in log_normal function)
                prior_var_spur
            )
        else:
            log_pz_e_spur = 0

        if self.reg_sm == 0:
            sm_part = (d2prior_d2z + torch.mul(0.5, torch.pow(dprior_dz, 2))).sum(dim=1).mean()
        else:
            sm_part = (d2prior_d2z + torch.mul(0.5, torch.pow(dprior_dz, 2)) + d2prior_d2z.pow(2).mul(self.reg_sm)).sum(dim=1).mean()

        objective_function = (
            (log_px_z + log_pz_d_inv_copy + log_pz_e_spur - log_qz_xde).mean().div(self.normalize_constant)   - 
            sm_part
        )

        return objective_function, z

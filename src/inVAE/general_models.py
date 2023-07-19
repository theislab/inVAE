from numbers import Number

from typing import List, Literal, Optional, Dict, Union

import time
import numpy as np
import torch
from torch import distributions as dist
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import grad

from torch.utils.tensorboard import SummaryWriter

import scanpy as sc
from anndata.experimental.pytorch import AnnLoader

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from abc import ABC, abstractmethod
from anndata import AnnData
from itertools import chain

from scipy import sparse

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

# Implementation from scanpy: see https://github.com/scverse/scanpy/blob/ed3b277b2f498e3cab04c9416aaddf97eec8c3e2/scanpy/_utils/__init__.py#L487
def check_nonnegative_integers(X: Union[np.ndarray, sparse.spmatrix]):
    """Checks values of X to ensure it is count data"""
    from numbers import Integral

    data = X if isinstance(X, np.ndarray) else X.data
    # Check no negatives
    if np.signbit(data).any():
        return False
    # Check all are integers
    elif issubclass(data.dtype.type, Integral):
        return True
    elif np.any(~np.equal(np.mod(data, 1), 0)):
        return False
    else:
        return True

def log_pdf(x, mu, v, reduce=True, param_shape=None, device = 'cpu'):
    """compute the log-pdf of a normal distribution with diagonal covariance"""
    if param_shape is not None:
        mu, v = mu.view(param_shape), v.view(param_shape)

    c = 2 * np.pi * torch.ones(1).to(device)
    lpdf = -0.5 * (torch.log(c) + v.log() + (x - mu).pow(2).div(v))
    if reduce:
        return lpdf.sum(dim=-1)
    else:
        return lpdf

def log_nb_positive(
    x: torch.Tensor,
    mu: torch.Tensor,
    theta: torch.Tensor,
    eps: float = 1e-8,
    log_fn: callable = torch.log,
    lgamma_fn: callable = torch.lgamma,
    reduce: bool = True
):
    """
    Log likelihood (scalar) of a minibatch according to a nb model.
    Parameters
    ----------
    x
        data
    mu
        mean of the negative binomial (has to be positive support) (shape: minibatch x genes)
    theta
        inverse dispersion parameter (has to be positive support) (shape: minibatch x genes)
    eps
        numerical stability constant
    """
    
    log = log_fn
    lgamma = lgamma_fn
    
    log_theta_mu_eps = log(theta + mu + eps)
    
    res = (
        theta * (log(theta + eps) - log_theta_mu_eps)
        + x * (log(mu + eps) - log_theta_mu_eps)
        + lgamma(x + theta)
        - lgamma(theta)
        - lgamma(x + 1)
    )
    
    if reduce:
        return res.sum(dim=-1)
    else:
        return res

class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass

class Normal(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self._dist = dist.normal.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'

    def sample(self, mu, v):
        eps = self._dist.sample(mu.size()).squeeze()
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def log_pdf(self, x, mu, v, reduce=True, param_shape=None):
        """compute the log-pdf of a normal distribution with diagonal covariance"""
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)
        lpdf = -0.5 * (torch.log(self.c) + v.log() + (x - mu).pow(2).div(v))
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf

def log_normal(x, mu=None, v=None, reduce=True):
    """Compute the log-pdf of a normal distribution with diagonal covariance"""
    if mu.shape[1] != v.shape[0] and mu.shape != v.shape:
        raise ValueError(f'The mean and variance vector do not have the same shape:\n\tmean: {mu.shape}\tvariance: {v.shape}')

    logpdf = -0.5 * (np.log(2 * np.pi) + v.log() + (x - mu).pow(2).div(v))

    if reduce:
        logpdf = logpdf.sum(dim=-1)

    return logpdf

class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', 
                 slope=0.1, device='cpu', end_with_act = False, batch_norm = False):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        self.end_with_act = end_with_act
        self.batch_norm = batch_norm

        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            if self.end_with_act:
                self.activation = [activation] * (self.n_layers)
            else:
                self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.activation = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'relu':
                self._act_f.append(F.relu)
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim)]
            if self.batch_norm:
                _bn_list = [nn.BatchNorm1d(self.output_dim, momentum=0.01, eps=0.001)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            if self.batch_norm:
                _bn_list = [nn.BatchNorm1d(self.hidden_dim[0], momentum=0.01, eps=0.001)]

            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
                if self.batch_norm:
                    _bn_list.append(nn.BatchNorm1d(self.hidden_dim[i], momentum=0.01, eps=0.001))

            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
            if self.batch_norm:
                _bn_list.append(nn.BatchNorm1d(self.output_dim, momentum=0.01, eps=0.001))

        self.fc = nn.ModuleList(_fc_list)

        if self.batch_norm:
            self.bn = nn.ModuleList(_bn_list)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                if self.batch_norm:
                    h = self.bn[c](self.fc[c](h))
                else:
                    h = self.fc[c](h)

                if self.end_with_act:
                    h = self._act_f[c](h)
            else:
                if self.batch_norm:
                    h = self._act_f[c](self.bn[c](self.fc[c](h)))
                else:
                    h = self._act_f[c](self.fc[c](h))
        return h
    
class inVAE(ABC):

    def get_negative_elbo(
        self, 
        adata: Optional[AnnData] = None,
    ):
        if adata is None:
            print('Getting ELBO of saved adata!')
            
            self.module.eval()

            transformed_data = self.transformed_data
        else:
            print(f'Calculating ELBO of passed adata by trying to transfer setup from the adata the model was trained on!')

            use_cuda = (self.device == 'cuda')

            data_loader = AnnLoader(adata, batch_size = 1, shuffle = False, use_cuda = use_cuda, convert = self.data_loading_encoders)

            transformed_data = data_loader.dataset[:]
        
        # Gene counts
        x = transformed_data.layers[self.layer] if self.layer is not None else transformed_data.X

        # Covariates for invariate prior
        inv_tensor_list = [transformed_data.obs[covar].float() for covar in self.list_inv_covar]
        inv_covar = torch.cat(inv_tensor_list, dim = 1) if (self.inv_covar_dim != 0) else None

        # Covariates for spurious prior
        spur_tensor_list = [transformed_data.obs[covar].float() for covar in self.list_spur_covar]
        spur_covar = torch.cat(spur_tensor_list, dim = 1) if (self.spur_covar_dim != 0) else None

        elbo, _ = self.module.elbo(x, inv_covar, spur_covar)

        neg_elbo = -elbo.detach().cpu().numpy()

        return neg_elbo

    def save(
        self,
        save_dir: str,
    ):
        print('Saving the pytorch module...')
        print('To load the model later you need to save the hyperparameters in a separate file/dictionary.')

        torch.save(self.module.state_dict(), save_dir)

    def load(
        self,
        save_dir: str,
    ):
        print('Loading the model from given checkpoint...')
        self.module.load_state_dict(torch.load(save_dir, map_location = torch.device(self.device)))
        # Put model in eval state (e.g. for batch-norm layers)
        self.module.eval()
    
    def get_latent_representation(
        self,
        adata: Optional[AnnData] = None,
        type: Literal['full', 'invariant', 'inv', 'spurious', 'spur'] = 'invariant',
    ):
        if adata is None:
            print('Using saved adata for latent representation!')

            self.module.eval()

            transformed_data = self.transformed_data
        else:
            print(f'Calculating latent representation of passed adata by trying to transfer setup from the adata the model was trained on!')

            use_cuda = (self.device == 'cuda')

            data_loader = AnnLoader(adata, batch_size = 1, shuffle = False, use_cuda = use_cuda, convert = self.data_loading_encoders)

            transformed_data = data_loader.dataset[:]

        with torch.no_grad():
            # Gene counts
            x = transformed_data.layers[self.layer] if self.layer is not None else transformed_data.X

            # Covariates for invariate prior
            inv_tensor_list = [transformed_data.obs[covar].float() for covar in self.list_inv_covar]
            inv_covar = torch.cat(inv_tensor_list, dim = 1) if (self.inv_covar_dim != 0) else None

            # Covariates for spurious prior
            spur_tensor_list = [transformed_data.obs[covar].float() for covar in self.list_spur_covar]
            spur_covar = torch.cat(spur_tensor_list, dim = 1) if (self.spur_covar_dim != 0) else None

            latent_mean, _ = self.module.encode(x, inv_covar, spur_covar)

            if type == 'full':
                # Do nothing: whole latente space
                pass
            elif (type == 'invariant') or (type == 'inv'):
                latent_mean = latent_mean[:, :self.latent_dim_inv]
            elif (type == 'spurious') or (type == 'spur'):
                latent_mean = latent_mean[:, self.latent_dim_inv:]
            else:
                print(f"{type} is not a valid type of latent representation! Type has to be in ['full', 'invariant', 'inv', 'spurious', 'spur'].")

        return latent_mean.detach().cpu().numpy()

    def train(
        self,
        n_epochs: int = None,
        lr_train: float = None,
        weight_decay: float = None,
        log_dir: str = None,
        log_freq: int = 25, # in iterations
        print_every_n_epochs: int = None,
    ):
        if n_epochs is None:
            n_epochs = 500
            print(f'n_epochs is None, defaulting to {n_epochs} epochs. If you want a different number of epochs, pass it to the train function!')

        if lr_train is None:
            raise ValueError('lr_train is None, set a learning rate!')
        
        if weight_decay is None:
            raise ValueError('weight_decay is None, set the weight decay!')
        
        self.module = self.module.to(device=self.device)
        self.module = self.module.float()

        optimizer = optim.Adam(self.module.parameters(), lr = lr_train, weight_decay = weight_decay)

        # Logger
        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)

        max_iter = len(self.data_loader) * n_epochs

        if print_every_n_epochs is None:
            print_every_n_iters = max_iter / 5
            print(f'Defaulting to printing training loss every {int(print_every_n_iters/len(self.data_loader))} epochs. If you want a different number, set it for the train function!')
        else:
            print_every_n_iters = print_every_n_epochs * len(self.data_loader)

        self.module.train()
        iteration = 0
        loss_epoch = 0

        #raise ValueError('end of train...')
        print('Starting training of model:')

        while iteration < max_iter:
            start = time.time()
            loss_epoch = 0
            for _, batch in enumerate(self.data_loader):
                ## Extract data
                # Gene counts
                x = batch.layers[self.layer] if self.layer is not None else batch.X

                # Covariates for invariate prior
                inv_tensor_list = [batch.obs[covar].float() for covar in self.list_inv_covar]
                inv_covar = torch.cat(inv_tensor_list, dim = 1) if (self.inv_covar_dim != 0) else None

                # Covariates for spurious prior
                spur_tensor_list = [batch.obs[covar].float() for covar in self.list_spur_covar]
                spur_covar = torch.cat(spur_tensor_list, dim = 1) if (self.spur_covar_dim != 0) else None

                #print(f'Shape of X: {x.shape}\nfirst obs: {x[0]}\nand datatype {x.dtype}')

                iteration += 1

                optimizer.zero_grad(set_to_none=True)

                objective_fct, _ = self.module.elbo(x, inv_covar, spur_covar)

                #print(f'\tFor the {i+1}-th iteration the loss is: {-objective_fct.detach().numpy()}')
                #print(f'and the estimated z is:\n {z_est.detach().numpy()}')

                objective_fct.mul(-1).backward()
                optimizer.step()

                temp_loss = -objective_fct.detach().cpu().numpy()
                loss_epoch += temp_loss
                
                if (log_dir is not None) and (iteration % log_freq == 0):
                    writer.add_scalar('train_loss', temp_loss, iteration)

            end = time.time()

            time_sec = end - start

            time_sec = np.round(time_sec, decimals=2)

            if iteration % print_every_n_iters == 0:
                print(f'\tepoch {int(iteration/len(self.data_loader))}/{n_epochs} took {time_sec}s; loss: {np.round(loss_epoch / len(self.data_loader), 2):.2f}')

            if np.isnan(loss_epoch):
                print(f'Loss is nan at epoch {int(iteration/len(self.data_loader))}/{n_epochs}, stopping training!')
                break
            
        self.module.eval()
        print('Training done!')

    @staticmethod
    def check_data(
        adata: AnnData,
        layer: Optional[str] = None,
        decoder_dist: Literal['nb', 'normal'] = None
    ):
        if decoder_dist == 'nb':
            is_count_data = check_nonnegative_integers(adata.layers[layer] if layer is not None else adata.X)

            if not is_count_data:
                print(f'Non-count data detected in adata in layer {layer} and the decoder distribution is "negative-binomial". Check that the data really contains counts!')
        elif decoder_dist == 'normal':
            #print('Assuming data is scaled to mean zero and variance one per gene, for decoder distribution "normal"!')
            mean_values = (adata.layers[layer] if layer is not None else adata.X).mean(axis=0)
            var_values = (adata.layers[layer] if layer is not None else adata.X).var(axis=0)

            max_mean = np.absolute(mean_values).max()

            max_var = np.absolute(var_values - 1).max()
            ind_max_var = np.argmax(np.absolute(var_values - 1))

            if max_mean > 1e-7:
                print(f'Warning: The biggest mean expression for a gene is {max_mean}!')
                print('Assuming scaled data here (mean zero and variance one) for "normal" decoder distribution the mean should be close to zero!')

            if max_var > 1e-4:
                print(f'Warning: The biggest variance for a gene expression {var_values[ind_max_var]}!')
                print('Assuming scaled data here (mean zero and variance one) for "normal" decoder distribution the variance should be close to one!')
        else:
            raise ValueError(f'{decoder_dist} is not in ["nb", "normal"]!')

    @staticmethod
    def setup_adata(
        adata: AnnData,
        #layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        label_key: Optional[str] = None,
        inv_covar_keys: Dict[str, List[str]] = None,
        spur_covar_keys: Dict[str, List[str]] = None,
        device: str = 'cpu',
        batch_size: int = 256,
        **kwargs,
    ):  
        # To return encoders for use later
        dict_encoders = {}

        # To use for setting up the data
        obs_encoder_list = []
        obs_encoders = {}

        ## Define transformers for the data

        # First: encode the batches
        encoder_batch = OneHotEncoder(sparse_output=False, dtype=np.float32)
        encoder_batch.fit(adata.obs[batch_key].to_numpy()[:, None])

        dict_encoders[batch_key] = encoder_batch
        obs_encoders[batch_key] = (lambda b: encoder_batch.transform(b.to_numpy()[:, None]))

        # Label Encoder for cell_types (or in general: label key)
        if label_key is not None:
            encoder_label = LabelEncoder()
            encoder_label.fit(adata.obs[label_key])

            dict_encoders[f'label_{label_key}'] = encoder_label

        if len(encoder_batch.categories_[0]) == 1:
            print('Warning: There is only one batch in the (training) data. This model might not behave as intended.')
            print('Consider using the whole latent space (option: type = "full") as a representation instead of only the invariant latent space!')

        for key, covars in spur_covar_keys.items():
            if key == 'cont':
                # Do we have to do something?
                pass
            elif key == 'cat':
                for covar in covars:
                    # Default for now is to use one-hot encoding for categorical vars
                    dict_encoders[covar] = OneHotEncoder(sparse_output=False, dtype=np.float32, handle_unknown='ignore')
                    dict_encoders[covar].fit(adata.obs[covar].to_numpy()[:, None])

                    # Try list approach:
                    #obs_encoders[covar] = lambda s: dict_encoders[covar].transform(s.to_numpy()[:, None])
                    obs_encoder_list.append({covar: lambda s, covar=covar: dict_encoders[covar].transform(s.to_numpy()[:, None])})
            else:
                raise ValueError(f'{key} is not in ["cont", "cat"] and therefore not a valid covariate key for the prior!')

        for key, covars in inv_covar_keys.items():
            if key == 'cont':
                # Do we have to do something?
                pass
            elif key == 'cat':
                for covar in covars:
                    # Default for now is to use one-hot encoding for categorical vars
                    dict_encoders[covar] = OneHotEncoder(sparse_output=False, dtype=np.float32, handle_unknown='ignore')
                    dict_encoders[covar].fit(adata.obs[covar].to_numpy()[:, None])

                    # list approach
                    #obs_encoders[covar] = lambda i: dict_encoders[covar].transform(i.to_numpy()[:, None])
                    obs_encoder_list.append({covar: lambda s, covar=covar: dict_encoders[covar].transform(s.to_numpy()[:, None])})
            else:
                raise ValueError(f'{key} is not in ["cont", "cat"] and therefore not a valid covariate key for the prior!')

        for enc in obs_encoder_list:
            obs_encoders.update(enc)

        encoders = {'obs' : obs_encoders}

        use_cuda = (device == 'cuda')

        data_loader = AnnLoader(adata, batch_size = batch_size, shuffle = True, use_cuda = use_cuda, convert = encoders)

        print('Data loading done!')

        transformed_data = data_loader.dataset[:]

        # Dict or tuple as return?
        return (dict_encoders, encoders, data_loader, transformed_data)

class FinVAE(inVAE):

    def __init__(
        self, 
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        label_key: Optional[str] = None,
        inv_covar_keys: Dict[str, List[str]] = None,
        spur_covar_keys: Dict[str, List[str]] = None,
        latent_dim: int = 10, 
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
        kl_rate: float = 1.0,
        batch_size: int = 256,
        elbo_version: Literal['kl_div', 'sample'] = 'sample',
        **kwargs,
    ):
        super().__init__()

        if adata is None:
            raise ValueError('Adata is None, check if you passed the data to the model!')
        
        self.adata = adata

        if layer is None:
            print('Layer is None, check if you want to specify the layer of adata!')
        
        # Assign all keys
        self.layer = layer
        self.batch_key = batch_key
        self.label_key = label_key

        self.inv_covar_keys = inv_covar_keys
        self.spur_covar_keys = spur_covar_keys

        # Number of genes -> Rename?
        self.data_dim = adata.shape[1]

        self.batch_size = batch_size

        self.device = device

        # Latent dimensions
        self.latent_dim = latent_dim
        self.latent_dim_spur = latent_dim_spur
        self.latent_dim_inv = latent_dim - latent_dim_spur

        # Set-up data
        
        # Check if the layer of the data contains the right data:
        self.check_data(adata, layer, decoder_dist)
        
        self.dict_encoders, self.data_loading_encoders, self.data_loader, self.transformed_data = self.setup_adata(adata, batch_key, label_key, inv_covar_keys, spur_covar_keys, device, batch_size)
        
        # Flatten list of covariates
        self.list_spur_covar = [value for key, value in self.spur_covar_keys.items() if key in ['cat', 'cont']]
        self.list_spur_covar = list(chain.from_iterable(self.list_spur_covar))

        self.list_inv_covar = [value for key, value in self.inv_covar_keys.items() if key in ['cat', 'cont']]
        self.list_inv_covar = list(chain.from_iterable(self.list_inv_covar))

        # Need to setup data first; then add dim of covariates:
        self.spur_covar_dim = sum([self.transformed_data.obs[covar].shape[1] for covar in self.list_spur_covar])
        self.inv_covar_dim = sum([self.transformed_data.obs[covar].shape[1] for covar in self.list_inv_covar])

        self.module = FinVAEmodule(
            latent_dim = latent_dim, 
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
            kl_rate = kl_rate,
            batch_size = batch_size,
            elbo_version = elbo_version,
            data_dim = self.data_dim,
            inv_covar_dim = self.inv_covar_dim,
            spur_covar_dim = self.spur_covar_dim,
        )
    
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
        kl_rate: float = 1.0,
        batch_size: int = 256,
        elbo_version: Literal['kl_div', 'sample'] = 'sample',
        data_dim: int = None,
        inv_covar_dim: int = None,
        spur_covar_dim: int = None,
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

        # Training HPs
        self.beta = kl_rate
        self.elbo_version = elbo_version
        self.batch_size = batch_size

        self.device = device

        # Data related values

        # Number of genes -> Rename?
        self.data_dim = data_dim

        # Need to setup data first; then add dim of covariates:
        self.inv_covar_dim = inv_covar_dim
        self.spur_covar_dim = spur_covar_dim

        #raise TypeError('Debug init function of FinVAEmodule...')

        ## Two priors:
        # One prior for invariant latent space conditioned on invariant covariates
        self.prior_dist_inv = Normal(device=device)

        # One prior for spurious latent space conditioned on spurious covariates
        self.prior_dist_spur = Normal(device=device)

        self.decoder_dist_fct = Normal(device=device)
        
        self.encoder_dist = Normal(device=device)

        # If invariant or spurious covariates are None: use N(0,1) prior
        if self.inv_covar_dim == 0:
            print('The covariates for the invariant prior are None. Defaulting to N(0,1) for that prior.')

        if self.spur_covar_dim == 0:
            print('The covariates for the spurious prior are None. Defaulting to N(0,1) for that prior.')
        
        ## prior_params
        # Logic: both priors are modeled as a multivariate normal distribution with diagonal covariance (i.e. independent components) for this model
        if self.fix_mean_prior and self.fix_var_prior:
            self.prior_mean_inv = torch.zeros(1).to(device)
            self.prior_mean_spur = torch.zeros(1).to(device)

            self.logl_inv = torch.ones(1).to(device)
            self.logl_spur = torch.ones(1).to(device)
        elif self.fix_mean_prior:
            self.prior_mean_inv = torch.zeros(1).to(device)
            self.prior_mean_spur = torch.zeros(1).to(device)

            if self.inv_covar_dim != 0:
                self.logl_inv = MLP(self.inv_covar_dim, self.latent_dim_inv, hidden_dim, n_layers, activation=activation, slope=slope, device=device).to(device)
            else:
                self.logl_inv = torch.ones(1).to(device)

            if self.spur_covar_dim != 0:
                self.logl_spur = MLP(self.spur_covar_dim, self.latent_dim_spur, hidden_dim, n_layers, activation=activation, slope=slope, device=device).to(device)
            else:
                self.logl_spur = torch.ones(1).to(device)
        elif self.fix_var_prior:
            if self.inv_covar_dim != 0:
                self.prior_mean_inv = MLP(self.inv_covar_dim, self.latent_dim_inv, hidden_dim, n_layers, activation=activation, slope=slope, device=device).to(device)
            else:
                self.prior_mean_inv = torch.zeros(1).to(device)

            if self.spur_covar_dim != 0:
                self.prior_mean_spur = MLP(self.spur_covar_dim, self.latent_dim_spur, hidden_dim, n_layers, activation=activation, slope=slope, device=device).to(device)
            else:
                self.prior_mean_spur = torch.zeros(1).to(device)

            self.logl_inv = torch.ones(1).to(device)
            self.logl_spur = torch.ones(1).to(device)
        elif (not self.fix_mean_prior) and (not self.fix_var_prior):
            # use one layer less for shared NN such that the whole prior NN has n_layers (end with act for first part)
            
            ## Invariant latent prior
            if self.inv_covar_dim != 0:
                self.prior_nn_inv = MLP(
                    self.inv_covar_dim, hidden_dim, hidden_dim, n_layers-1, activation=activation, slope=slope, device=device, end_with_act=True
                ).to(device)

                self.prior_mean_inv = nn.Linear(hidden_dim, self.latent_dim_inv, device=device)
                self.logl_inv = nn.Linear(hidden_dim, self.latent_dim_inv, device=device)
            else:
                self.prior_mean_inv = torch.zeros(1).to(device)

            ## Noise latent prior
            if self.spur_covar_dim != 0:
                self.prior_nn_spur = MLP(
                    self.spur_covar_dim, hidden_dim, hidden_dim, n_layers-1, activation=activation, slope=slope, device=device, end_with_act=True
                ).to(device)

                self.prior_mean_spur = nn.Linear(hidden_dim, self.latent_dim_spur, device=device)
                self.logl_spur = nn.Linear(hidden_dim, self.latent_dim_spur, device=device)
            else:
                self.prior_mean_spur = torch.zeros(1).to(device)

        # decoder params
        if self.decoder_dist == 'normal':
            self.f = MLP(
                latent_dim,
                self.data_dim, 
                hidden_dim, 
                n_layers, 
                activation=activation,
                slope=slope,
                device=device,
                batch_norm=batch_norm
            ).to(device)
            
            self.decoder_var = .01 * torch.ones(1).to(device)
        elif self.decoder_dist == 'nb':
            ## Decoder for NB distribution consists of:
            #  mean -> (raw mean values + Softmax) * library_size
            #  theta -> parameter shared across cells (so variability per gene)

            self.decoder_raw_mean = MLP(
                input_dim=latent_dim, 
                output_dim=hidden_dim, 
                hidden_dim=hidden_dim, 
                n_layers=n_layers, 
                activation=activation, 
                device=device,
                batch_norm=self.batch_norm
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

    def decode(self, z):
        if self.decoder_dist == 'normal':
            f = self.f(z)
            return f, self.decoder_var
        elif self.decoder_dist == 'nb':
            decoder_raw_mean = self.decoder_raw_mean(z)
            decoder_mean = self.decoder_freq(decoder_raw_mean)
            
            return decoder_mean, torch.exp(self.decoder_raw_theta)

    def prior(self, inv_covar, spur_covar):
        if self.fix_mean_prior and self.fix_var_prior:
            return (self.prior_mean_inv, self.prior_mean_spur), (self.logl_inv, self.logl_spur)
        elif self.fix_mean_prior:
            logl_inv = (self.logl_inv(inv_covar).exp() + 1e-4) if (inv_covar is not None) else self.logl_inv
            logl_spur = (self.logl_spur(spur_covar).exp() + 1e-4)  if (spur_covar is not None) else self.logl_spur

            return (self.prior_mean_inv, self.prior_mean_spur), (logl_inv, logl_spur)
        elif self.fix_var_prior:
            prior_mean_inv = self.prior_mean_inv(inv_covar) if (inv_covar is not None) else self.prior_mean_inv
            prior_mean_spur = self.prior_mean_spur(spur_covar) if (spur_covar is not None) else self.prior_mean_spur

            return (prior_mean_inv, prior_mean_spur), (self.logl_inv, self.logl_spur)
        elif (not self.fix_mean_prior) and (not self.fix_var_prior):
            if inv_covar is not None:
                prior_shared_inv = self.prior_nn_inv(inv_covar)

                prior_mean_inv = self.prior_mean_inv(prior_shared_inv)
                logl_inv = self.logl_inv(prior_shared_inv).exp() + 1e-4
            else:
                prior_mean_inv = self.prior_mean_inv
                logl_inv = self.logl_inv

            if spur_covar is not None:
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
    
    def get_log_decoder_density(self, x, z):
        if self.decoder_dist == 'normal':
            decoder_params = self.decode(z)
            log_px_z = self.decoder_dist_fct.log_pdf(x, *decoder_params)
        elif self.decoder_dist == 'nb':
            # decoder_mean are only the freq
            decoder_mean, decoder_theta = self.decode(z)

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
            decoder_params = self.decode(z)

            return decoder_params, encoder_params, z, prior_params_mean, prior_params_var
        elif self.decoder_dist == 'nb':
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
            log_pzs_e = self.prior_dist_spur.log_pdf(z[:, self.latent_dim_inv:], prior_params_mean[1], prior_params_var[1])

            return (log_px_z + self.beta * (log_pzi_d + log_pzs_e - log_qz_xde)).mean(), z
        elif self.elbo_version == 'kl_div':
            if torch.any(torch.isnan(g)) or torch.any(torch.isnan(v)):
                return torch.tensor(float('nan')), z
            
            # Encoder distribution
            qz_xde = dist.Normal(g, v.sqrt())

            # Prior distribution (independent components -> diagonal cov)
            full_prior_mean = torch.cat(prior_params_mean, dim=1) if not self.fix_mean_prior else prior_params_mean[0]
            full_prior_var = torch.cat(prior_params_var, dim=1) if not self.fix_var_prior else prior_params_var[0]

            pz_de = dist.Normal(full_prior_mean, full_prior_var.sqrt())

            # KL-divergence between encoder and prior
            kl_div = dist.kl_divergence(qz_xde, pz_de).sum(dim=1)

            # loss
            return (log_px_z - self.beta * kl_div).mean(), z

class NFinVAE(inVAE):

    def __init__(
        self, 
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        label_key: Optional[str] = None,
        inv_covar_keys: Dict[str, List[str]] = None,
        spur_covar_keys: Dict[str, List[str]] = None,
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
        batch_size: int = 256,
        reg_sm: float = 0.0,
        output_dim_prior_nn: int =  None,
        hidden_dim_prior: int =  None,
        n_layers_prior: int =  None,
        **kwargs,
    ):
        super().__init__()

        if adata is None:
            raise ValueError('Adata is None, check if you passed the data to the model!')
        
        self.adata = adata

        if layer is None:
            print('Layer is None, check if you want to specify the layer of adata!')
        
        # Assign all keys
        self.layer = layer
        self.batch_key = batch_key
        self.label_key = label_key

        self.inv_covar_keys = inv_covar_keys
        self.spur_covar_keys = spur_covar_keys

        # Number of genes -> Rename?
        self.data_dim = adata.shape[1]

        self.batch_size = batch_size

        self.device = device

        # Latent dimensions
        self.latent_dim = latent_dim
        self.latent_dim_spur = latent_dim_spur
        self.latent_dim_inv = latent_dim - latent_dim_spur

        # Set-up data

        # Check data first
        self.check_data(adata, layer, decoder_dist)

        self.dict_encoders, self.data_loading_encoders, self.data_loader, self.transformed_data = self.setup_adata(adata, batch_key, label_key, inv_covar_keys, spur_covar_keys, device, batch_size)
        
        # Flatten list of covariates
        self.list_spur_covar = [value for key, value in self.spur_covar_keys.items() if key in ['cat', 'cont']]
        self.list_spur_covar = list(chain.from_iterable(self.list_spur_covar))

        self.list_inv_covar = [value for key, value in self.inv_covar_keys.items() if key in ['cat', 'cont']]
        self.list_inv_covar = list(chain.from_iterable(self.list_inv_covar))

        # Dim of covariates:
        self.spur_covar_dim = sum([self.transformed_data.obs[covar].shape[1] for covar in self.list_spur_covar])
        self.inv_covar_dim = sum([self.transformed_data.obs[covar].shape[1] for covar in self.list_inv_covar])

        self.module = NFinVAEmodule(
            latent_dim = latent_dim, 
            latent_dim_spur = latent_dim_spur,
            n_layers = n_layers, 
            hidden_dim = hidden_dim,
            activation = activation, 
            slope = slope,
            device = device,
            normalize_constant = normalize_constant,
            fix_mean_spur_prior = fix_mean_spur_prior,
            fix_var_spur_prior =  fix_var_spur_prior,
            decoder_dist = decoder_dist,
            batch_norm = batch_norm,
            batch_size = batch_size,
            data_dim = self.data_dim,
            inv_covar_dim = self.inv_covar_dim,
            spur_covar_dim = self.spur_covar_dim,
            reg_sm = reg_sm,
            output_dim_prior_nn = output_dim_prior_nn,
            hidden_dim_prior = hidden_dim_prior,
            n_layers_prior = n_layers_prior
        )

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
        batch_size: int = 256,
        data_dim: int = None,
        inv_covar_dim: int = None,
        spur_covar_dim: int = None,
        reg_sm: float = 0,
        output_dim_prior_nn: int = None,
        hidden_dim_prior: int = None,
        n_layers_prior: int = None,
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
        self.reg_sm = reg_sm
        self.normalize_constant = normalize_constant

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
            batch_norm = batch_norm
        ).to(self.device)

        # Latent mean and log variance
        self.latent_mean = nn.Linear(hidden_dim, self.latent_dim)
        self.latent_log_var = nn.Linear(hidden_dim, self.latent_dim)

        # decoder
        if decoder_dist == 'normal':
            self.decoder_mean = MLP(
                input_dim=latent_dim, 
                output_dim=data_dim, 
                hidden_dim=hidden_dim, 
                n_layers=n_layers, 
                activation=activation, 
                device=device,
                batch_norm=batch_norm
            ).to(self.device)

            self.decoder_var = torch.mul(0.01, torch.ones(data_dim, device = self.device))
        elif decoder_dist == 'nb':
            ## Decoder for NB distribution consists of:
            #  mean -> (raw mean values + Softmax) * library_size
            #  theta -> parameter shared across cells

            self.decoder_raw_mean = MLP(
                input_dim=latent_dim, 
                output_dim=hidden_dim, 
                hidden_dim=hidden_dim, 
                n_layers=n_layers, 
                activation=activation, 
                device=device,
                batch_norm=batch_norm
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
            raise ValueError('The covariates for the invariant prior are None. Use FinVAE the factorized inVAE model if this was intentional!')

        # If spurious covariates are None: use N(0,1) prior
        if self.spur_covar_dim == 0:
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
            device=device
        ).to(self.device)

        self.params_t_nn = MLP(
            input_dim=self.inv_covar_dim, 
            output_dim=output_dim_prior_nn, 
            hidden_dim=hidden_dim_prior, 
            n_layers=n_layers_prior, 
            activation=activation, 
            device=device
        ).to(self.device)
        
        self.params_t_suff = MLP(
            input_dim=self.inv_covar_dim, 
            output_dim=2*self.latent_dim_inv, 
            hidden_dim=hidden_dim_prior, 
            n_layers=n_layers_prior, 
            activation=activation, 
            device=device
        ).to(self.device)

        # Factorized prior for spurious latent space
        # Logic: factorized prior is modeled as a multivariate normal distribution with diagonal covariance (i.e. independent components) for this model
        if self.fix_mean_spur_prior and self.fix_var_spur_prior:
            self.prior_mean_spur = torch.zeros(1).to(device)

            self.logl_spur = torch.ones(1).to(device)
        elif self.fix_mean_spur_prior:
            self.prior_mean_spur = torch.zeros(1).to(device)

            if self.spur_covar_dim != 0:
                self.logl_spur = MLP(self.spur_covar_dim, self.latent_dim_spur, hidden_dim_prior, n_layers_prior, activation=activation, slope=slope, device=device).to(device)
            else:
                self.logl_spur = torch.ones(1).to(device)
        elif self.fix_var_spur_prior:
            if self.spur_covar_dim != 0:
                self.prior_mean_spur = MLP(self.spur_covar_dim, self.latent_dim_spur, hidden_dim_prior, n_layers_prior, activation=activation, slope=slope, device=device).to(device)
            else:
                self.prior_mean_spur = torch.zeros(1).to(device)

            self.logl_spur = torch.ones(1).to(device)
        elif (not self.fix_mean_spur_prior) and (not self.fix_var_spur_prior):
            # use one layer less for shared NN such that the whole prior NN has n_layers (end with act for first part)
            
            ## Noise latent prior
            if self.spur_covar_dim != 0:
                self.prior_nn_spur = MLP(
                    self.spur_covar_dim, hidden_dim_prior, hidden_dim_prior, n_layers_prior-1, activation=activation, slope=slope, device=device, end_with_act=True
                ).to(device)

                self.prior_mean_spur = nn.Linear(hidden_dim_prior, self.latent_dim_spur, device=device)
                self.logl_spur = nn.Linear(hidden_dim_prior, self.latent_dim_spur, device=device)
            else:
                self.prior_mean_spur = torch.zeros(1).to(device)
        
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

    def decode(self, z):
        if self.decoder_dist == 'normal':
            decoder_mean = self.decoder_mean(z)

            return decoder_mean
        elif self.decoder_dist == 'nb':
            decoder_raw_mean = self.decoder_raw_mean(z)
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
            logl_spur = (self.logl_spur(spur_covar).exp() + 1e-4)  if (spur_covar is not None) else self.logl_spur

            return self.prior_mean_spur, logl_spur
        elif self.fix_var_spur_prior:
            prior_mean_spur = self.prior_mean_spur(spur_covar) if (spur_covar is not None) else self.prior_mean_spur

            return prior_mean_spur, self.logl_spur
        elif (not self.fix_mean_spur_prior) and (not self.fix_var_spur_prior):
            if spur_covar is not None:
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

    def get_log_decoder_density(self, x, z):
        if self.decoder_dist == 'normal':
            decoder_mean = self.decode(z)

            log_px_z = log_normal(x, decoder_mean, self.decoder_var)
        elif self.decoder_dist == 'nb':
            # decoder_mean are only the freq
            decoder_freq, decoder_theta = self.decode(z)

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
            decoder_mean = self.decode(z)

            return decoder_mean, latent_mean, latent_logvar, z
        elif self.decoder_dist == 'nb':
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

        log_pz_e_spur = log_normal(
            z[:, self.latent_dim_inv:], 
            prior_mean_spur * torch.ones(prior_var_spur.shape, device = self.device), # need for shape match (could change check in log_normal function)
            prior_var_spur
        )
        
        if self.reg_sm == 0:
            sm_part = (d2prior_d2z + torch.mul(0.5, torch.pow(dprior_dz, 2))).sum(dim=1).mean()
        else:
            sm_part = (d2prior_d2z + torch.mul(0.5, torch.pow(dprior_dz, 2)) + d2prior_d2z.pow(2).mul(self.reg_sm)).sum(dim=1).mean()

        objective_function = (
            (log_px_z + log_pz_d_inv_copy + log_pz_e_spur - log_qz_xde).mean().div(self.normalize_constant)   - 
            sm_part
        )

        return objective_function, z

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
            batch_key = 'batch',
            label_key = 'cell_type',
            inv_covar_keys = inv_covar_keys,
            spur_covar_keys = spur_covar_keys,
            latent_dim = 10, 
            latent_dim_spur = 1,
            n_layers = 2, 
            hidden_dim = 128,
            activation = 'relu', 
            slope = 0.1, # only needed when activation is Leaky ReLU ('lrelu')
            device = 'cpu',
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
            batch_key = 'batch',
            label_key = 'cell_type',
            inv_covar_keys = inv_covar_keys,
            spur_covar_keys = spur_covar_keys,
            latent_dim = 10, 
            latent_dim_spur = 1,
            n_layers = 2, 
            hidden_dim = 128,
            activation = 'relu', 
            slope = 0.1, # only needed when activation is Leaky ReLU ('lrelu')
            device = 'cpu',
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
            batch_key = 'batch',
            label_key = 'cell_type',
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
            batch_key = 'batch',
            label_key = 'cell_type',
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

if __name__ == '__main__':
    debug_models(model_name = 'NFinVAE', decoder_dist = 'nb', device = 'cpu')
    #debug_saving_loading(model_name = 'NFinVAE', decoder_dist = 'nb', device_from = 'cpu', device_to = 'cpu', only_loading = False)
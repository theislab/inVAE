from typing import Union
from numbers import Integral

import numpy as np
import torch
from torch import distributions as dist
from torch import nn

from scipy import sparse

def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)

# Implementation from scanpy: see https://github.com/scverse/scanpy/blob/ed3b277b2f498e3cab04c9416aaddf97eec8c3e2/scanpy/_utils/__init__.py#L487
def check_nonnegative_integers(X: Union[np.ndarray, sparse.spmatrix]):
    """Checks values of X to ensure it is count data"""

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
    #if mu.shape[1] != v.shape[0] and mu.shape != v.shape:
    #    raise ValueError(f'The mean and variance vector do not have the same shape:\n\tmean: {mu.shape}\tvariance: {v.shape}')

    logpdf = -0.5 * (np.log(2 * np.pi) + v.log() + (x - mu).pow(2).div(v))

    if reduce:
        logpdf = logpdf.sum(dim=-1)

    return logpdf

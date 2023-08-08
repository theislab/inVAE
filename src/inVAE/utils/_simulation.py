from typing import List
from itertools import product

import numpy as np
import pandas as pd
from anndata import AnnData

from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LinearRegression

def get_linear_score(x, y):
    reg = LinearRegression().fit(x, y)
    return reg.score(x, y)

def mcc(x, y, method='pearson'):
    """
    A numpy implementation of the mean correlation coefficient metric.
    :param x: numpy.ndarray
    :param y: numpy.ndarray
    :param method: str, optional
            The method used to compute the correlation coefficients.
                The options are 'pearson' and 'spearman'
                'pearson':
                    use Pearson's correlation coefficient
                'spearman':
                    use Spearman's nonparametric rank correlation coefficient
    :return: float
    """
    
    d = x.shape[1]
    if method == 'pearson':
        cc = np.corrcoef(x, y, rowvar=False)[:d, d:]
    elif method == 'spearman':
        cc = spearmanr(x, y)[0][:d, d:]
    else:
        raise ValueError('not a valid method: {}'.format(method))
        
    cc = np.abs(cc)
    score = cc[linear_sum_assignment(-1 * cc)].mean()
    return score

# code adapted from lachapelle et al. (their code assumes x_dim = z_dim = h_dim)
def prepare_params_decoder(x_dim, z_dim, h_dim=40, neg_slope=0.2):
    if z_dim > h_dim or h_dim > x_dim:
        raise ValueError("CHECK dim <= h_dim <= x_dim")
    # sampling NN weight matrices
    W1 = np.random.normal(size=(z_dim, h_dim))
    W1 = np.linalg.qr(W1.T)[0].T
    W1 *= np.sqrt(2 / (1 + neg_slope**2)) * np.sqrt(2.0 / (z_dim + h_dim))

    W2 = np.random.normal(size=(h_dim, h_dim))
    W2 = np.linalg.qr(W2.T)[0].T
    # print("distance to identity:", np.max(np.abs(np.matmul(W2, W2.T) - np.eye(h_dim)))
    W2 *= np.sqrt(2 / (1 + neg_slope**2)) * np.sqrt(2.0 / (2 * h_dim))

    W3 = np.random.normal(size=(h_dim, h_dim))
    W3 = np.linalg.qr(W3.T)[0].T
    # print("distance to identity:", np.max(np.abs(np.matmul(W3, W3.T) - np.eye(h_dim))))
    W3 *= np.sqrt(2 / (1 + neg_slope**2)) * np.sqrt(2.0 / (2 * h_dim))

    W4 = np.random.normal(size=(h_dim, x_dim))
    W4 = np.linalg.qr(W4.T)[0].T
    # print("distance to identity:", np.max(np.abs(np.matmul(W4, W4.T) - np.eye(h_dim))))
    W4 *= np.sqrt(2 / (1 + neg_slope**2)) * np.sqrt(2.0 / (x_dim + h_dim))
    return {"W1": W1, "W2": W2, "W3": W3, "W4": W4}


def decoder(z, params, neg_slope=0.2):
    W1, W2, W3, W4 = params["W1"], params["W2"], params["W3"], params["W4"]
    # note that this decoder is almost surely invertible WHEN dim <= h_dim <= x_dim
    # since Wx is injective
    # when columns are linearly indep, which happens almost surely,
    # plus, composition of injective functions is injective.
    h1 = np.matmul(z, W1)
    h1 = np.maximum(neg_slope * h1, h1)  # leaky relu
    h2 = np.matmul(h1, W2)
    h2 = np.maximum(neg_slope * h2, h2)  # leaky relu
    h3 = np.matmul(h2, W3)
    h3 = np.maximum(neg_slope * h3, h3)  # leaky relu
    logits = np.matmul(h3, W4)
    logits /= np.std(logits)
    e_x = np.exp(logits - np.max(logits))
    return e_x / e_x.sum(axis=0)

def synthetic_data(
    n_cells_per_comb: int = 100,
    #n_batches: int = 3,
    n_cell_types: int = 2,
    n_conditions: int = 3,
    n_latent_inv: int = 3,
    n_latent_spur: int = 2,
    n_genes: int = 100,
    shift_cell_type: List[float] = [-20, 20],
    shift_conditions: List[float] = [0, 5, 5.2],
    mean_batch: List[float] = [0],
    var_batch: List[float] = [0, 1, 2, 3, 4, 5],
    verbose: bool = True,
    correlate: bool = True
):
    if var_batch is None:
        var_batch = range(n_conditions * n_cell_types)
        
    n_latents = n_latent_inv + n_latent_spur

    z = np.zeros((n_conditions * n_cell_types * n_cells_per_comb, n_latents))

    obs = np.zeros((n_conditions * n_cell_types * n_cells_per_comb, 3))

    obs_df = pd.DataFrame(obs, columns=['batch','cell_type','disease'])

    tmp = 0

    if correlate:
        var_tmp = np.eye(n_latents) + np.eye(n_latents, k=1) + np.eye(n_latents, k=-1)
        var_tmp[n_latent_inv:, n_latent_inv:] = 0
        var_tmp[range(n_latent_inv,n_latents), range(n_latent_inv,n_latents)] = 1# + var_batch[tmp]
    else:
        var_tmp = np.eye(n_latents)

    for cond in range(n_conditions):
        #generate n_cell_per_comb cells for each cell type:
        for cell in range(n_cell_types):
            inv = [np.array(shift_cell_type[cell])+np.array(shift_conditions[cond])]*n_latent_inv
            spur = mean_batch*n_latent_spur

            if verbose:
                print(inv+spur)
                print(var_tmp)
                print()

            z[
                (tmp * n_cells_per_comb) : ((tmp+1) * n_cells_per_comb) 
            ] = np.random.multivariate_normal(
                inv+spur, var_tmp, size=(n_cells_per_comb)
            )

            obs_df.iloc[(tmp * n_cells_per_comb) : ((tmp+1) * n_cells_per_comb),0] = 0
            obs_df.iloc[(tmp * n_cells_per_comb) : ((tmp+1) * n_cells_per_comb),1] = cell
            obs_df.iloc[(tmp * n_cells_per_comb) : ((tmp+1) * n_cells_per_comb),2] = cond

            tmp+=1

    # finally, get the decoder, and get gene expression x for the cells
    params = prepare_params_decoder(n_genes, n_latents)
    x = decoder(z, params=params)
    x = np.random.poisson(lam=1e6 * x)

    # shuffle dataset
    ind = np.random.permutation(np.arange(obs_df.shape[0]))

    # dump into anndata
    adata = AnnData(x, dtype=np.float32)
    adata.obs = obs_df
    adata.obsm["groundtruth_latent"] = z
    #adata.uns["prior_mean"] = action_specific_prior_mean
    adata = adata[ind]

    for key in adata.obs:
        adata.obs[key] = pd.Categorical(adata.obs[key])
        adata.obs[key] = adata.obs[key].astype(str)

    adata.obsm['groundtruth_latent_inv'] = adata.obsm['groundtruth_latent'][:,:n_latent_inv]
    adata.obsm['groundtruth_latent_spur'] = adata.obsm['groundtruth_latent'][:,n_latent_inv:]
    adata.layers['raw'] = adata.X.copy()

    return adata

def sparse_shift(
    n_cells_per_comb: int = 500,
    n_batches: int = 3,
    n_cell_types: int = 2,
    n_disease_states: int = 2,
    n_latent_inv: int = 9,
    n_latent_spur: int = 1,
    n_genes: int = 100,
    shift_cell_type: List[float] = [5., 2.],
    shift_disease: List[float] = [0., 1.],
    mean_batch: List[float] = [0., 1. , 2.],
    var_batch: List[float] = [1., 2., 3.],
) -> AnnData:
    np.random.seed(0)

    n_latent = n_latent_inv + n_latent_spur
    combs = list(product(range(n_cell_types), range(n_disease_states)))
    n_combs = len(combs)

    # start by generating target mean for every combination of cell type and disease state
    targets = np.ones((n_combs, n_latent))

    # Set the mean of the spurious latents to zero
    targets[:,n_latent_inv:] = 0

    # now generate shifts and apply mask to get target effects
    #shift_sign = (
    #    2 * (np.random.uniform(0, 1, size=targets.shape) > 0.5).astype(float) - 1
    #)
    shift_sign = 1

    # Absolute shift defined per cell type and disease
    shift_abs = np.zeros(shape=targets.shape)
    for i, comb in enumerate(combs):
        shift_cell_tmp = shift_cell_type[comb[0]]
        shift_disease_tmp = shift_disease[comb[1]]
        shift_abs[i,:n_latent_inv] = np.random.normal(shift_cell_tmp + shift_disease_tmp, 0.5, size=(1, n_latent_inv))

    action_specific_prior_mean = targets * shift_abs * shift_sign

    # now we can go around simulating cells and z for all those cells
    z = np.zeros((n_batches * n_combs * n_cells_per_comb, n_latent))
    cells_per_batch = n_combs * n_cells_per_comb
    for batch in range(n_batches):
        action_specific_prior_mean[:, n_latent_inv:] = mean_batch[batch]
        print(f'Mean for experiment:\n{action_specific_prior_mean}')
        var_tmp = np.eye(n_latent)
        var_tmp[range(n_latent_inv, n_latent), range(n_latent_inv, n_latent)] += var_batch[batch]
        for comb in range(n_combs):
            z[
                (batch * cells_per_batch) + (comb * n_cells_per_comb) : (batch * cells_per_batch) + ((comb+1) * n_cells_per_comb) 
            ] = np.random.multivariate_normal(
                action_specific_prior_mean[comb], var_tmp, size=(n_cells_per_comb)
            )

    # finally, get the decoder, and get gene expression x for the cells
    params = prepare_params_decoder(n_genes, n_latent)
    x = decoder(z, params=params)
    x = np.random.poisson(lam=1e6 * x)
    # put labels in these
    cell_types = np.concatenate(n_batches * [np.concatenate([n_cells_per_comb * [c[0]] for c in combs])]) 
    disease = np.concatenate(n_batches * [np.concatenate([n_cells_per_comb * [c[1]] for c in combs])])
    batches = np.concatenate([cells_per_batch * [b] for b in range(n_batches)])

    # shuffle dataset
    ind = np.random.permutation(np.arange(n_batches * cells_per_batch))
    x = x[ind]
    cell_types = cell_types[ind]
    disease = disease[ind]
    batches = batches[ind]
    z = z[ind]

    # dump into anndata
    adata = AnnData(x, dtype=np.float32)
    adata.obs["batch"] = pd.Categorical(batches)
    adata.obs["cell_type"] = pd.Categorical(cell_types)
    adata.obs["disease"] = pd.Categorical(disease)
    adata.obsm["groundtruth_latent"] = z
    adata.uns["prior_mean"] = action_specific_prior_mean
    return adata
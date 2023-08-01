# inVAE

inVAE is a conditionally invariant variational autoencoder that identifies both spurious (distractors) and invariant features. 
It leverages domain variability to learn conditionally invariant representations. We show that inVAE captures biological variations in single-cell datasets obtained from diverse conditions and labs. 
inVAE incorporates biological covariates and mechanisms such as disease states, to learn an invariant data representation. This improves cell classification accuracy significantly. 

![logo](./images/inVAE_black.png|width=20px)


## Installation

1. PyPI only <br/> 
```pip install invae```<br/>

2. Development Version (latest version on github) <br/>
```git clone https://github.com/theislab/inVAE```<br/>
```pip install .```<br/>

## Example

[Integration of Human Lung Cell Atlas using both healthy and disease samples](https://github.com/theislab/inVAE/blob/master/notebooks/inVAE_LungAtlas.ipynb)


## Dependencies

* scanpy==1.9.3
* torch==2.0.1
* tensorboard==2.13.0
* anndata==0.8.0


## Citation

[H. Aliee, F. Kapl, S. Hediyeh-Zadeh, F. J. Theis, Conditionally Invariant Representation Learning for Disentangling Cellular Heterogeneity, 2023](https://arxiv.org/abs/2307.00558)


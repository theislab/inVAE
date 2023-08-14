from typing import List, Literal, Optional, Dict

import time
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F

from torch.utils.tensorboard import SummaryWriter

import scanpy as sc
from anndata.experimental.pytorch import AnnLoader

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from abc import ABC, abstractmethod
from anndata import AnnData

from ..utils import check_nonnegative_integers, ModularMultiClassifier

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
        verbose: bool = True,
    ):  
        if verbose:
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
        latent_type: Literal['full', 'invariant', 'inv', 'spurious', 'spur'] = 'invariant',
        verbose: bool = True,
    ):
        if adata is None:
            if verbose:
                print('Using saved adata for latent representation!')

            self.module.eval()

            transformed_data = self.transformed_data
        else:
            if verbose:
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

            if latent_type == 'full':
                # Do nothing: whole latente space
                pass
            elif (latent_type == 'invariant') or (latent_type == 'inv'):
                latent_mean = latent_mean[:, :self.latent_dim_inv]
            elif (latent_type == 'spurious') or (latent_type == 'spur'):
                latent_mean = latent_mean[:, self.latent_dim_inv:]
            else:
                print(f"{latent_type} is not a valid type of latent representation! Type has to be in ['full', 'invariant', 'inv', 'spurious', 'spur'].")

        return latent_mean.detach().cpu().numpy()
     
    def predict(
        self,
        adata: AnnData,
        dataset_type: Literal['train', 'val', 'validation', 'test'] = 'train',
    ):
        self.classifier.eval()

        if dataset_type == 'train':
            print('Using latent representation of the adata the model was trained on. Make sure you have trained the classifier before!')
            latent_inv_train = torch.tensor(self.get_latent_representation(latent_type = 'inv', verbose = False), device=self.device)
            prediction = self.classifier(latent_inv_train)
        elif 'val' in dataset_type:
            print('Using saved sampled latent representation for validation adata. Make sure you have trained the classifier before!')
            sampled_latent_val = self.saved_latent['val']
            sampled_latent_inv_val = sampled_latent_val[:, :self.latent_dim_inv]

            if adata is not None:
                assert adata.n_obs == sampled_latent_inv_val.shape[0]

            prediction = self.classifier(sampled_latent_inv_val)
        elif 'test' in dataset_type:
            if 'test' in self.saved_latent:
                print('Using saved sampled latent representation for test adata. Make sure you have trained the classifier before!')
                sampled_latent_test = self.saved_latent['test']
                sampled_latent_inv_test = sampled_latent_test[:, :self.latent_dim_inv]
            else:
                print('Sampling latent representation for test adata. This may take a while. Make sure you have trained the classifier before!')
                
                # Put data on right device and apply transforms (one-hot encoder, ...)
                use_cuda = (self.device == 'cuda')
                data_loader = AnnLoader(adata, batch_size = 1, shuffle = False, use_cuda = use_cuda, convert = self.data_loading_encoders)
                transformed_data = data_loader.dataset[:]

                # Assign x, z, y, nr_samples and (if necessary) invariant and spurious covariates
                x_test = transformed_data.layers[self.layer] if self.layer is not None else transformed_data.X
                latent_train = torch.tensor(self.get_latent_representation(latent_type = 'full', verbose = False), device=self.device)

                encoder_label = self.dict_encoders[f'label_{self.label_key}']
                label_tensor_test = torch.tensor(encoder_label.transform(adata.obs[self.label_key]), device = self.device)

                nr_samples = self.optimize_latent_dict['nr_samples']
                
                # Covariates for invariate prior
                inv_tensor_list = [transformed_data.obs[covar].float() for covar in self.list_inv_covar]
                inv_covar = torch.cat(inv_tensor_list, dim = 1) if (self.inv_covar_dim != 0) else None

                # Covariates for spurious prior
                spur_tensor_list = [transformed_data.obs[covar].float() for covar in self.list_spur_covar]
                spur_covar = torch.cat(spur_tensor_list, dim = 1) if (self.spur_covar_dim != 0) else None

                # sampling and optimizing latent for test
                init_z_test = self._sample_latents_for_prediction(
                    x = x_test,
                    latent_sample = latent_train,
                    nr_samples = nr_samples,
                    inv_covar = inv_covar,
                    spur_covar = spur_covar
                )

                final_z_test, _ = self._optimize_latents_for_prediction(
                    x = x_test, # dim: nr_obs_test x genes
                    init_z = init_z_test, # dim: nr_obs_test x latent_dim
                    label_tensor = label_tensor_test, # dim: nr_obs_test x 1
                    inv_covar = inv_covar,
                    spur_covar = spur_covar
                )

                self.saved_latent['test'] = final_z_test.clone()

                sampled_latent_inv_test = final_z_test[:, :self.latent_dim_inv]
            
            prediction = self.classifier(sampled_latent_inv_test)

        # Transform back to string representation of labels
        prediction = prediction.argmax(dim = 1, keepdim = True)
        prediction = self.dict_encoders[f'label_{self.label_key}'].inverse_transform(prediction.view(-1).detach().cpu().numpy())
        return prediction

    def train_classifier(
        self,
        adata_val: AnnData,
        batch_key: Optional[str] = None,
        label_key: Optional[str] = None,
        n_epochs_train_class: int = 500,
        n_epochs_opt_val: int = 100,
        nr_samples: int = 100,
        hidden_dim_class: int = 50,
        n_layers_class: int = 1,
        act_class: Literal['relu', 'lrelu'] = 'relu',
        lr_train_class: float = 0.01,
        lr_opt_val: float = 0.001,
        class_print_every_n_epochs: int = None,
        opt_val_print_every_n_epochs: int = None
    ):
        # Save batch and label_key
        self.batch_key = batch_key
        self.label_key = label_key

        if class_print_every_n_epochs is None:
            class_print_every_n_epochs = n_epochs_train_class / 5

        if opt_val_print_every_n_epochs is None:
            opt_val_print_every_n_epochs = n_epochs_opt_val / 5

        val_loader = AnnLoader(adata_val, batch_size = 1, shuffle = False, use_cuda = (self.device == 'cuda'), convert = self.data_loading_encoders)

        # Assign x_val, and (if necessary) invariant and spurious covariates
        full_data_val = val_loader.dataset[:]
        x_val = full_data_val.layers[self.layer] if self.layer is not None else full_data_val.X

        # Covariates for invariate prior
        inv_tensor_list = [full_data_val.obs[covar].float() for covar in self.list_inv_covar]
        inv_covar_val = torch.cat(inv_tensor_list, dim = 1) if (self.inv_covar_dim != 0) else None

        # Covariates for spurious prior
        spur_tensor_list = [full_data_val.obs[covar].float() for covar in self.list_spur_covar]
        spur_covar_val = torch.cat(spur_tensor_list, dim = 1) if (self.spur_covar_dim != 0) else None

        self.module.eval()

        nr_val_obs = adata_val.n_obs
        latent_dim = self.latent_dim
        latent_dim_inv = self.latent_dim_inv

        # Get invariant latent space representation
        latent_sample = torch.tensor(self.get_latent_representation(self.adata, latent_type='full'), device=self.device).view(-1, latent_dim)

        # Set-up encoders for batches and labels
        
        # Label Encoder for cell_types (or in general: label key)
        if label_key is not None:
            encoder_label = LabelEncoder()
            encoder_label.fit(self.adata.obs[label_key])

            self.dict_encoders[f'label_{label_key}'] = encoder_label
        else:
            raise ValueError('Please specify the key for the labels in order to train the classifier!')

        encoder_label = self.dict_encoders[f'label_{self.label_key}']
        label_tensor_train = torch.tensor(encoder_label.fit_transform(self.adata.obs[self.label_key]), device = self.device)
        label_tensor_val = torch.tensor(encoder_label.transform(adata_val.obs[self.label_key]), device = self.device)

        encoder_batch = LabelEncoder()
        batch_tensor_train = torch.tensor(encoder_batch.fit_transform(self.adata.obs[self.batch_key]), device = self.device)

        if len(encoder_label.classes_) == 1:
            print('Warning: There is only one batch in the (training) data. This model might not behave as intended.')
            print('Consider using the whole latent space (option: type = "full") as a representation instead of only the invariant latent space!')

        print('Starting sampling of latents for the val data...')
        
        # Initialzing latent samples for val data via samples of training data
        init_z = self._sample_latents_for_prediction(
            x = x_val,
            latent_sample = latent_sample,
            nr_samples = nr_samples,
            inv_covar = inv_covar_val,
            spur_covar = spur_covar_val
        )

        print('Sampling done!')

        hp_dict_class = {
            'hidden_dim': hidden_dim_class,
            'n_layers': n_layers_class,
            'activation': act_class
        }
        
        classifier = ModularMultiClassifier(
            input_dim = self.latent_dim_inv, 
            n_classes = len(encoder_label.classes_),
            **hp_dict_class,
            device = self.device
        ).to(self.device)

        target = label_tensor_train.view((-1, 1))
        
        env_tensor = batch_tensor_train.view(-1, 1)

        unique_env = torch.unique(env_tensor)
        
        envs = []
        
        for e in unique_env:
            envs.append({
                'latent': latent_sample[(env_tensor == e).view(-1), :latent_dim_inv],
                'target': target[(env_tensor == e).view(-1)]
            })
        
        optimizer_train = optim.Adam(classifier.parameters(), lr = lr_train_class)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_train, factor=0.1, patience=50, verbose=True)

        #LOG_DIR = 
        
        #test_writer = SummaryWriter(log_dir=LOG_DIR)
        
        classifier.train()
        
        for index_train in range(n_epochs_train_class):
            
            for env in envs:
                env['pred'] = classifier(env['latent'].float())
                env['loss'] = F.nll_loss(env['pred'].float(), env['target'].long().view(-1))

            output = torch.stack([env['loss'] for env in envs]).mean()
            
            optimizer_train.zero_grad()
            output.backward()
            optimizer_train.step()
            scheduler.step(output)

            if (index_train % class_print_every_n_epochs == 0) or (index_train == (n_epochs_train_class - 1)):
                with torch.no_grad():
                    nll_loss = output.detach().cpu().numpy()
                    
                    #Compute acc
                    for env in envs:
                        pred = env['pred'].argmax(dim = 1, keepdim = True)
                        env['n_correct_pred'] = pred.eq(env['target'].view_as(pred)).sum().item()
                        
                acc = np.sum([env['n_correct_pred'] for env in envs]) / len(target)
                        
                print(f'\tepoch {index_train+1}/{n_epochs_train_class}: nll loss {nll_loss:.2f}\t train_acc {acc:.2f}')

        # Assign classifier to model to use later for test prediction
        self.classifier = classifier
        self.classifier.eval()
        
        with torch.no_grad():
            y_pred_val = classifier(init_z[:, :latent_dim_inv].view(-1, latent_dim_inv)).float().to(self.device)
            y_true = label_tensor_val.view(-1, 1)

            # Calculate accuracy
            pred_val = y_pred_val.argmax(dim = 1, keepdim = True)
            acc_val = pred_val.eq(y_true.view_as(pred_val)).sum().item() / y_true.shape[0]

        print(f'\tThe val acc before optimizing the latents is: {acc_val:.3f}')

        acc_array = np.zeros([n_epochs_opt_val])

        self.optimize_latent_dict = {
            'lr_opt_val': lr_opt_val,
            'n_epochs_opt_val': n_epochs_opt_val,
            'nr_samples': nr_samples
        }

        print('Starting to optimize sampled latents for validation data...')

        # Optimize latents for val data (see function description)

        final_z, acc_array = self._optimize_latents_for_prediction(
            x = x_val, # dim: nr_obs_val x genes
            init_z = init_z, # dim: nr_obs_val x latent_dim
            label_tensor = label_tensor_val, # dim: nr_obs_val x 1
            inv_covar = inv_covar_val,
            spur_covar = spur_covar_val
        )

        self.saved_latent['val'] = final_z.clone()
                    
        print('Optimizing latents for validation data done!')
        
        assert len(final_z) == nr_val_obs

        acc_array = acc_array / nr_val_obs

        for i in range(0, n_epochs_opt_val, opt_val_print_every_n_epochs):
            print(f'\tThe val acc after optimizing {i+1}/{n_epochs_opt_val} is: {acc_array[i]:.3f}')
            
        print(f'\tThe val acc after optimizing the latents is: {acc_array[(n_epochs_opt_val - 1)]:.3f}')

    def train(
        self,
        n_epochs: int = None,
        lr_train: float = None,
        weight_decay: float = None,
        log_dir: str = None,
        log_freq: int = 25, # in iterations
        print_every_n_epochs: int = None,
        checkpoint_dir: str = None,
        n_checkpoints: int = 0,
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

                iteration += 1

                optimizer.zero_grad(set_to_none=True)

                objective_fct, _ = self.module.elbo(x, inv_covar, spur_covar)

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

            if (n_checkpoints > 0) and (iteration % int(max_iter/n_checkpoints) == 0):
                self.save(f'{checkpoint_dir}/checkpoint_epoch_{int(iteration/len(self.data_loader))}.pt')
            
        self.module.eval()
        print('Training done!')

    def _optimize_latents_for_prediction(
        self,
        x: torch.Tensor, # dim: nr_obs x genes
        init_z: torch.Tensor, # dim: nr_obs x latent_dim
        label_tensor: torch.Tensor, # dim: nr_obs x 1
        inv_covar: torch.Tensor = None,
        spur_covar: torch.Tensor = None,
    ):  
        # Extract parameters for optimizing the latents from dict (sha
        lr_opt_val = self.optimize_latent_dict['lr_opt_val']
        n_epochs_opt_val = self.optimize_latent_dict['n_epochs_opt_val']

        # Array for saving accuracies across epochs
        acc_array = np.zeros([n_epochs_opt_val])
        
        latent_dim = self.latent_dim
        latent_dim_inv = self.latent_dim_inv

        for ind in range(x.shape[0]):
            x_tmp, y_tmp, z_tmp = (
                x[ind], 
                label_tensor[ind], 
                init_z[ind]
            )

            if self.module.inject_covar_in_latent:
                spur_tmp = (
                    spur_covar[ind]
                )

            z_tmp.requires_grad_(True)

            optimizer_val = optim.Adam([z_tmp], lr = lr_opt_val, weight_decay = lr_opt_val/10)
                
            for iteration in range(1, (n_epochs_opt_val+1)):
                if self.module.inject_covar_in_latent:
                    log_px_z = self.module.get_log_decoder_density(x_tmp, z_tmp.view(-1, latent_dim), None, spur_tmp)
                else:
                    log_px_z = self.module.get_log_decoder_density(x_tmp, z_tmp.view(-1, latent_dim))

                loss = (
                    log_px_z.sum()
                )
                
                ## Optimizer part
                optimizer_val.zero_grad()
                loss.mul(-1).backward()
                optimizer_val.step()
                
                # Calculate accuracy
                with torch.no_grad():
                    y_pred = self.classifier(z_tmp[:latent_dim_inv].view(-1, latent_dim_inv).float())
                    y_true = y_tmp.view(-1,1)

                    pred = y_pred.argmax(dim = 1, keepdim = True)
                    acc_array[(iteration - 1)] = acc_array[(iteration - 1)] + pred.eq(y_true.view_as(pred)).sum().item()

        return init_z, acc_array

    def _sample_latents_for_prediction(
        self,
        x: torch.Tensor, # dim: nr_obs x genes
        latent_sample: torch.Tensor, # dim: nr_obs_train x latent_dim
        nr_samples: int = 100,
        inv_covar: torch.Tensor = None,
        spur_covar: torch.Tensor = None,
    ):
        # Storing the initially sampled z
        init_z = torch.zeros([x.shape[0], latent_sample.shape[1]], device=self.device)

        # For every cell in the data (i.e. with x the corresponding gene counts) sample from the latent space of the training data (i.e. from latent_sample)
        # to get a latent representation for that sample that closely matches the gene count data in x (highest decoder density of x and z)
        for index_val_example in range(init_z.shape[0]):
            sample_ind = np.random.choice(latent_sample.shape[0], nr_samples, replace = False)

            single_train_z = latent_sample[sample_ind]

            x_tmp, z_tmp = (
                x[index_val_example].expand(nr_samples, -1).to(self.device),
                # Shape: nr_samples x latent_dim
                single_train_z.to(self.device)    
            )

            if self.module.inject_covar_in_latent:
                spur_tmp = (
                    spur_covar[index_val_example].expand(nr_samples, -1)
                )

            if self.module.inject_covar_in_latent:
                log_px_z = self.module.get_log_decoder_density(x_tmp, z_tmp.view(-1, self.latent_dim), None, spur_tmp)
            else:
                log_px_z = self.module.get_log_decoder_density(x_tmp, z_tmp.view(-1, self.latent_dim))

            init_z[index_val_example] = single_train_z[torch.argmax(log_px_z).view(-1)]

        return init_z

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
            # Assuming data is scaled to mean zero and variance one per gene, for decoder distribution "normal"
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

        for key, covars in spur_covar_keys.items():
            if key == 'cont':
                # Do we have to do something?
                pass
            elif key == 'cat':
                for covar in covars:
                    # Default for now is to use one-hot encoding for categorical vars
                    dict_encoders[covar] = OneHotEncoder(sparse_output=False, dtype=np.float32, handle_unknown='ignore')
                    dict_encoders[covar].fit(adata.obs[covar].to_numpy()[:, None])

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

        return (dict_encoders, encoders, data_loader, transformed_data)

from numbers import Number

from torch import nn
from torch.nn import functional as F

from ._helper_functions import weights_init

class MLP(nn.Module):

    def __init__(
        self, 
        input_dim, 
        output_dim, 
        hidden_dim, 
        n_layers, 
        activation='none', 
        slope=0.1, 
        device='cpu', 
        end_with_act = False, 
        batch_norm = False,
        dropout_rate = 0
    ):

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        self.end_with_act = end_with_act
        self.batch_norm = batch_norm
        self.dropout_rate = dropout_rate

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

            if self.dropout_rate > 0:
                _dr_list = [nn.Dropout(p = self.dropout_rate)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0])]
            if self.batch_norm:
                _bn_list = [nn.BatchNorm1d(self.hidden_dim[0], momentum=0.01, eps=0.001)]
            
            if self.dropout_rate > 0:
                _dr_list = [nn.Dropout(p = self.dropout_rate)]

            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i]))
                if self.batch_norm:
                    _bn_list.append(nn.BatchNorm1d(self.hidden_dim[i], momentum=0.01, eps=0.001))
                if self.dropout_rate > 0:
                    _dr_list.append(nn.Dropout(p = self.dropout_rate))

            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim))
            if self.batch_norm:
                _bn_list.append(nn.BatchNorm1d(self.output_dim, momentum=0.01, eps=0.001))
            if self.dropout_rate > 0:
                _dr_list.append(nn.Dropout(p = self.dropout_rate))
            

        self.fc = nn.ModuleList(_fc_list)

        if self.batch_norm:
            self.bn = nn.ModuleList(_bn_list)

        if self.dropout_rate > 0:
            self.dr = nn.ModuleList(_dr_list)

    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)

                if self.batch_norm:
                    h = self.bn[c](h)

                if self.end_with_act:
                    h = self._act_f[c](h)

                if self.dropout_rate > 0:
                    h = self.dr[c](h)
            else:
                h = self.fc[c](h)

                if self.batch_norm:
                    h = self.bn[c](h)

                h = self._act_f[c](h)

                if self.dropout_rate > 0:
                    h = self.dr[c](h)

        return h
    
class ModularMultiClassifier(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, n_layers, n_classes, activation, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_classes = n_classes
        
        self._main = MLP(
            input_dim = input_dim, 
            output_dim = n_classes, 
            hidden_dim = hidden_dim, 
            n_layers = n_layers, 
            activation = activation, 
            device = device
        ).to(device)
        
        self.apply(weights_init)

    def forward(self, input):
        x = self._main(input)
        out = F.log_softmax(x, dim = 1)
        return out
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 
from torch.utils.data import DataLoader
from tqdm import tqdm   


##### Simple fully connected artificial neural network 
class SimpleANN(nn.Module):
    def __init__(self, **kwargs):

        super(SimpleANN, self).__init__()

        ###### Input Hparams ########################
        self.input_dim          =  kwargs['input_dim']
        self.feature_dim        =  kwargs['feature_dim']
        self.network_dimensions =  kwargs['network_dimensions']
        self.feature_dim        =  kwargs['feature_dim']
        self.num_classes        =  kwargs['num_classes']
        self.softplus_beta      =  kwargs.get('nn_softplus_beta',-1)

        self.network_dimensions = [self.input_dim] + [int(x) for x in self.network_dimensions.split("-")] + [self.feature_dim]
        ################################################
        
        self.f = []
        for i in range(len(self.network_dimensions) -1):
            fcl = nn.Linear(self.network_dimensions[i], self.network_dimensions[i+1], bias=True)
            self.f.append(fcl)
            if not i == (len(self.network_dimensions) -2): 
                if self.softplus_beta <= 0:
                    self.f.append(nn.ReLU(inplace = True))                                       
                else:
                    self.f.append(nn.Softplus(beta = self.softplus_beta))
        
        self.f.append(nn.Linear(self.network_dimensions[-1], self.num_classes)) # For regression set num_classes = 1 (or number of preds in case of multiple regr)
        self.f = nn.Sequential(*self.f)

    def forward(self, x):
        # x should be flattened with a single channel (grayscale)
        if len(x.shape) > 2:
            if len(x.shape) == 4:
                # if there is channel information
                x = x[...,0]
            x = torch.flatten(x, 1)
        logits = self.f(x)         
        return logits 

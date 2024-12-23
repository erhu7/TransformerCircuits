import torch
from torch import nn
from torch.nn import functional as F
import einops 

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 

from tqdm.notebook import trange

from dataclasses import dataclass

@dataclass
class Config:
    n_instances = None
    n_features = None
    n_hidden = None


class AutoEncoder:
    def __init__(self,
                   config,
                   feature_sparsity,
                   feature_importance):
        super().__init__()    

        self.config = config
        self.W = nn.Parameter(torch.empty([config.n_instances, config.n_features, config.n_hidden]))
        nn.init.xavier_normal_(self.W)
        self.b = nn.Parameter(torch.zeros([config.n_instances, config.n_features]))
        
        if feature_sparsity is None:
            self.feature_sparsity = torch.ones(())
        else:
            self.feature_sparsity = feature_sparsity
        
        if feature_importance is None:
            self.feature_importance = torch.ones(())
        else:
            self.feature_importance = feature_importance

    def forward(self, X):
        h = torch.einsum("...if,ifh -> ...ih", X, self.W)
        out = torch.einsum("...ih,ifh -> ...if", h, self.W)
        out = F.relu(out + self.b)
        return out

    def generate_batch(self, batch_size):
        batch_dims = [batch_size, self.config.n_instances, self.config.n_features]
        batch = torch.rand(batch_dims)
        return torch.where(torch.rand(batch_dims) <= self.feature_sparsity,
                            batch,
                              0)
    

def constant_lr(*_):
  return 1.0

def optimize(model,
             batch_size=1024,
             steps=10000,
             lr=1e-3,
             hook_freq = 10
             ):
    cfg = model.config
    opt = torch.optim.AdamW(list(model.parameters()), lr=lr)
    
    n_hooks = int(steps/hook_freq)
    losses = np.empty([n_hooks, cfg.n_instances, cfg.n_features])
    weights = np.empty([n_hooks, cfg.n_instances, cfg.n_features, cfg.n_hidden])

    for step in trange(steps):
       opt.zero_grad(set_to_none=True)
       X = model.generate_batch(batch_size)
       X_pred = model(X)
       error = model.feature_importance*torch.pow(X - X_pred, 2)
       # batch error
       loss = einops.reduce(error, "b i f -> i", "mean").sum()
       # backpropagate + update newtork
       loss.backward()
       opt.step()
       if step % hook_freq == 0:
        
        losses[int(step/hook_freq)] = einops.reduce(error, "b i f -> i f", "mean").detach().numpy()
        weights[int(step/hook_freq), ...] = model.W.detach().numpy().copy()


    return dict(losses=losses, weights=weights)
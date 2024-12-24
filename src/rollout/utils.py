import torch
from torch import nn
from torch.nn import functional as F
import einops 
import numpy as np


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


def movmean(data, window_size):
    """
    Calculate the moving mean of a 1D array.

    Parameters:
    data (list or np.array): Input data.
    window_size (int): The size of the moving window.

    Returns:
    np.array: The moving mean of the input data.
    """
    if not isinstance(data, (list, np.ndarray)):
        raise ValueError("Input data should be a list or numpy array")
    
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("Window size should be a positive integer")
    
    data = np.array(data)
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size




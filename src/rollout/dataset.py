import torch
import numpy as np
from itertools import permutations
from torch.utils.data import Dataset
import math

def generate_combinations(num_tokens, sequence_length, n_back, return_output=True):
    # generate all permutations of the first sequence_length-1 elements
    # TODO: Could subsample the permutations to reduce the number of combinations
    all_perms = list(permutations(range(num_tokens), sequence_length-1))
    combinations = torch.zeros(len(all_perms), sequence_length, dtype=torch.long)
    combinations[:,:sequence_length-1] = torch.tensor(all_perms)
    # now set the last element to a random element in the sequence
    prompt_inds = torch.randint(0, sequence_length-1-n_back, (len(combinations),), dtype=torch.long)
    combinations[:,-1] = combinations[np.arange(len(combinations)),prompt_inds]
    if return_output:
        output_inds = prompt_inds + n_back
        output = combinations[np.arange(len(combinations)),output_inds]
        return combinations, output
    return combinations
        
class InductionDataset:
    # A dataset class for generating sequences with induction patterns.
    # TODO could subclass torch.utils.data.Dataset for more flexibility
    def __init__(self, num_tokens, sequence_length, n_back=1, random_seed=42, train_fraction=0.8):
        """
        Initializes the InductionDataset with the given parameters.
        Args:
            num_tokens (int): Total number of unique tokens.
            sequence_length (int): Length of each sequence.
            n_back (int, optional): Number of steps back to look for the induction pattern. Defaults to 1.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
            train_fraction (float, optional): Fraction of data to be used for training. Defaults to 0.8.
        """
        torch.manual_seed(random_seed)
        assert num_tokens > sequence_length, "num_tokens must be greater than sequence_length"
        assert n_back < sequence_length-1, "n_back must be less than sequence_length-1"
        
        
        self.n = num_tokens
        self.n_back = n_back

        self.X, self.y = generate_combinations(num_tokens, sequence_length, n_back)
        shuffle_idx = torch.randperm(len(self.X))
        self.X = self.X[shuffle_idx]
        self.y = self.y[shuffle_idx]
        self.n_samples = len(self.X)

        self.n_train = int(self.n_samples * 0.8)
        self.train_idx = torch.arange(self.n_train)

        self.test_idx = torch.arange(self.n_train, self.n_samples)
        self.n_test = len(self.test_idx)

    def __len__(self):
        return self.n_samples
    
    def generate_combinations(self, num_tokens=None, sequence_length=None, n_back=None):
        # generate all permutations of the first sequence_length-1 elements
        # TODO: Could subsample the permutations to reduce the number of combinations
        if num_tokens is None:
            num_tokens = self.n
        if sequence_length is None:
            sequence_length = self.sequence_length
        if n_back is None:
            n_back = self.n_back
        all_perms = list(permutations(range(num_tokens), sequence_length-1))
        combinations = torch.zeros(len(all_perms), sequence_length, dtype=torch.long)
        combinations[:,:sequence_length-1] = torch.tensor(all_perms)
        # now set the last element to a random element in the sequence
        prompt_inds = torch.randint(0, sequence_length-1-n_back, (len(combinations),))
        combinations[:,-1] = combinations[np.arange(len(combinations)),prompt_inds]
        output_inds = prompt_inds + n_back
        output = combinations[np.arange(len(combinations)),output_inds]
        return combinations, output

    def generate_batch(self, batch_size, type='train'):
        """
        Generates a batch of data for training or testing.
        Args:
            batch_size (int): Number of samples in the batch.
            type (str, optional): Type of data to generate ('train' or 'test'). Defaults to 'train'.
        Returns:
            tuple: A tuple containing the input sequences (X) and the output sequences (y).
        """
        assert type in ['train', 'test'], "type must be either 'train' or 'test'"
        if type == 'train':
            idx = self.train_idx[torch.randint(0, self.n_train, (batch_size,))]
        else:
            idx = self.test_idx[torch.randint(0, self.n_test, (batch_size,))]
        X = self.X[idx]
        y = self.y[idx]
        return X, y
    



class MultiTaskDataset(InductionDataset):
    # A dataset class for generating sequences with induction patterns.
    # TODO could subclass torch.utils.data.Dataset for more flexibility
    def __init__(self, num_tokens, sequence_length, n_backs=[1,2], random_seed=42, train_fraction=0.8):
        """
        Initializes the InductionDataset with the given parameters.
        Args:
            num_tokens (int): Total number of unique tokens.
            sequence_length (int): Length of each sequence.
            n_back (int, optional): Number of steps back to look for the induction pattern. Defaults to 1.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
            train_fraction (float, optional): Fraction of data to be used for training. Defaults to 0.8.
        """
        torch.manual_seed(random_seed)
        assert num_tokens > sequence_length, "num_tokens must be greater than sequence_length"
        assert np.all([n_back < sequence_length-1 for n_back in n_backs]), "n_back must be less than sequence_length-1"
        
        self.n = num_tokens
        self.n_backs = n_backs
        self.sequence_length = sequence_length
        self.X = []
        self.y = []
        for ix,n_back in enumerate(n_backs):
            _X, y = self.generate_combinations(num_tokens, sequence_length-1, n_back)
            X = torch.zeros((_X.shape[0], sequence_length), dtype=torch.long)
            X[:,0] = ix + num_tokens # context should be out of range
            X[:,1:] = _X

            self.X.append(X)
            self.y.append(y)
        self.X = torch.cat(self.X)
        self.y = torch.cat(self.y)

        shuffle_idx = torch.randperm(len(self.X))
        self.X = self.X[shuffle_idx]
        self.y = self.y[shuffle_idx]
        self.n_samples = len(self.X)

        self.n_train = int(self.n_samples * 0.8)
        self.train_idx = torch.arange(self.n_train)

        self.test_idx = torch.arange(self.n_train, self.n_samples)
        self.n_test = len(self.test_idx)

    

class InductionValueDataset(Dataset):
    def __init__(self, num_tokens, sequence_length, random_seed=42):
        assert sequence_length <= num_tokens, "sequence_length must be less than or equal to num_tokens"

        torch.manual_seed(random_seed)
        self.num_tokens = num_tokens
        self.sequence_length = sequence_length

        
        X, y = generate_combinations(num_tokens, sequence_length, 1, return_output=True)
        self.n_samples = X.shape[0]

        self.data = X

        self.task_tokens = dict(induction=num_tokens,
                                value=num_tokens+1)
        self.labels = dict(induction=y,
                           value=torch.max(X, dim=1).values)
        # set the second half of the data to be the value task
        del X, y 

        shuffle_idx = torch.randperm(self.n_samples)

        self.n_train = int(self.n_samples * 0.8)
        self.train_idx = torch.arange(self.n_train)

        self.test_idx = torch.arange(self.n_train, self.n_samples)
        self.n_test = len(self.test_idx)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx, task=None):
        assert task in [None, "value", "induction"], "task must be None, value, or induction. If None, a task will be randomly selected."
        if task is None:
            task = np.random.choice(["value", "induction"])
        if type(idx == int):
            n_items = 1
        else:
            n_items = len(idx)
        sample = torch.zeros(n_items, self.sequence_length + 1)
        sample[:, 1:] = self.data[idx]
        sample[:, 0] = self.task_tokens[task]
        label = self.labels[task][idx]
        
        return sample, label
    
    def generate_batch(self, batch_size, type='train', task=None):
        """
        Generates a batch of data for training or testing.
        Args:
            batch_size (int): Number of samples in the batch.
            type (str, optional): Type of data to generate ('train' or 'test'). Defaults to 'train'.
        Returns:
            tuple: A tuple containing the input sequences (X) and the output sequences (y).
        """
        #TODO: implment interleaved tasks in training
        assert task in [None, "value", "induction"], "task must be None, value, or induction. If None, a task will be randomly selected."
        
        if task is None:
            task = np.random.choice(["value", "induction"])

        assert type in ['train', 'test'], "type must be either 'train' or 'test'"
        if type == 'train':
            idx = self.train_idx[torch.randint(0, self.n_train, (batch_size,))]
        else:
            idx = self.test_idx[torch.randint(0, self.n_test, (batch_size,))]
        task_column = torch.ones(batch_size, 1, dtype=torch.long)*self.task_tokens[task]
        X = torch.cat((task_column, self.data[idx]), dim=1)
        y = self.labels[task][idx]
        return X, y
    
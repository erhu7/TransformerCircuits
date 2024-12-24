import torch
import torch.nn as nn
import numpy as np


class PositionalEmbedding(nn.Module):
    """
    Positional Embedding layer for transformer models.

    This layer generates sinusoidal positional embeddings as described in the 
    "Attention is All You Need" paper. The embeddings are precomputed and stored 
    in a tensor, which is indexed during the forward pass.

    Args:
        num_positions (int): The maximum number of positions (sequence length) to embed.
        embedding_dim (int): The dimension of the embedding vector.
        batch_first (bool, optional): If True, the input and output tensors are provided 
                                      as (batch, sequence, feature). Default is True.
        *args: Additional arguments for the parent class.
        **kwargs: Additional keyword arguments for the parent class.

    Attributes:
        d_model (int): The dimension of the embedding vector.
        batch_first (bool): Indicates whether the input and output tensors are provided 
                            as (batch, sequence, feature).
        embedding (torch.Tensor): The precomputed positional embeddings.

    Methods:
        forward(x):
            Computes the positional embeddings for the input tensor `x`.

            Args:
                x (torch.Tensor): The input tensor of shape (batch, sequence, feature) 
                                  if `batch_first` is True, otherwise (sequence, batch, feature).

            Returns:
                torch.Tensor: The positional embeddings corresponding to the input tensor `x`.
    """
    def __init__(self, num_positions, embedding_dim, batch_first=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_model = embedding_dim
        self.batch_first = batch_first
        denominators = (10000)**(torch.arange(0, embedding_dim, 2)/embedding_dim)
        positions = torch.arange(0, num_positions)[:, None] / denominators[None, :]

        self.embedding = torch.zeros(num_positions, embedding_dim, requires_grad=False)
        self.embedding[:, 0::2] = torch.sin(positions)
        self.embedding[:, 1::2] = torch.cos(positions)

    def forward(self, x):
        if self.batch_first:
            return self.embedding[:x.shape[1], :]
        return self.embedding[:x.shape[0], :]

class SimpleTransformer(nn.Module):
    """
    A simple Transformer model for sequence classification tasks.
    Args:
        d_model (int): The dimension of the embedding vector.
        n_tokens (int): The number of tokens in the vocabulary.
        max_positions (int): The maximum number of positions for positional encoding.
        n_heads (int): The number of attention heads in the multi-head attention layer.
        *args: Additional arguments for the nn.Module superclass.
        **kwargs: Additional keyword arguments for the nn.Module superclass.
    Attributes:
        embed (nn.Embedding): Embedding layer for token embeddings.
        position (PositionalEmbedding): Positional embedding layer.
        attention_layer (nn.MultiheadAttention): Multi-head attention layer.
        classify (nn.Linear): Linear layer for classification.
    Methods:
        forward(x):
            Forward pass of the model.
            Args:
                x (torch.Tensor): Input tensor of token indices with shape (batch_size, sequence_length).
            Returns:
                torch.Tensor: Logits for the last token in the sequence with shape (batch_size, n_tokens).
    """
    def __init__(self, d_model, n_tokens, max_positions, n_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # semantic + positional embedding
        self.embed = nn.Embedding(n_tokens, d_model)
        self.position = PositionalEmbedding(max_positions, d_model)
        # single attention layer
        self.attention_layer = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)
        # classification layer ouptuts the probability of each token in the vocabulary. 
        self.classify = nn.Linear(d_model, n_tokens)

    def forward(self, x):
        x = self.embed(x)
        
        positions = self.position(x)
        x = x + positions

        attn, attention_weights = self.attention_layer(x, x, x)
        x = x + attn
        
        logits = self.classify(x)
        final_logit = logits[:, -1, :]
        return final_logit
    
class ComplexTransformer(nn.Module):
    """
    ComplexTransformer is a neural network model that combines semantic and positional embeddings with multi-head attention layers to perform sequence classification.
    Attributes:
        max_positions (int): The maximum number of positions in the input sequence.
        embed (nn.Embedding): Embedding layer for token embeddings.
        position (PositionalEmbedding): Positional embedding layer.
        attention_layer_1 (nn.MultiheadAttention): First multi-head attention layer.
        attention_layer_2 (nn.MultiheadAttention): Second multi-head attention layer.
        classify (nn.Linear): Linear layer for classification.
    Methods:
        __init__(d_model, n_tokens, max_positions, n_heads, *args, **kwargs):
            Initializes the ComplexTransformer model with the given parameters.
        forward(x):
            Performs a forward pass through the model.
    """
    def __init__(self, d_model, n_tokens, max_positions, n_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # semantic + positional embedding
        self.max_positions = max_positions
        self.embed = nn.Embedding(n_tokens, d_model)
        self.position = PositionalEmbedding(max_positions, d_model)
        # single attention layer
        self.attention_layer_1 = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)

        self.attention_layer_2 = nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True)
        # classification layer ouptuts the probability of each token in the vocabulary. 
        self.classify = nn.Linear(d_model, n_tokens)

    def forward(self, x):
        """
        Perform a forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes) representing the final logits.
        """
        x = self.embed(x)
        
        positions = self.position(x)
        x = x + positions
        
        attn_1 , attention_weights = self.attention_layer_1(x, x, x, attn_mask=torch.tril(torch.ones(self.max_positions, self.max_positions)))
        x = x + attn_1

        attn_2, attention_weights = self.attention_layer_2(x, x, x, attn_mask=torch.tril(torch.ones(self.max_positions, self.max_positions)))
        x = x + attn_2
        
        logits = self.classify(x)
        final_logit = logits[:, -1, :]
        return final_logit
    
def optimize_model(model, criterion, optimizer, dataset, n_epochs=1000, batch_size=32):
    """
    Trains and evaluates the given model using the specified criterion and optimizer.
    Args:
        model (torch.nn.Module): The model to be trained and evaluated.
        criterion (torch.nn.Module): The loss function used to evaluate the model.
        optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
        dataset (Dataset): The dataset object that provides training and testing data.
        n_epochs (int, optional): The number of epochs to train the model. Default is 1000.
        batch_size (int, optional): The number of samples per batch. Default is 32.
    Returns:
        tuple: A tuple containing two lists:
            - train_losses (list of float): The training loss recorded at each epoch.
            - test_losses (list of float): The testing loss recorded at each epoch.
    """
    train_losses = []
    test_losses = []
    for epoch in range(n_epochs):
        model.train()
        X, y = dataset.generate_batch(batch_size, type='train')
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        model.eval()
        X, y = dataset.generate_batch(batch_size, type='test')
        y_pred = model(X)
        loss = criterion(y_pred, y)
        test_losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_losses[-1]}, Test Loss: {test_losses[-1]}")
    return train_losses, test_losses
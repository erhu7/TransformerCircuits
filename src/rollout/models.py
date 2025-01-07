import torch
import torch.nn as nn
import numpy as np


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings."""
    def __init__(self, num_positions: int, embedding_dim: int, batch_first: bool = True):
        super().__init__()
        self.d_model = embedding_dim
        self.batch_first = batch_first
        
        # Compute position encodings once
        denominators = (10000)**(torch.arange(0, embedding_dim, 2)/embedding_dim)
        positions = torch.arange(0, num_positions)[:, None] / denominators[None, :]
        
        self.embedding = torch.zeros(num_positions, embedding_dim, requires_grad=False)
        self.embedding[:, 0::2] = torch.sin(positions)
        self.embedding[:, 1::2] = torch.cos(positions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1] if self.batch_first else x.shape[0]
        return self.embedding[:seq_len, :]

class FlexibleTransformer(nn.Module):
    """A transformer with configurable number of attention layers."""
    
    def __init__(
        self,
        d_model: int,
        n_tokens: int,
        max_positions: int,
        n_heads: int,
        n_attn_layers: int,
        include_bias: bool = False
    ):
        super().__init__()
        # Embeddings
        self.embed = nn.Embedding(n_tokens, d_model)
        self.position = PositionalEmbedding(max_positions, d_model)
        
        # Create n attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=n_heads, batch_first=True, bias=include_bias)
            for _ in range(n_attn_layers)
        ])
        
        # Output layer
        self.classify = nn.Linear(d_model, n_tokens)
        
        # Create and register attention mask
        mask = torch.tril(torch.ones(max_positions, max_positions))
        self.register_buffer('attn_mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fwd_internal(x)
    
    def forward_with_weights(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return self._fwd_internal(x, return_weights=True)
    
    def _fwd_internal(self, x: torch.Tensor, return_weights: bool = False):
        if self.attn_mask.device != x.device:
            self.attn_mask = self.attn_mask.to(x.device)
            
        x = self.embed(x)
        x = x + self.position(x)
        attention_weights = []
        for attention_layer in self.attention_layers:
            attn_out, weights = attention_layer(x, x, x, attn_mask=self.attn_mask)
            x = x + attn_out
            if return_weights:
                attention_weights.append(weights.detach())
        
        logits = self.classify(x)
        final_logit = logits[:, -1, :]
        
        return (final_logit, attention_weights) if return_weights else final_logit

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
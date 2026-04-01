import numpy as np
import math
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn


def calculate_rate(snr, W_bandwidth):
    """Calculate the data rate based on the SNR using Shannon - Hartley formula."""
    return W_bandwidth * np.log2(1 + snr)

def indicator(r_threshold, rate):
    """Indicator function that returns 1 if the rate meets the threshold, else 0."""
    return 1 if rate >= r_threshold else 0


# ==================== Neural Network Utils ====================

def init(module, weight_init, bias_init, gain=1):
    """Initialize module weights and biases."""
    if isinstance(module, nn.Linear):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module


def ortho_init(scale=np.sqrt(2)):
    def _ortho_init(tensor, gain=1):
        if tensor.dim() < 2:
            raise ValueError("Only tensors with 2 or more dimensions are supported")
        rows = tensor.size(0)
        cols = np.prod(tensor.shape[1:])
        flattened = torch.randn(rows, cols)
        u, s, v = torch.svd(flattened)
        tensor.data.copy_(u[:, :cols].reshape(tensor.shape) * gain)
        return tensor
    return _ortho_init


class MLPBase(nn.Module):
    """
    MLP feature extractor for flat observations.
    
    Args:
        obs_shape: shape of observation (single int or tuple)
        hidden_dim: hidden layer dimension
        num_layers: number of hidden layers
        use_orthogonal: whether to use orthogonal initialization
        use_rnn: whether to use RNN (affects output shape)
    """
    
    def __init__(self, obs_shape, hidden_dim=64, num_layers=2, hidden_dims=None, use_orthogonal=True, activation=nn.ReLU):
        super().__init__()
        
        if isinstance(obs_shape, (tuple, list)):
            obs_dim = int(np.prod(obs_shape))
        else:
            obs_dim = obs_shape
        
        self.hidden_dim = hidden_dim
        self.use_orthogonal = use_orthogonal
        
        gain = np.sqrt(2) if use_orthogonal else 1.0
        def init_fn(tensor, gain=1):
            if use_orthogonal:
                nn.init.orthogonal_(tensor, gain=gain)
            else:
                nn.init.xavier_uniform_(tensor)
        
        # Build MLP layers
        if hidden_dims is not None:
            layer_dims = [int(dim) for dim in hidden_dims]
        else:
            layer_dims = [int(hidden_dim)] * int(num_layers)

        if len(layer_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer size")

        layers = []
        layers.append(nn.Linear(obs_dim, layer_dims[0]))
        layers.append(activation())

        for idx in range(1, len(layer_dims)):
            layers.append(nn.Linear(layer_dims[idx - 1], layer_dims[idx]))
            layers.append(activation())
        
        # Initialize weights
        for layer in layers:
            if isinstance(layer, nn.Linear):
                init_fn(layer.weight, gain=gain)
                nn.init.constant_(layer.bias, 0)
        
        self.fc_layers = nn.Sequential(*layers)
        self.output_dim = layer_dims[-1]
        
    def forward(self, obs):
        return self.fc_layers(obs)


class ACTLayer(nn.Module):
    """
    Action output layer for discrete action spaces.
    
    Outputs logits, which are converted to probabilities via softmax.
    """
    
    def __init__(self, action_dim, hidden_dim, use_orthogonal=True, gain=0.01):
        super().__init__()
        self.action_dim = action_dim
        self.fc = nn.Linear(hidden_dim, action_dim)
        
        if use_orthogonal:
            nn.init.orthogonal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_uniform_(self.fc.weight)
        
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, hidden_state):
        """Output action logits."""
        return self.fc(hidden_state)

class CNNLayer(nn.Module):
    """
    CNN feature extractor for image observations.
    
    Args:
        obs_shape: shape of observation (C, H, W)
        hidden_dim: hidden layer dimension
        num_layers: number of hidden layers
        use_orthogonal: whether to use orthogonal initialization
    """
    
    def __init__(self, obs_shape, hidden_dim=64, num_layers=2, use_orthogonal=True, activation=nn.ReLU):
        super().__init__()
        
        self.use_orthogonal = use_orthogonal
        
        gain = np.sqrt(2) if use_orthogonal else 1.0
        def init_fn(tensor, gain=1):
            if use_orthogonal:
                nn.init.orthogonal_(tensor, gain=gain)
            else:
                nn.init.xavier_uniform_(tensor)
        
        # Build CNN layers
        layers = []
        in_channels = obs_shape[0]
        out_channels = 32
        
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(activation())
            init_fn(layers[-2].weight, gain=gain)
            nn.init.constant_(layers[-2].bias, 0)
            in_channels = out_channels
            out_channels *= 2
        
        self.cnn_layers = nn.Sequential(*layers)
        
    def forward(self, obs):
        return self.cnn_layers(obs)

class RNNLayer(nn.Module):
    """
    RNN layer for sequential observations.
    
    Args:
        input_dim: dimension of input features
        hidden_dim: hidden layer dimension
        num_layers: number of RNN layers
        use_orthogonal: whether to use orthogonal initialization
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, use_orthogonal=True):
        super().__init__()
        
        self.use_orthogonal = use_orthogonal
        
        gain = np.sqrt(2) if use_orthogonal else 1.0
        def init_fn(tensor, gain=1):
            if use_orthogonal:
                nn.init.orthogonal_(tensor, gain=gain)
            else:
                nn.init.xavier_uniform_(tensor)
        
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers)
        
        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                init_fn(param.data, gain=gain)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
    
    def forward(self, obs, hidden_state):
        """Forward pass through RNN
        Args:
            obs: (seq_len, batch_size, input_dim) input features
            hidden_state: (num_layers, batch_size, hidden_dim) initial hidden state
        Returns:
            output: (seq_len, batch_size, hidden_dim) RNN output features
            hidden_state: (num_layers, batch_size, hidden_dim) final hidden state
        """        
        output, hidden_state = self.rnn(obs, hidden_state)
        return output, hidden_state
# def compute_gae(rewards, values, next_value, gamma=0.99, gae_lambda=0.95):
#     """
#     Compute Generalized Advantage Estimation (GAE).
    
#     Args:
#         rewards: (T, N) reward tensor
#         values: (T, N) value estimates
#         next_value: (N,) value of next state
#         gamma: discount factor
#         gae_lambda: GAE lambda parameter
    
#     Returns:
#         advantages: (T, N) advantage estimates
#         returns: (T, N) target returns
#     """
#     T, N = rewards.shape
#     advantages = torch.zeros_like(rewards)
#     gae = torch.zeros(N, dtype=rewards.dtype, device=rewards.device)
    
#     next_val = next_value
#     for t in reversed(range(T)):
#         delta = rewards[t] + gamma * next_val - values[t]
#         gae = delta + gamma * gae_lambda * gae
#         advantages[t] = gae
#         next_val = values[t]
    
#     returns = advantages + values
#     return advantages, returns


# def huber_loss(values, targets, huber_delta=10.0):
#     """Compute Huber loss."""
#     diff = (targets - values).abs()
#     loss = torch.where(diff < huber_delta, 0.5 * diff ** 2, huber_delta * (diff - 0.5 * huber_delta))
#     return loss.mean()




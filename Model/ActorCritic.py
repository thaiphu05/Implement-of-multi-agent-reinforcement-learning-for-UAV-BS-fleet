import torch
import torch.nn as nn
import numpy as np
from utils import MLPBase, ACTLayer


class R_Actor(nn.Module):
    """
    Args:
        obs_shape: observation dimension or shape
        action_dim: number of discrete actions
        hidden_dim: hidden layer dimension (default: 512)
        num_layers: number of hidden layers (default: 2)
        use_orthogonal: use orthogonal weight initialization (default: True)
        gain: initialization gain for output layer (default: 0.01)
        activation: activation function class (default: nn.ReLU)
        device: device to run on (default: cpu)
    """
    
    def __init__(self, obs_shape, action_dim, hidden_dim=512, num_layers=2, hidden_dims=None,
                 use_orthogonal=True, gain=0.01, activation=nn.ReLU,
                 device=torch.device("cuda")):
        super(R_Actor, self).__init__()
        
        self.obs_shape = obs_shape
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.use_orthogonal = use_orthogonal
        self._gain = gain
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        # Feature extractor (MLPBase handles both flat and image observations)
        self.base = MLPBase(
            obs_shape=obs_shape,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            use_orthogonal=use_orthogonal,
            activation=activation
        )
        
        # Action output layer (discrete actions)
        self.act = ACTLayer(
            action_dim=action_dim,
            hidden_dim=self.base.output_dim,
            use_orthogonal=use_orthogonal,
            gain=gain
        )
        
        self.to(device)

    def _to_tensor(self, x, dtype=torch.float32):
        if isinstance(x, np.ndarray):
            return torch.as_tensor(x, dtype=dtype, device=self.device)
        if torch.is_tensor(x):
            return x.to(device=self.device, dtype=dtype)
        return torch.as_tensor(x, dtype=dtype, device=self.device)

    def _apply_available_actions_mask(self, logits, available_actions):
        if available_actions is None:
            return logits
        action_mask = self._to_tensor(available_actions, dtype=torch.float32)
        if action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)
        invalid = action_mask <= 0.0
        return logits.masked_fill(invalid, -1e10)
    
    def forward(self, obs, available_actions=None, deterministic=False):
        """
        Args:
            obs: (batch_size, obs_dim) observation tensor or numpy array
            deterministic: if True, return argmax action; else sample from distribution
            
        Returns:
            actions: (batch_size,) sampled/argmax actions
            action_log_probs: (batch_size,) log probabilities of selected actions
        """
        obs = self._to_tensor(obs, dtype=torch.float32)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self.base(obs)
        
        action_logits = self.act(features)
        action_logits = self._apply_available_actions_mask(action_logits, available_actions)
        dist = torch.distributions.Categorical(logits=action_logits)
        
        if deterministic:
            actions = action_logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        
        action_log_probs = dist.log_prob(actions)
        
        return actions, action_log_probs
    
    def evaluate_actions(self, obs, actions, available_actions=None):
        """
        Args:
            obs: (batch_size, obs_dim) observations
            actions: (batch_size,) actions taken
            
        Returns:
            action_log_probs: (batch_size,) log probabilities of actions
            dist_entropy: scalar, average entropy of action distribution
        """
        obs = self._to_tensor(obs, dtype=torch.float32)
        actions = self._to_tensor(actions, dtype=torch.long)
        features = self.base(obs)
        action_logits = self.act(features)
        action_logits = self._apply_available_actions_mask(action_logits, available_actions)
        dist = torch.distributions.Categorical(logits=action_logits)
        action_log_probs = dist.log_prob(actions)
        dist_entropy = dist.entropy().mean()
        
        return action_log_probs, dist_entropy


class R_Critic(nn.Module):
    """
    Args:
        obs_shape: observation dimension (typically concatenated from all agents)
        hidden_dim: hidden layer dimension (default: 512)
        num_layers: number of hidden layers (default: 2)
        use_orthogonal: use orthogonal weight initialization (default: True)
        activation: activation function class (default: nn.ReLU)
        device: device to run on (default: cpu)
    """
    
    def __init__(self, obs_shape, hidden_dim=512, num_layers=2, hidden_dims=None,
                 use_orthogonal=True, activation=nn.ReLU,
                 device=torch.device("cpu")):
        super(R_Critic, self).__init__()
        
        self.obs_shape = obs_shape
        self.hidden_dim = hidden_dim
        self.use_orthogonal = use_orthogonal
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        self.base = MLPBase(
            obs_shape=obs_shape,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            hidden_dims=hidden_dims,
            use_orthogonal=use_orthogonal,
            activation=activation
        )
        
        self.v_out = nn.Linear(self.base.output_dim, 1)
        if use_orthogonal:
            nn.init.orthogonal_(self.v_out.weight, gain=1.0)
        else:
            nn.init.xavier_uniform_(self.v_out.weight)
        nn.init.constant_(self.v_out.bias, 0)
        
        self.to(device)
    
    def forward(self, obs):
        """        
        Args:
            obs: (batch_size, obs_dim) observation tensor or numpy array
            
        Returns:
            values: (batch_size,) value estimates
        """
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).to(**self.tpdv)
        elif not isinstance(obs, torch.Tensor):
            obs = torch.FloatTensor(obs).to(**self.tpdv)
        else:
            obs = obs.to(**self.tpdv)
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        features = self.base(obs)
        values = self.v_out(features).squeeze(-1)
        
        return values
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
            init_fn(layers[-2].weigMht, gain=gain)
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

def infer_checkpoint(path=None, map_location="cpu"):
    """Load a checkpoint and expose actor/critic weights for inference.

    Args:
        path: checkpoint file path or a directory containing checkpoint files.
        map_location: device mapping passed to ``torch.load``.

    Returns:
        A dictionary with loaded checkpoint and extracted actor/critic state dicts.
    """
    if path is None:
        return None

    checkpoint_path = path
    if os.path.isdir(path):
        candidates = [
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.endswith((".pt", ".pth"))
        ]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint file found in directory: {path}")
        checkpoint_path = max(candidates, key=os.path.getmtime)

    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if isinstance(checkpoint, dict):
        actor_state_dict = checkpoint.get("actor")
        critic_state_dict = checkpoint.get("critic")
    else:
        actor_state_dict = None
        critic_state_dict = None

    return {
        "checkpoint_path": checkpoint_path,
        "checkpoint": checkpoint,
        "actor_state_dict": actor_state_dict,
        "critic_state_dict": critic_state_dict,
    }


def plot_assignment_snapshot(snapshot, save_path=None, show=True, dpi=140, uav_paths=None):
    """Draw a two-panel assignment figure without partition boundaries.

    Args:
        snapshot: dict returned by MultiUAVEnv.get_visualization_snapshot().
        save_path: optional output path (png/jpg).
        show: whether to display the plot window.
        dpi: figure DPI when saving.
        uav_paths: optional UAV trajectories with shape (num_uav, T, 2) or
            list of arrays each shaped (T, 2).

    Returns:
        (fig, axes) tuple from matplotlib.
    """
    required_keys = {
        "user_positions",
        "assignment",
        "uav_positions",
        "mbs_position",
        "map_min",
        "map_max",
    }
    missing_keys = required_keys.difference(snapshot.keys())
    if missing_keys:
        raise KeyError(f"Snapshot missing keys: {sorted(missing_keys)}")

    user_positions = np.asarray(snapshot["user_positions"], dtype=np.float32)
    assignment = np.asarray(snapshot["assignment"], dtype=np.int32)
    uav_positions = np.asarray(snapshot["uav_positions"], dtype=np.float32)
    mbs_position = np.asarray(snapshot["mbs_position"], dtype=np.float32)
    map_min = float(snapshot["map_min"])
    map_max = float(snapshot["map_max"])

    if user_positions.ndim != 2 or user_positions.shape[1] != 2:
        raise ValueError("user_positions must have shape (N, 2)")

    num_uavs = int(uav_positions.shape[0])
    mbs_user_color = "#1f77b4"
    server_colors = [mbs_user_color, "#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#8c564b"]
    uav_marker_colors = ["red", "blue", "green", "purple", "orange"]
    server_labels = ["mBS"] + [f"UAV {i}" for i in range(num_uavs)]

    normalized_paths = None
    if uav_paths is not None:
        if isinstance(uav_paths, np.ndarray):
            if uav_paths.ndim != 3 or uav_paths.shape[0] != num_uavs or uav_paths.shape[2] != 2:
                raise ValueError("uav_paths ndarray must have shape (num_uav, T, 2)")
            normalized_paths = [uav_paths[i] for i in range(num_uavs)]
        else:
            if len(uav_paths) != num_uavs:
                raise ValueError("uav_paths list length must match number of UAVs")
            normalized_paths = [np.asarray(path, dtype=np.float32) for path in uav_paths]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax0 = axes[0]
    ax0.scatter(
        user_positions[:, 0],
        user_positions[:, 1],
        s=14,
        facecolors="none",
        edgecolors="#1f77b4",
        linewidths=0.8,
    )

    mbs_served_mask = assignment == 0
    if np.any(mbs_served_mask):
        ax0.scatter(
            user_positions[mbs_served_mask, 0],
            user_positions[mbs_served_mask, 1],
            s=20,
            marker="v",
            color=server_colors[0],
            alpha=0.9,
            label="mBS",
        )

    # Mark UAV positions explicitly (UAV 0 in red, UAV 1 in blue)
    for i in range(num_uavs):
        marker_color = uav_marker_colors[i % len(uav_marker_colors)]
        if normalized_paths is not None and normalized_paths[i].shape[0] >= 2:
            ax0.plot(
                normalized_paths[i][:, 0],
                normalized_paths[i][:, 1],
                color=marker_color,
                linestyle="-",
                linewidth=1.4,
                alpha=0.85,
                zorder=3,
            )
        ax0.scatter(
            uav_positions[i, 0],
            uav_positions[i, 1],
            marker="s",
            s=70,
            c=marker_color,
            edgecolors="k",
            linewidths=0.8,
            zorder=4,
            label=f"UAV {i}",
        )

    ax0.set_xlim(map_min, map_max)
    ax0.set_ylim(map_min, map_max)
    ax0.set_aspect("equal", adjustable="box")
    ax0.grid(True, alpha=0.25)

    # Right: users grouped by selected server (same color as serving UAV)
    ax1 = axes[1]

    # mBS-served users
    mbs_mask = assignment == 0
    if np.any(mbs_mask):
        ax1.scatter(
            user_positions[mbs_mask, 0],
            user_positions[mbs_mask, 1],
            s=16,
            marker="v",
            facecolors="none",
            edgecolors="#1f77b4",
            linewidths=0.95,
            label="mBS-served users",
        )

    # UAV-served users (same color as serving UAV marker)
    for i in range(num_uavs):
        mask = assignment == (i + 1)
        if not np.any(mask):
            continue
        marker_color = uav_marker_colors[i % len(uav_marker_colors)]
        ax1.scatter(
            user_positions[mask, 0],
            user_positions[mask, 1],
            s=16,
            marker="o",
            facecolors="none",
            edgecolors=marker_color,
            linewidths=0.95,
            label=f"UAV {i}-served users",
        )

    unsatisfied_mask = assignment < 0
    if np.any(unsatisfied_mask):
        ax1.scatter(
            user_positions[unsatisfied_mask, 0],
            user_positions[unsatisfied_mask, 1],
            s=16,
            marker="x",
            color="#7f7f7f",
            linewidths=0.8,
            label="Unserved",
        )

    # Plot base-station positions for reference
    ax1.scatter(
        mbs_position[0],
        mbs_position[1],
        marker="s",
        s=70,
        c=mbs_user_color,
        edgecolors="k",
        linewidths=0.9,
        zorder=4,
        label="mBS",
    )
    for i in range(num_uavs):
        marker_color = uav_marker_colors[i % len(uav_marker_colors)]
        if normalized_paths is not None and normalized_paths[i].shape[0] >= 2:
            ax1.plot(
                normalized_paths[i][:, 0],
                normalized_paths[i][:, 1],
                color=marker_color,
                linestyle="-",
                linewidth=1.4,
                alpha=0.85,
                zorder=3,
            )
        ax1.scatter(
            uav_positions[i, 0],
            uav_positions[i, 1],
            marker="s",
            s=70,
            c=marker_color,
            edgecolors="k",
            linewidths=0.8,
            zorder=4,
            label=f"UAV {i}",
        )

    ax1.set_xlim(map_min, map_max)
    ax1.set_ylim(map_min, map_max)
    ax1.set_aspect("equal", adjustable="box")
    ax1.grid(True, alpha=0.25)
    # Remove duplicate legend labels from repeated scatter calls.
    handles, labels = ax1.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=max(2, num_uavs + 1), frameon=True, fontsize=8)

    if save_path is not None:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes



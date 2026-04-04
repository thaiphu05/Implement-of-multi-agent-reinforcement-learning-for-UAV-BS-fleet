import torch
import torch.nn as nn
from Model.ActorCritic import R_Actor, R_Critic

class R_MAPPOPolicy:
    """
    MAPPO Policy class. Wraps an actor and a critic network to compute actions
    and value predictions.
    Uses Centralized Training with Decentralized Execution (CTDE), where the critic
    has access to centralized global state (share_obs) and actor has access to local
    observations (obs).
    """

    def __init__(self, args, obs_space, share_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.initial_lr = args.lr
        self.critic_lr = args.critic_lr
        self.initial_critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        self.hidden_dims = getattr(args, "hidden_dims", None)

        self.obs_space = obs_space
        self.share_obs_space = share_obs_space
        self.act_space = act_space

        self.actor = R_Actor(self.obs_space, self.act_space, 
                             hidden_dim=args.hidden_size, 
                             hidden_dims=self.hidden_dims,
                             activation=nn.ReLU, 
                             device=self.device)
        self.critic = R_Critic(self.share_obs_space, 
                               hidden_dim=args.hidden_size, 
                               hidden_dims=self.hidden_dims,
                               activation=nn.ReLU, 
                               device=self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def eval_mode(self):
        self.actor.eval()
        self.critic.eval()

    def get_actor_params(self):
        return self.actor.parameters()

    def get_critic_params(self):
        return self.critic.parameters()

    def lr_decay(self, episode, episodes):
        frac = 1.0 - (episode / max(1, episodes))
        actor_lr = self.initial_lr * frac
        critic_lr = self.initial_critic_lr * frac
        for group in self.actor_optimizer.param_groups:
            group["lr"] = actor_lr
        for group in self.critic_optimizer.param_groups:
            group["lr"] = critic_lr

    def get_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        """
        Compute actions and value outputs for given inputs.
        """
        actions, action_log_probs = self.actor(obs,
                                               available_actions=available_actions,
                                               deterministic=deterministic)
        values = self.critic(share_obs)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, share_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        """
        values = self.critic(share_obs)
        return values

    def evaluate_actions(self, share_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs and entropy, as well as value predictions.
        (CTDE: critic takes share_obs, actor takes obs)
        """
        # Actor
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     action,
                                                                     available_actions=available_actions)
        
        # Critic
        values = self.critic(share_obs)

        return values, action_log_probs, dist_entropy

    def save(self, path):
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load(self, path, map_location=None, load_optimizers=True):
        checkpoint = torch.load(path, map_location=map_location or self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

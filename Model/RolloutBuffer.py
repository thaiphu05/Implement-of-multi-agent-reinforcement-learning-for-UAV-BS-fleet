import numpy as np
import torch


class MultiAgentRolloutBuffer:
    """On-policy rollout buffer for MAPPO with GAE."""

    def __init__(
        self,
        episode_length,
        n_agents,
        obs_dim,
        share_obs_dim,
        action_dim,
        gamma=0.99,
        gae_lambda=0.95,
        device=torch.device("cpu"),
    ):
        self.episode_length = int(episode_length)
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.share_obs_dim = int(share_obs_dim)
        self.action_dim = int(action_dim)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.reset()

    def reset(self):
        t = self.episode_length
        n = self.n_agents

        self.obs = np.zeros((t, n, self.obs_dim), dtype=np.float32)
        self.share_obs = np.zeros((t, n, self.share_obs_dim), dtype=np.float32)
        self.actions = np.zeros((t, n), dtype=np.int64)
        self.action_log_probs = np.zeros((t, n), dtype=np.float32)
        self.rewards = np.zeros((t, n), dtype=np.float32)
        self.value_preds = np.zeros((t + 1, n), dtype=np.float32)
        self.returns = np.zeros((t + 1, n), dtype=np.float32)
        self.advantages = np.zeros((t, n), dtype=np.float32)
        self.masks = np.ones((t + 1, n), dtype=np.float32)
        self.available_actions = np.ones((t, n, self.action_dim), dtype=np.float32)

    def insert(
        self,
        step,
        share_obs,
        obs,
        actions,
        rewards,
        values,
        action_log_probs,
        masks,
        available_actions=None,
    ):
        self.share_obs[step] = np.asarray(share_obs, dtype=np.float32)
        self.obs[step] = np.asarray(obs, dtype=np.float32)
        self.actions[step] = np.asarray(actions, dtype=np.int64)
        self.rewards[step] = np.asarray(rewards, dtype=np.float32)
        self.value_preds[step] = np.asarray(values, dtype=np.float32)
        self.action_log_probs[step] = np.asarray(action_log_probs, dtype=np.float32)
        self.masks[step + 1] = np.asarray(masks, dtype=np.float32)

        if available_actions is not None:
            self.available_actions[step] = np.asarray(available_actions, dtype=np.float32)

    def compute_returns_and_advantages(self, next_values):
        self.value_preds[-1] = np.asarray(next_values, dtype=np.float32)
        gae = np.zeros((self.n_agents,), dtype=np.float32)

        for step in reversed(range(self.episode_length)):
            delta = (
                self.rewards[step]
                + self.gamma * self.value_preds[step + 1] * self.masks[step + 1]
                - self.value_preds[step]
            )
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.advantages[step] = gae
            self.returns[step] = self.advantages[step] + self.value_preds[step]

        self.returns[-1] = self.value_preds[-1]

    def feed_forward_generator(self, num_mini_batch=None, mini_batch_size=None):
        t, n = self.episode_length, self.n_agents
        batch_size = t * n

        obs = self.obs.reshape(batch_size, self.obs_dim)
        share_obs = self.share_obs.reshape(batch_size, self.share_obs_dim)
        actions = self.actions.reshape(batch_size)
        value_preds = self.value_preds[:-1].reshape(batch_size)
        returns = self.returns[:-1].reshape(batch_size)
        old_action_log_probs = self.action_log_probs.reshape(batch_size)
        advantages = self.advantages.reshape(batch_size)
        masks = self.masks[1:].reshape(batch_size)
        available_actions = self.available_actions.reshape(batch_size, self.action_dim)

        if mini_batch_size is None:
            if num_mini_batch is None or num_mini_batch <= 0:
                raise ValueError("Either num_mini_batch or mini_batch_size must be set.")
            mini_batch_size = batch_size // num_mini_batch

        permutation = np.random.permutation(batch_size)
        for start in range(0, batch_size, mini_batch_size):
            end = start + mini_batch_size
            idx = permutation[start:end]
            if idx.size == 0:
                continue

            yield {
                "obs": torch.as_tensor(obs[idx], dtype=torch.float32, device=self.device),
                "share_obs": torch.as_tensor(share_obs[idx], dtype=torch.float32, device=self.device),
                "actions": torch.as_tensor(actions[idx], dtype=torch.long, device=self.device),
                "value_preds": torch.as_tensor(value_preds[idx], dtype=torch.float32, device=self.device),
                "returns": torch.as_tensor(returns[idx], dtype=torch.float32, device=self.device),
                "old_action_log_probs": torch.as_tensor(old_action_log_probs[idx], dtype=torch.float32, device=self.device),
                "advantages": torch.as_tensor(advantages[idx], dtype=torch.float32, device=self.device),
                "masks": torch.as_tensor(masks[idx], dtype=torch.float32, device=self.device),
                "available_actions": torch.as_tensor(available_actions[idx], dtype=torch.float32, device=self.device),
            }

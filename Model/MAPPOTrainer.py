import torch
import torch.nn.functional as F


class R_MAPPOTrainer:
    """PPO update logic for MAPPO policies."""

    def __init__(self, policy, args, device=torch.device("cpu")):
        self.policy = policy
        self.device = device

        self.clip_param = getattr(args, "clip_param", 0.2)
        self.ppo_epoch = getattr(args, "ppo_epoch", 10)
        self.num_mini_batch = getattr(args, "num_mini_batch", 4)
        self.value_loss_coef = getattr(args, "value_loss_coef", 0.5)
        self.entropy_coef = getattr(args, "entropy_coef", 0.003)
        self.max_grad_norm = getattr(args, "max_grad_norm", 10.0)
        self.use_clipped_value_loss = getattr(args, "use_clipped_value_loss", True)

    def cal_value_loss(self, values, value_preds_batch, return_batch):
        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * F.mse_loss(values, return_batch)
        return value_loss

    def train(self, buffer):
        self.policy.train_mode()

        advantages = torch.as_tensor(
            buffer.advantages, dtype=torch.float32, device=self.device
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        buffer.advantages = advantages.cpu().numpy()

        value_loss_epoch = 0.0
        policy_loss_epoch = 0.0
        dist_entropy_epoch = 0.0

        update_count = 0
        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator(
                num_mini_batch=self.num_mini_batch
            )
            for sample in data_generator:
                obs_batch = sample["obs"]
                share_obs_batch = sample["share_obs"]
                actions_batch = sample["actions"]
                value_preds_batch = sample["value_preds"]
                return_batch = sample["returns"]
                old_action_log_probs_batch = sample["old_action_log_probs"]
                adv_targ = sample["advantages"]
                masks_batch = sample["masks"]
                available_actions_batch = sample["available_actions"]

                values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
                    share_obs_batch,
                    obs_batch,
                    None,
                    None,
                    actions_batch,
                    masks_batch,
                    available_actions=available_actions_batch,
                    active_masks=None,
                )

                ratios = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratios * adv_targ
                surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = self.cal_value_loss(values, value_preds_batch, return_batch)

                self.policy.actor_optimizer.zero_grad()
                actor_loss = policy_loss - self.entropy_coef * dist_entropy
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(), self.max_grad_norm
                )
                self.policy.actor_optimizer.step()

                self.policy.critic_optimizer.zero_grad()
                critic_loss = self.value_loss_coef * value_loss
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(), self.max_grad_norm
                )
                self.policy.critic_optimizer.step()

                value_loss_epoch += value_loss.item()
                policy_loss_epoch += policy_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                update_count += 1

        if update_count == 0:
            return {
                "value_loss": 0.0,
                "policy_loss": 0.0,
                "dist_entropy": 0.0,
            }

        return {
            "value_loss": value_loss_epoch / update_count,
            "policy_loss": policy_loss_epoch / update_count,
            "dist_entropy": dist_entropy_epoch / update_count,
        }

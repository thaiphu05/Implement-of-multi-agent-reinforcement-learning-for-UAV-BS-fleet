import argparse
from types import SimpleNamespace

import numpy as np
import torch

from Model.Enviroment import MultiUAVEnv
from Model.MAPPOPolicy import R_MAPPOPolicy
from Model.MAPPOTrainer import R_MAPPOTrainer
from Model.RolloutBuffer import MultiAgentRolloutBuffer


def build_hotspot_users(num_users=250, map_limit=1000.0, rate_threshold=20e6, rate_mbs=20e6, seed=42):
    rng = np.random.default_rng(seed)
    hotspot_centers = np.array([
        [-500.0, -500.0],
        [500.0, -500.0],
        [-500.0, 500.0],
        [500.0, 500.0],
    ], dtype=np.float32)
    hotspot_std = 300.0
    
    user_matrix = []
    for center in hotspot_centers:
        num_users_in_hotspot = num_users // len(hotspot_centers)
        users_x = rng.normal(loc=center[0], scale=hotspot_std, size=num_users_in_hotspot)
        users_y = rng.normal(loc=center[1], scale=hotspot_std, size=num_users_in_hotspot)
        for x, y in zip(users_x, users_y):
            x_clipped = np.clip(x, -map_limit + 20, map_limit - 20)
            y_clipped = np.clip(y, -map_limit + 20, map_limit - 20)
            if(x+y > 0):
                user_matrix.append((float(x_clipped), float(y_clipped), rate_mbs))
            else:
                user_matrix.append((float(x_clipped), float(y_clipped), rate_threshold))
    
    # for _ in range(num_users):
    #     x = float(rng.uniform(-1000.0, 1000.0))
    #     y = float(rng.uniform(-1000.0, 1000.0))
    #     user_matrix.append((x, y, 20e6)) 

    return user_matrix



def parse_args():
    parser = argparse.ArgumentParser("Minimal MAPPO training for MultiUAVEnv")
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--num_users", type=int, default=250)
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--episode_length", type=int, default=300)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[1024, 512])
    parser.add_argument("--rate_threshold", type=float, default=20e6)
    parser.add_argument("--slot_length", type=float, default=1)
    parser.add_argument("--uav_k_factor", type=float, default=50.0)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-4)
    parser.add_argument("--opti_eps", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_param", type=float, default=0.1)
    parser.add_argument("--ppo_epoch", type=int, default=10)
    parser.add_argument("--num_mini_batch", type=int, default=4)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--checkpoint", type=str, default="mappo_checkpoint")
    parser.add_argument("--env_change_prob_start", type=float, default=0.0001)
    parser.add_argument("--env_change_prob_end", type=float, default=0.2)
    return parser.parse_args()


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def linear_schedule(start, end, progress):
    progress = float(np.clip(progress, 0.0, 1.0))
    return float(start + (end - start) * progress)


def main():
    cfg = parse_args()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    users = build_hotspot_users(
        num_users=cfg.num_users,
        rate_threshold=cfg.rate_threshold,
        seed=cfg.seed,
    )
    env = MultiUAVEnv(
        nums_UAV=cfg.num_agents,
        user_matrix=users,
        max_steps=cfg.episode_length,
        user_walk_speed=1.0,
        uav_k_factor=cfg.uav_k_factor,
    )

    obs_dim = int(env.get_observation(0).shape[0])
    share_obs_dim = int(env.get_all_observations().shape[0])
    act_dim = int(env.action_space.nvec[0])

    args = SimpleNamespace(
        lr=cfg.lr,
        critic_lr=cfg.critic_lr,
        opti_eps=cfg.opti_eps,
        weight_decay=cfg.weight_decay,
        hidden_size=cfg.hidden_size,
        hidden_dims=cfg.hidden_dims,
        clip_param=cfg.clip_param,
        ppo_epoch=cfg.ppo_epoch,
        num_mini_batch=cfg.num_mini_batch,
        entropy_coef=cfg.entropy_coef,
        value_loss_coef=cfg.value_loss_coef,
        max_grad_norm=cfg.max_grad_norm,
        use_clipped_value_loss=True,
    )

    policy = R_MAPPOPolicy(
        args=args,
        obs_space=obs_dim,
        share_obs_space=share_obs_dim,
        act_space=act_dim,
        device=device,
    )
    trainer = R_MAPPOTrainer(policy=policy, args=args, device=device)

    for idx, episode in enumerate(range(1, cfg.episodes + 1)):
        policy.lr_decay(episode - 1, cfg.episodes)
        buffer = MultiAgentRolloutBuffer(
            episode_length=cfg.episode_length,
            n_agents=cfg.num_agents,
            obs_dim=obs_dim,
            share_obs_dim=share_obs_dim,
            action_dim=act_dim,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            device=device,
        )
        progress = (episode - 1) / max(1, cfg.episodes - 1)
        env_change_prob = linear_schedule(
            cfg.env_change_prob_start,
            cfg.env_change_prob_end,
            progress,
        )
        do_env_change = np.random.rand() < env_change_prob
        obs, info = env.reset(seed=cfg.seed + episode, random_walk=do_env_change)
        share_obs = np.repeat(info["share_obs"][None, :], cfg.num_agents, axis=0).astype(np.float32)
        episode_reward = 0.0
        served_total_last_step = 0
        served_by_uav_last_step = np.zeros((cfg.num_agents,), dtype=np.int32)
        mbs_served_last_step = 0

        for step in range(cfg.episode_length):
            with torch.no_grad():
                values, actions, action_log_probs, _, _ = policy.get_actions(
                    share_obs=share_obs,
                    obs=obs,
                    rnn_states_actor=None,
                    rnn_states_critic=None,
                    masks=None,
                    available_actions=None,
                    deterministic=False,
                )

            actions_np = to_numpy(actions).astype(np.int64)
            values_np = to_numpy(values).astype(np.float32)
            log_probs_np = to_numpy(action_log_probs).astype(np.float32)
            

            next_obs, rewards, terminated, truncated, info = env.step(
                actions_np.tolist(),
                time_step=cfg.slot_length,
            )
            done = bool(terminated or truncated)
            masks = np.zeros((cfg.num_agents,), dtype=np.float32) if done else np.ones((cfg.num_agents,), dtype=np.float32)

            buffer.insert(
                step=step,
                share_obs=share_obs,
                obs=obs,
                actions=actions_np,
                rewards=rewards,
                values=values_np,
                action_log_probs=log_probs_np,
                masks=masks,
                available_actions=None,
            )
            
            # for reward in rewards:
            #     print (f"{reward}", end=" ")
            # print("\n")

            episode_reward += float(np.mean(rewards))
            obs = next_obs
            share_obs = np.repeat(info["share_obs"][None, :], cfg.num_agents, axis=0).astype(np.float32)
            served_total_last_step = int(info.get("connected_users", 0))
            served_by_uav_last_step = np.asarray(
                info.get("uav_serviced_counts", np.zeros((cfg.num_agents,), dtype=np.int32)),
                dtype=np.int32,
            )
            mbs_served_last_step = int(info.get("mbs_served_users", 0))

            if done:
                break

        next_share_obs = share_obs
        with torch.no_grad():
            next_values = policy.get_values(next_share_obs, None, None)
        next_values_np = to_numpy(next_values).astype(np.float32)

        buffer.compute_returns_and_advantages(next_values_np)
        train_info = trainer.train(buffer)

        print(
            f"Episode {episode:04d} | reward={episode_reward:.3f} | "
            f"served_total_last_step={served_total_last_step} | "
            f"served_by_uav_last_step={served_by_uav_last_step.tolist()} | "
            f"mbs_served_last_step={mbs_served_last_step} | "
            # f"env_change_prob={env_change_prob:.3f} | "
            # f"env_changed={int(do_env_change)} | "
            f"policy_loss={train_info['policy_loss']:.4f} | "
            f"value_loss={train_info['value_loss']:.4f} | "
            f"entropy={train_info['dist_entropy']:.4f}"
        )

        if episode % cfg.save_interval == 0:
            policy.save(cfg.checkpoint+f"_{episode}" + ".pt")

    policy.save(cfg.checkpoint)
    print(f"Training finished. Saved checkpoint to {cfg.checkpoint}")


if __name__ == "__main__":
    main()

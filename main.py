import numpy as np
import torch
from types import SimpleNamespace
import argparse


from Channel.channel_model import Channel_Model_mBS
from Channel.channel_model import Channel_Model_UAV
from Model.Enviroment import MultiUAVEnv
from Model.MAPPOPolicy import R_MAPPOPolicy
from utils import calculate_rate
from utils import infer_checkpoint
from utils import plot_assignment_snapshot

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

def main_0():
    # Parameters adapted from the provided simulation section.
    # Note: noise power in the paper is written as 90 dBm, but receiver noise is
    # physically expected to be negative; use -90 dBm for a meaningful SNR.
    channel_uav = Channel_Model_UAV(f_c=5.8e9, alpha=2.7, sigma2_dbm=-90, k_factor=50.0)
    channel_mbs = Channel_Model_mBS(f_c=2e9, sigma2_dbm=-90)
    
    d_2D_UAV = 500  # Representative user distance in meters (within a 2x2 km area)
    d_2D_mBS = 1800  # Representative user distance in meters (within a 2x2 km area)
    p_tx_uav_dbm = 30  # UAV transmit power in dBm
    p_tx_mbs_dbm = 46  # mBS transmit power in dBm
    h_mBS = 35  # Typical suburban macro BS height in meters
    h_UAV = 120  # UAV height in meters
    sigma_logf = 2  # Shadowing std from the papermBS
    
    snr_uav = np.mean([channel_uav.get_snr(d_2D_UAV, h_UAV, p_tx_uav_dbm) for _ in range(10000)])
    snr_mbs = np.mean([channel_mbs.get_snr(d_2D_mBS, h_mBS, p_tx_mbs_dbm, sigma_logf) for _ in range(10000)])
    
    W_bandwidth = 20e6  # Bandwidth in Hz
    rate_uav = calculate_rate(snr_uav, W_bandwidth)
    rate_mbs = calculate_rate(snr_mbs, W_bandwidth)
    
    snr_uav_db = 10 * np.log10(snr_uav)
    snr_mbs_db = 10 * np.log10(snr_mbs)

    print(f"UAV  -> SNR: {snr_uav:.4f} (linear), {snr_uav_db:.2f} dB | Rate: {rate_uav / 1e6:.2f} Mbps")
    print(f"mBS  -> SNR: {snr_mbs:.4f} (linear), {snr_mbs_db:.2f} dB | Rate: {rate_mbs / 1e6:.2f} Mbps")
    
def parse_args():
    parser = argparse.ArgumentParser("Minimal MAPPO training for MultiUAVEnv")
    parser.add_argument("--main", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=100)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.main == 0:
        main_0()
    elif args.main == 1:
        main_1(check_point=args.checkpoint)
    else:
        print(f"Invalid main argument: {args.main}. Use 0 for channel test, 1 for inference.")

def main_1(check_point=None):
    num_agents = 3
    num_users = 250
    episode_length = 300
    rng = np.random.default_rng(42)

    user_matrix = build_hotspot_users(num_users=num_users, seed=42)  # Generate user positions and rate requirements

    env = MultiUAVEnv(
        start_pos=(1000, 1000),
        nums_UAV=num_agents,
        user_matrix=user_matrix,
        max_steps=episode_length,
        user_walk_speed=1.0,
        uav_k_factor=50.0,
    )

    obs_dim = int(env.get_observation(0).shape[0])
    share_obs_dim = int(env.get_all_observations().shape[0])
    act_dim = int(env.action_space.nvec[0])

    args = SimpleNamespace(
        lr=1e-3,
        critic_lr=1e-4,
        opti_eps=1e-5,
        weight_decay=0.0,
        hidden_size=1024,
        hidden_dims=[1024, 512],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = R_MAPPOPolicy(
        args=args,
        obs_space=obs_dim,
        share_obs_space=share_obs_dim,
        act_space=act_dim,
        device=device,
    )
    name_checkpoint = f"mappo_checkpoint_{check_point}.pt" if check_point is not None else "mappo_checkpoint.pt"
    ckpt_info = infer_checkpoint(name_checkpoint, map_location=device)
    if ckpt_info is None:
        raise FileNotFoundError("Checkpoint path is None")

    actor_state_dict = ckpt_info.get("actor_state_dict")
    critic_state_dict = ckpt_info.get("critic_state_dict")
    if actor_state_dict is None or critic_state_dict is None:
        raise KeyError("Checkpoint does not contain 'actor' or 'critic' weights")

    policy.actor.load_state_dict(actor_state_dict)
    policy.critic.load_state_dict(critic_state_dict)
    policy.eval_mode()

    obs, info = env.reset(seed=42, random_walk=False)
    share_obs = np.repeat(info["share_obs"][None, :], num_agents, axis=0).astype(np.float32)

    total_reward = 0.0
    served_total_last_step = 0
    served_by_uav_last_step = np.zeros((num_agents,), dtype=np.int32)
    uav_paths = [[env.uav_states[i].copy()] for i in range(num_agents)]

    for _ in range(episode_length):
        with torch.no_grad():
            _, actions, _, _, _ = policy.get_actions(
                share_obs=share_obs,
                obs=obs,
                rnn_states_actor=None,
                rnn_states_critic=None,
                masks=None,
                available_actions=None,
                deterministic=False,
            )

        actions_np = actions.detach().cpu().numpy().astype(np.int64)
        next_obs, rewards, terminated, truncated, info = env.step(actions_np.tolist(), time_step=1)

        total_reward += float(np.mean(rewards))
        obs = next_obs
        share_obs = np.repeat(info["share_obs"][None, :], num_agents, axis=0).astype(np.float32)
        served_total_last_step = int(info.get("connected_users", 0))
        served_by_uav_last_step = np.asarray(
            info.get("uav_serviced_counts", np.zeros((num_agents,), dtype=np.int32)),
            dtype=np.int32,
        )
        for i in range(num_agents):
            uav_paths[i].append(env.uav_states[i].copy())

        if terminated or truncated:
            break

    print(f"Loaded checkpoint: {ckpt_info['checkpoint_path']}")
    print(f"Infer reward={total_reward:.3f}")
    print(f"served_total_last_step={served_total_last_step}")
    print(f"served_by_uav_last_step={served_by_uav_last_step.tolist()}")

    snapshot = env.get_visualization_snapshot()
    output_path = "Results/image/infer_assignment.png"
    plot_assignment_snapshot(snapshot, save_path=output_path, show=True, uav_paths=uav_paths)
    print(f"Saved assignment figure to {output_path}")
    

if __name__ == "__main__":
    
    main()
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from Channel.UAV import UAV
from Channel.User import User
from Channel.channel_model import Channel_Model_UAV
from Channel.channel_model import Channel_Model_mBS
from utils import calculate_rate


class MultiUAVEnv(gym.Env):
    def __init__(
        self,
        start_pos=(1000, 1000),
        max_steps=100,
        nums_UAV=1,
        user_matrix=None,
        grid_size=30,
        user_walk_speed=1.0,
        uav_k_factor=50.0,
    ):
        super(MultiUAVEnv, self).__init__()
        self.nums_UAV = nums_UAV
        self.grid_size = grid_size
        self.start_pos = np.array(start_pos, dtype=np.float32)
        self.initial_user_specs = [
            (float(x), float(y), float(rate_threshold)) for x, y, rate_threshold in user_matrix
        ] if user_matrix is not None else []
        self.user_matrix = [User(x, y, rate_threshold) for x, y, rate_threshold in self.initial_user_specs]
        self.channel_uav = Channel_Model_UAV(f_c=5.8e9, alpha=2.7, sigma2_dbm=-90, k_factor=uav_k_factor)
        self.channel_mbs = Channel_Model_mBS(f_c=2e9, sigma2_dbm=-90)
        self.UAV = UAV(height=120, velocity=5, p_tx_uav_dbm=30)
        self.mbs_height = 35
        self.p_tx_mbs_dbm = 46
        self.sigma_logf = 2
        self.user_walk_speed = float(user_walk_speed)
        self.map_min = -1000.0
        self.map_max = 1000.0
        self.min_uav_separation = 250.0
        self.crowding_penalty_coef = 0.25
        self.last_uav_serviced_counts = np.zeros(self.nums_UAV, dtype=np.float32)
        self.last_connected_users = 0
        self.last_mbs_served_users = 0
        
        self.action_space = spaces.MultiDiscrete([5] * self.nums_UAV)
        
        # Calculate observation dimension:
        # pos_k (2) + pos_mbs (2) + others_pos (2*(nums_UAV-1))
        # + user_heatmap (grid_size^2) + assignment_ratio (nums_UAV + 1)
        obs_dim = 2 + 2 + 2 * (self.nums_UAV - 1) + self.grid_size * self.grid_size + (self.nums_UAV + 1)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.nums_UAV, obs_dim), dtype=np.float32)
        share_obs_dim = 2 * self.nums_UAV + 2 + self.grid_size * self.grid_size + (self.nums_UAV + 1)
        self.share_observation_space = spaces.Box(low=-1, high=1, shape=(share_obs_dim,), dtype=np.float32)
        
        self.uav_states = np.tile(start_pos, (nums_UAV, 1)).astype(np.float32)
        self.mBS_pos = np.array([1000.0, 1000.0], dtype=np.float32)
        self.max_steps = max_steps
        self.time_slot = 0
        

    def reset(self, *, seed=None, random_walk=False):
        super().reset(seed=seed)
        self.uav_states = np.tile(self.start_pos, (self.nums_UAV, 1)).astype(np.float32)
        self.time_slot = 0
        
        self.user_matrix = [User(x, y, rate_threshold) for x, y, rate_threshold in self.initial_user_specs]
        self.last_uav_serviced_counts = np.zeros(self.nums_UAV, dtype=np.float32)
        self.last_connected_users = 0
        self.last_mbs_served_users = 0
        init_uav_serviced, init_connected_users, init_mbs_served = self.evaluate_connections(
            channel_samples=self._sample_channel_state()
        )
        self.last_uav_serviced_counts = init_uav_serviced.astype(np.float32)
        self.last_connected_users = int(init_connected_users)
        self.last_mbs_served_users = int(init_mbs_served)
        if random_walk:
            self._random_walk_users(time_step=1)
        
        observations = np.stack(
            [self.get_observation(i) for i in range(self.nums_UAV)],
            axis=0,
        ).astype(np.float32)
        info = {"share_obs": self.get_all_observations()}
        return observations, info

    def step(self, actions, time_step = 1):
            if np.isscalar(actions):
                actions = [int(actions)]

            if len(actions) != self.nums_UAV:
                raise ValueError(f"Expected {self.nums_UAV} actions, got {len(actions)}")

            self.time_slot += 1
            channel_samples = self._sample_channel_state()
            prev_uav_serviced, prev_connected_users, _ = self.evaluate_connections(
                channel_samples=channel_samples
            )

            for i, action in enumerate(actions):
                next_state = self.uav_states[i].copy()

                if action == 1:
                    next_state[1] += self.UAV.velocity * time_step  # North
                elif action == 2:
                    next_state[1] -= self.UAV.velocity * time_step  # South
                elif action == 3:
                    next_state[0] -= self.UAV.velocity * time_step  # West
                elif action == 4:
                    next_state[0] += self.UAV.velocity * time_step  # East

                if np.all((next_state >= -1000) & (next_state <= 1000)):
                    self.uav_states[i] = next_state

            self._random_walk_users(time_step)

            current_uav_serviced, current_connected_users, current_mbs_served = self.evaluate_connections(
                channel_samples=channel_samples
            )
            self.last_uav_serviced_counts = current_uav_serviced.astype(np.float32)
            self.last_connected_users = int(current_connected_users)
            self.last_mbs_served_users = int(current_mbs_served)

            if current_connected_users > prev_connected_users:
                gt = 1
            elif current_connected_users < prev_connected_users:
                gt = -1
            else:
                gt = 0

            rewards = []
            wl = 0.7
            # crowding_penalties = self._compute_crowding_penalties()
            for i in range(self.nums_UAV):
                if current_uav_serviced[i] > prev_uav_serviced[i]:
                    lt_k = 1
                elif current_uav_serviced[i] < prev_uav_serviced[i]:
                    lt_k = -1
                else:
                    lt_k = 0
                base_reward = wl * lt_k + (1 - wl) * gt
                rewards.append(base_reward )

            truncated = self.time_slot >= self.max_steps
            observations = np.stack(
                [self.get_observation(i) for i in range(self.nums_UAV)],
                axis=0,
            ).astype(np.float32)
            info = {
                "share_obs": self.get_all_observations(),
                "connected_users": int(current_connected_users),
                "uav_serviced_counts": current_uav_serviced.astype(np.int32),
                "mbs_served_users": int(current_mbs_served),
                # "uav_crowding_penalty": crowding_penalties.astype(np.float32),
            }
            return observations, np.asarray(rewards, dtype=np.float32), False, truncated, info

    def _compute_crowding_penalties(self):
        penalties = np.zeros((self.nums_UAV,), dtype=np.float32)
        if self.nums_UAV <= 1:
            return penalties

        for i in range(self.nums_UAV):
            for j in range(self.nums_UAV):
                if i == j:
                    continue
                dist = np.linalg.norm(self.uav_states[i] - self.uav_states[j])
                if dist < self.min_uav_separation:
                    penalties[i] += (self.min_uav_separation - dist) / self.min_uav_separation
        return penalties

    def _sample_channel_state(self):
        num_users = len(self.user_matrix)
        return {
            "uav_fading_power": self.channel_uav.sample_fading_power((num_users, self.nums_UAV)),
            "mbs_shadowing": np.random.normal(0.0, self.sigma_logf, size=(num_users,)).astype(np.float32),
        }

    def evaluate_connections(self, channel_samples=None):
        serviced_counts = np.zeros(self.nums_UAV, dtype=np.float32)
        connected_count = 0
        mbs_served_count = 0

        for user_idx, user in enumerate(self.user_matrix):
            mbs_shadow = None
            if channel_samples is not None:
                mbs_shadow = float(channel_samples["mbs_shadowing"][user_idx])
            d_mbs = np.sqrt((user.x - self.mBS_pos[0]) ** 2 + (user.y - self.mBS_pos[1]) ** 2)
            snr_mbs = self.channel_mbs.get_snr(
                d_2D=d_mbs,
                h_mBS=self.mbs_height,
                p_tx_dbm=self.p_tx_mbs_dbm,
                sigma_logf=self.sigma_logf,
                shadowing_db=mbs_shadow,
            )

            best_snr_uav = -np.inf
            best_uav_idx = -1
            for i in range(self.nums_UAV):
                uav_fading = None
                if channel_samples is not None:
                    uav_fading = float(channel_samples["uav_fading_power"][user_idx, i])
                d_uav = np.sqrt((user.x - self.uav_states[i][0]) ** 2 + (user.y - self.uav_states[i][1]) ** 2)
                snr_uav = self.channel_uav.get_snr(
                    d_2D=d_uav,
                    h_UAV=self.UAV.h,
                    p_tx_dbm=self.UAV.p_tx_uav_dbm,
                    fading_power=uav_fading,
                )
                if snr_uav > best_snr_uav:
                    best_snr_uav = snr_uav
                    best_uav_idx = i

            max_snr = max(snr_mbs, best_snr_uav)
            rate = calculate_rate(max_snr, 20e6)
            
            if rate >= user.rate_threshold:
                user.connected = True
                connected_count += 1
                if best_snr_uav >= snr_mbs:
                    serviced_counts[best_uav_idx] += 1
                else:
                    mbs_served_count += 1
            else:
                user.connected = False
        return serviced_counts, connected_count, mbs_served_count

    def _get_assignment_ratios(self):
        total_users = max(1, len(self.user_matrix))
        assignment_counts = np.concatenate(
            [
                self.last_uav_serviced_counts.astype(np.float32),
                np.array([self.last_mbs_served_users], dtype=np.float32),
            ],
            axis=0,
        )
        return assignment_counts / float(total_users)

    def get_visualization_snapshot(self, channel_samples=None):
        if channel_samples is None:
            channel_samples = self._sample_channel_state()

        user_positions = np.array([[user.x, user.y] for user in self.user_matrix], dtype=np.float32)
        num_users = user_positions.shape[0]
        connected_mask = np.zeros((num_users,), dtype=bool)
        # -1: unsatisfied, 0: mBS, 1..nums_UAV: UAV index + 1
        assignment = np.full((num_users,), -1, dtype=np.int32)

        for user_idx, user in enumerate(self.user_matrix):
            mbs_shadow = float(channel_samples["mbs_shadowing"][user_idx])
            d_mbs = np.sqrt((user.x - self.mBS_pos[0]) ** 2 + (user.y - self.mBS_pos[1]) ** 2)
            snr_mbs = self.channel_mbs.get_snr(
                d_2D=d_mbs,
                h_mBS=self.mbs_height,
                p_tx_dbm=self.p_tx_mbs_dbm,
                sigma_logf=self.sigma_logf,
                shadowing_db=mbs_shadow,
            )

            best_server = 0
            best_snr = snr_mbs
            for i in range(self.nums_UAV):
                uav_fading = float(channel_samples["uav_fading_power"][user_idx, i])
                d_uav = np.sqrt((user.x - self.uav_states[i][0]) ** 2 + (user.y - self.uav_states[i][1]) ** 2)
                snr_uav = self.channel_uav.get_snr(
                    d_2D=d_uav,
                    h_UAV=self.UAV.h,
                    p_tx_dbm=self.UAV.p_tx_uav_dbm,
                    fading_power=uav_fading,
                )
                if snr_uav > best_snr:
                    best_snr = snr_uav
                    best_server = i + 1

            rate = calculate_rate(best_snr, 20e6)
            if rate >= user.rate_threshold:
                connected_mask[user_idx] = True
                assignment[user_idx] = best_server

        return {
            "user_positions": user_positions,
            "connected_mask": connected_mask,
            "assignment": assignment,
            "uav_positions": self.uav_states.copy(),
            "mbs_position": self.mBS_pos.copy(),
            "map_min": self.map_min,
            "map_max": self.map_max,
        }

    def _random_walk_users(self, time_step):
        if not self.user_matrix:
            return
        move_distance = self.user_walk_speed * float(time_step)
        for user in self.user_matrix:
            direction = np.random.uniform(0.0, 2.0 * np.pi)
            user.x += move_distance * np.cos(direction)
            user.y += move_distance * np.sin(direction)
            user.x = float(np.clip(user.x, self.map_min, self.map_max))
            user.y = float(np.clip(user.y, self.map_min, self.map_max))

    def generate_heatmap(self, grid_size=None):
        if grid_size is None:
            grid_size = self.grid_size

        heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
        if not self.user_matrix:
            return heatmap.flatten()

        map_min = -1000
        cell_size = (2.0 * 1000) / grid_size

        total_counts = np.zeros((grid_size, grid_size), dtype=np.float32)
        served_counts = np.zeros((grid_size, grid_size), dtype=np.float32)

        for user in self.user_matrix:
            x_idx = int((user.x - map_min) / cell_size)
            y_idx = int((user.y - map_min) / cell_size)
            x_idx = int(np.clip(x_idx, 0, grid_size - 1))
            y_idx = int(np.clip(y_idx, 0, grid_size - 1))

            total_counts[y_idx, x_idx] += 1.0
            if user.connected:
                served_counts[y_idx, x_idx] += 1.0

        total_counts_norm = total_counts / max(1, len(self.user_matrix))
        served_ratio = np.divide(
            served_counts,
            total_counts,
            out=np.zeros_like(served_counts),
            where=total_counts > 0,
        )

        # Blend user density and service ratio in each cell.
        heatmap = 0.5 * total_counts_norm + 0.5 * served_ratio

        return heatmap.flatten()

    def get_observation(self, agent_idx):
        pos_k = self.uav_states[agent_idx] / 1000.0 # Normalization về [-1, 1]
        
        pos_mbs = self.mBS_pos / 1000.0
        
        others_pos = np.delete(self.uav_states, agent_idx, axis=0) / 1000.0
        others_pos_flat = others_pos.flatten()

        user_heatmap = self.generate_heatmap(grid_size=self.grid_size)
        assignment_ratio = self._get_assignment_ratios()
        
        obs = np.concatenate([pos_k, pos_mbs, others_pos_flat, user_heatmap, assignment_ratio]).astype(np.float32)
        # obs = spaces.utils.flatten(obs) 
        return obs
    def get_all_observations(self):

        all_uav_pos = self.uav_states.flatten() / 1000.0
        mbs_pos = self.mBS_pos / 1000.0
        heatmap = self.generate_heatmap(grid_size=self.grid_size)
        assignment_ratio = self._get_assignment_ratios()
        state = np.concatenate([all_uav_pos, mbs_pos, heatmap, assignment_ratio]).astype(np.float32)
        return state

    def render(self, mode='human'):
        pass  
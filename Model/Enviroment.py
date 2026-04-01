import gym
from gymnasium import spaces
from Channel.UAV import UAV
from Channel.channel_model import Channel_Model_UAV
from Channel.channel_model import Channel_Model_mBS
from Channel.mBS import mBS
from Channel.User import User
import numpy as np
from utils import calculate_rate

class MultiUAVEnv(gym.Env):
    def __init__(
        self,
        start_pos=(0, 0),
        max_steps=100,
        nums_UAV=1,
        user_matrix=None,
        grid_size=10,
    ):
        super(MultiUAVEnv, self).__init__()
        self.nums_UAV = nums_UAV
        self.grid_size = grid_size
        self.start_pos = np.array(start_pos, dtype=np.float32)
        self.user_matrix = [User(x, y, rate_threshold) for x, y, rate_threshold in user_matrix] if user_matrix is not None else []
        self.channel_uav = Channel_Model_UAV()
        self.channel_mbs = Channel_Model_mBS()
        self.UAV = UAV(height=120, velocity=5, p_tx_uav_dbm=30)
        
        self.action_space = spaces.MultiDiscrete([5] * self.nums_UAV)
        
        # Calculate observation dimension: 
        # pos_k (2) + pos_mbs (2) + others_pos (2*(nums_UAV-1)) + user_heatmap (grid_size^2)
        obs_dim = 2 + 2 + 2 * (self.nums_UAV - 1) + self.grid_size * self.grid_size
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.nums_UAV, obs_dim), dtype=np.float32)
        
        self.uav_states = np.tile(start_pos, (nums_UAV, 1)).astype(np.float32)
        self.mBS_pos = np.array([1000.0, 1000.0], dtype=np.float32)
        self.max_steps = max_steps
        self.time_slot = 0
        

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.uav_states = np.tile(self.start_pos, (self.nums_UAV, 1)).astype(np.float32)
        self.time_slot = 0
        for user in self.user_matrix:
            user.connected = False

        state = self.get_all_observations()
        info = {}
        return state, info
    
    def step(self, actions, time_step = 1):
            if np.isscalar(actions):
                actions = [int(actions)]

            if len(actions) != self.nums_UAV:
                raise ValueError(f"Expected {self.nums_UAV} actions, got {len(actions)}")

            self.time_slot += 1
            prev_uav_serviced, prev_connected_users = self.evaluate_connections()

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

            current_uav_serviced, current_connected_users = self.evaluate_connections()

            if current_connected_users > prev_connected_users:
                gt = 1
            elif current_connected_users < prev_connected_users:
                gt = -1
            else:
                gt = 0

            rewards = []
            wl = 0.5 
            for i in range(self.nums_UAV):
                if current_uav_serviced[i] > prev_uav_serviced[i]:
                    lt_k = 1
                elif current_uav_serviced[i] < prev_uav_serviced[i]:
                    lt_k = -1
                else:
                    lt_k = 0
                rewards.append(wl * lt_k + (1 - wl) * gt)

            truncated = self.time_slot >= self.max_steps
            observations = self.get_all_observations()
            return observations, np.asarray(rewards, dtype=np.float32), False, truncated, {}

    def evaluate_connections(self):
        serviced_counts = np.zeros(self.nums_UAV, dtype=np.float32)
        connected_count = 0

        for user in self.user_matrix:
            d_mbs = np.sqrt((user.x - self.mBS_pos[0]) ** 2 + (user.y - self.mBS_pos[1]) ** 2)
            snr_mbs = self.channel_mbs.get_snr(d_mbs)

            best_snr_uav = -np.inf
            best_uav_idx = -1
            for i in range(self.nums_UAV):
                d_uav = np.sqrt((user.x - self.uav_states[i][0]) ** 2 + (user.y - self.uav_states[i][1]) ** 2)
                snr_uav = self.channel_uav.get_snr(d_uav)
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
                user.connected = False
        return serviced_counts, connected_count

    def generate_unsatisfied_heatmap(self):

        heatmap = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        if not self.user_matrix:
            return heatmap.flatten()

        map_min = -1000
        cell_size = (2.0 * 1000) / self.grid_size

        for user in self.user_matrix:
            if user.connected:
                continue

            x_idx = int((user.x - map_min) / cell_size)
            y_idx = int((user.y - map_min) / cell_size)
            x_idx = int(np.clip(x_idx, 0, self.grid_size - 1))
            y_idx = int(np.clip(y_idx, 0, self.grid_size - 1))
            heatmap[y_idx, x_idx] += 1.0

        heatmap /= max(1, len(self.user_matrix))
        return heatmap.flatten()

    def get_observation(self, agent_idx):
        pos_k = self.uav_states[agent_idx] / 1000.0 # Normalization về [-1, 1]
        
        pos_mbs = self.mBS_pos / 1000.0
        
        others_pos = np.delete(self.uav_states, agent_idx, axis=0) / 1000.0
        others_pos_flat = others_pos.flatten()

        user_heatmap = self.generate_unsatisfied_heatmap(grid_size=self.grid_size)
        
        obs = np.concatenate([pos_k, pos_mbs, others_pos_flat, user_heatmap]).astype(np.float32)
        # obs = spaces.utils.flatten(obs) 
        return obs
    def get_all_observations(self):
        # State chứa mọi thứ: vị trí tất cả UAV, mBS, và Heatmap đầy đủ
        # Đây là thông tin "thần thánh" mà Critic sẽ dùng
        all_uav_pos = self.uav_states.flatten() / 1000.0
        mbs_pos = self.mBS_pos / 1000.0
        heatmap = self.generate_unsatisfied_heatmap()
        state = np.concatenate([all_uav_pos, mbs_pos, heatmap]).astype(np.float32)
        return state

    def render(self, mode='human'):
        pass  
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from superhexagon import SuperHexagonInterface
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class SuperHexagonGymEnv(gym.Env):
    """
    Gym env for SuperHexagon.
    Actions: 0=Stay, 1=Left, 2=Right.
    Observation: [norm wall distances, player angle, direction].
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        """Initialize env and define spaces."""
        super().__init__()
        self.env = SuperHexagonInterface(
            frame_skip=1, run_afap=True, allow_game_restart=True
        )
        self.action_space = spaces.Discrete(3)
        self.n_slots = self.env.get_num_slots()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2 + self.n_slots,), dtype=np.float32
        )
        self.episode_frames = 0

    def reset(self, seed=None, options=None):
        """Reset env and return initial observation."""
        super().reset(seed=seed)
        self.env.reset()
        self.episode_frames = 0
        obs = self._get_state()
        return obs, {}

    def step(self, action):
        """
        Take action and return (obs, reward, done, truncated, info).
        """
        _, _, done = self.env.step(action)
        self.episode_frames += 1
        obs = self._get_state()
        reward = self._get_reward(done, action)
        return obs, reward, done, False, {}

    def _get_wall_distances(self):
        """
        Compute minimal distance to wall for each slot; if there is no wall, set it to 5000.0.
        """
        min_distances = np.full(self.n_slots, np.inf, dtype=np.float32)
        for wall in self.env.get_walls():
            if wall.distance > 0 and wall.enabled:
                slot_idx = wall.slot % self.n_slots
                min_distances[slot_idx] = min(min_distances[slot_idx], wall.distance)
        min_distances[min_distances == np.inf] = 5000.0
        return min_distances

    def _get_state(self):
        """
        Build state vector: [normalized wall info, player angle, direction].
        """
        wall_info = self._get_wall_distances()
        wall_info = wall_info / wall_info.sum()
        player_angle = np.array(
            [0.5 + 0.5 * math.sin(math.radians(self.env.get_triangle_angle()))]
        )
        player_slot_idx = self.env.get_triangle_slot()
        player_slot_onehot = np.zeros(self.n_slots, dtype=np.float32)
        player_slot_onehot[player_slot_idx] = 1.0
        direction = np.array([self.get_exit_direction()], dtype=np.float32)
        state = np.concatenate([wall_info, player_angle, direction])
        return state

    def _get_reward(self, done, action):
        """
        Return reward; -10 if done, else adjust by angular distance.
        """
        debug = self.env.debug_mode
        if done:
            if debug:
                print("Game over! Return -10.0")
            return -10.0
        wall_info = self._get_wall_distances()
        wall_info = wall_info / wall_info.sum()
        player_slot_idx = self.env.get_triangle_slot()
        player_slot_onehot = np.zeros(self.n_slots, dtype=np.float32)
        player_slot_onehot[player_slot_idx] = 1.0
        reward = 1
        distance = self.get_exit_distance()
        distance_factor = distance / 360
        if distance_factor == 0:
            reward *= 1 if action == 0 else 0.7
        else:
            reward *= -distance_factor
        if debug:
            print("-------------------------------------------------------")
            print(f"Player Angle: {self.env.get_triangle_angle()}")
            print(f"Player Slot: {self.env.get_triangle_slot()}")
            print(f"Player One-Hot: {player_slot_onehot}")
            print(f"Distance: {distance} and Factor {distance_factor}")
            print(f"Action Taken: {action}")
            print(f"State: {self._get_state()}")
            print(f"Calculated Reward: {reward}")
            print("-------------------------------------------------------\n")
        return float(reward)

    def get_exit_distance(self):
        """
        Return minimal angular distance to a wall gap.
        """
        num_slots = self.env.get_num_slots()
        wall_info = self._get_wall_distances()
        wall_info = wall_info / wall_info.sum()
        player_angle = self.env.get_triangle_angle()
        player_slot = self.env.get_triangle_slot()
        left_border = 0
        right_border = 0
        if wall_info[player_slot] > wall_info.min():
            return 0
        for i in range(player_slot, player_slot + num_slots):
            index = i % num_slots
            if wall_info[index] > wall_info.min():
                left_border = index * (360 / num_slots)
                break
        for i in range(player_slot, player_slot - num_slots, -1):
            index = i % num_slots
            if wall_info[index] > wall_info.min():
                right_border = (index + 1) * (360 / num_slots)
                break
        left_distance = min(
            abs(player_angle - left_border), 360 - abs(player_angle - left_border)
        )
        right_distance = min(
            abs(player_angle - right_border), 360 - abs(player_angle - right_border)
        )
        return min(left_distance, right_distance)

    def get_exit_direction(self):
        """
        Return direction indicator (0.0 for left, 1.0 for right) to the next wall gap.
        """
        num_slots = self.env.get_num_slots()
        wall_info = self._get_wall_distances()
        wall_info = wall_info / wall_info.sum()
        player_angle = self.env.get_triangle_angle()
        player_slot = self.env.get_triangle_slot()
        left_border = 0
        right_border = 0
        if wall_info[player_slot] > wall_info.min():
            return 0.5
        for i in range(player_slot, player_slot + num_slots):
            index = i % num_slots
            if wall_info[index] > wall_info.min():
                left_border = index * (360 / num_slots)
                break
        for i in range(player_slot, player_slot - num_slots, -1):
            index = i % num_slots
            if wall_info[index] > wall_info.min():
                right_border = (index + 1) * (360 / num_slots)
                break
        left_distance = min(
            abs(player_angle - left_border), 360 - abs(player_angle - left_border)
        )
        right_distance = min(
            abs(player_angle - right_border), 360 - abs(player_angle - right_border)
        )
        return 0.0 if left_distance < right_distance else 1.0

    def render(self, mode="human"):
        """Render method (not implemented)."""
        pass


class EntropyCoefficientScheduler(BaseCallback):
    """
    Callback to linearly schedule the entropy coefficient during training.
    """

    def __init__(self, total_timesteps: int, initial_ent_coef: float, final_ent_coef: float, verbose: int = 0):
        """
        **\_\_init\_\_**
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef

    def _on_step(self) -> bool:
        """
        **_on_step**
        """
        progress = 1.0 - (self.num_timesteps / self.total_timesteps)
        new_ent_coef = self.final_ent_coef + progress * (
            self.initial_ent_coef - self.final_ent_coef
        )
        self.model.ent_coef = new_ent_coef
        return True


if __name__ == "__main__":
    """
    Trains a PPO agent on the SuperHexagonGymEnv using an entropy coefficient scheduler.
    """
    env = SuperHexagonGymEnv()
    total_timesteps = 1_000_000
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cpu",
        n_steps=4096,
        batch_size=32,
        learning_rate=2e-4,
        gamma=0.999,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.6,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[512, 512, 256]),
    )
    ent_coef_scheduler = EntropyCoefficientScheduler(
        total_timesteps=total_timesteps, initial_ent_coef=0.1, final_ent_coef=0.005
    )
    model.learn(total_timesteps=total_timesteps, callback=ent_coef_scheduler)
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from superhexagon import SuperHexagonInterface


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
        """Take action and return (obs, reward, done, truncated, info)."""
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

    def _get_norm_wall_distances(self):
        """
        Compute minimal distance to wall for each slot in normalized space [0-1]
        """
        wall_distances = self._get_wall_distances()
        wall_distances = wall_distances / wall_distances.sum()
        return wall_distances

    def _get_norm_player_angle(self):
        """
        Calculates the angle of the player as a continuous variable, by converting it into a sinus function and scaling/offsetting it.
        This was done to combat the sudden jump from 360° to 0°. We observed that the agent was "confused" by this sudden change.

        Returns:
            (float): Normalized player angle in space [0-1]
        """
        return 0.5 + 0.5 * math.sin(math.radians(self.env.get_triangle_angle()))

    def _get_state(self):
        """
        Gathers all observations in a numpy array

        Example output: `[0.80090751, 0.5, 0.0967198, 0.3832005, 0.02007971, 0.3832005, 0.0967198, 0.02007971]`

        Returns:
            (ndarray[float]): A one-dimensional numpy array representing all observations
        """
        return np.concatenate(
            [
                np.array([self._get_norm_player_angle()]),
                np.array([self._get_direction_indicator()]),
                self._get_norm_wall_distances(),
            ]
        )

    def _get_reward(self, done, action):
        """
        Calculations the reward that the agent gets for the current state and the action it took.

        Args:
            done (bool): If True, gives out max penalty because the agent hit a wall; otherwise normal calculated reward.
            action (int): Action that the agent took for this step. Used for some reward dependencies.

        Returns:
            (float): A positive reward value or a negative penalty value
        """
        debug = self.env.debug_mode
        if done:
            if debug:
                print("Game over! Return -10.0")
            return -10.0
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

    def _get_direction_indicator(self):
        """
        Calculates the closet direction to a free slot and encodes it into three buckets

        Returns:
            (float): direction indicator to closet free slot; 0.0: left, 1.0: right, 0.5: already in safe slot
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

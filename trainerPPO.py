import math

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from superhexagon import SuperHexagonInterface
from stable_baselines3 import PPO


class SuperHexagonGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self.env = SuperHexagonInterface(
            frame_skip=1, run_afap=True, allow_game_restart=True
        )
        self.action_space = spaces.Discrete(3)  # Left, Right, Stay

        self.n_slots = self.env.get_num_slots()
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2 + 2 * self.n_slots,),  # Add a comma to make it a tuple
            dtype=np.float32,
        )
        self.episode_frames = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()
        self.episode_frames = 0

        obs = self._get_state()
        return obs, {}

    def step(self, action):
        _, _, done = self.env.step(action)
        self.episode_frames += 1

        # Wall-Info und Zustand
        obs = self._get_state()

        # Reward
        reward = self._get_reward(done, action)

        # Falls du eine maximale Episodenlänge haben möchtest, kannst du hier
        # done oder truncated setzen. Hier nur done = True, truncated = False.
        return obs, reward, done, False, {}

    def _compute_wall_info(self):
        """Gibt für jeden Slot den minimalen Abstand zur Wand zurück (continuous distances)."""
        min_distances = np.full(self.n_slots, np.inf, dtype=np.float32)
        for wall in self.env.get_walls():
            if wall.distance > 0 and wall.enabled:
                slot_idx = wall.slot % self.n_slots
                min_distances[slot_idx] = min(min_distances[slot_idx], wall.distance)

        # Ersetze Inf durch 5000.0
        min_distances[min_distances == np.inf] = 5000.0
        return min_distances

    def _get_state(self):
        wall_info = self._compute_wall_info()
        wall_info = wall_info / wall_info.sum()

        player_angle = np.array(
            [math.sin(0.5 + 0.5 * math.radians(self.env.get_triangle_angle()))]
        )

        # Player slot als One-Hot:
        player_slot_idx = self.env.get_triangle_slot()
        player_slot_onehot = np.zeros(self.n_slots, dtype=np.float32)
        player_slot_onehot[player_slot_idx] = 1.0

        # Beispiel: Wir wollen den Level als float beibehalten:
        base_state = np.array(
            [
                self.env.get_level(),
            ],
            dtype=np.float32,
        )

        # Rohzustand: [base_state, wall_info, player_slot_onehot, player_angle]
        state = np.concatenate(
            [base_state, wall_info, player_slot_onehot, player_angle]
        )

        return state

    def _get_reward(self, done, action):
        # 1) Falls das Spiel vorbei ist, gib einen starken Malus
        if done:
            return -100.0

        # 2) Hole das Distanz-Array
        distances = (
            self._compute_wall_info()
        )  # => z.B. array(float32) mit length = n_slots
        player_slot = self.env.get_triangle_slot()

        # 3) Bestimme das globale Minimum und Maximum
        d_min = distances.min()
        d_max = distances.max()

        # 4) Falls alle Distanzen identisch sind
        if d_min == d_max:
            # dann sind alle Slots gleich gut, wir setzen auf 0.0 (mittig in [-0.1,0.1])
            scaled = np.full_like(distances, 0.0)
        else:
            # a) Normalisierung auf [0,1]
            x = (distances - d_min) / (d_max - d_min)
            # b) Verschiebung auf [-0.1,0.1]
            scaled = 0.2 * x - 0.1

        # 5) Wert für den Player-Slot (in [-0.1,0.1])
        reward = scaled[player_slot]

        if action == 0:
            # Aktion "Stehen"
            if reward > 0:
                # Richtiger Slot -> stark belohnen
                reward *= 2.5
            else:
                # Falscher Slot -> stark bestrafen
                reward *= 10
        else:
            # Aktion "Gehen"
            if reward > 0:
                # Richtiger Slot -> bestrafen
                reward *= -1
            else:
                # Falscher Slot -> belohnen (das negative Reward "umdrehen")
                reward *= -7.5

        return float(reward)

    def render(self, mode="human"):
        pass


if __name__ == "__main__":
    env = SuperHexagonGymEnv()
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cpu",
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.999,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.7,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    model.learn(total_timesteps=100_000_000)

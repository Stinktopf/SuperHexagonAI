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
        # Beobachtungsraum: Wir erwarten hier uint8-Bilder (0-255) und einen float-Vektor
        self.observation_space = spaces.Dict(
            {
                "vector": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(2 + 2 * self.n_slots,),
                    dtype=np.float32,
                ),
                "img_large": spaces.Box(
                    low=0,
                    high=255,
                    shape=(60, 60, 1),
                    dtype=np.uint8,
                ),
                "img_small": spaces.Box(
                    low=0,
                    high=255,
                    shape=(60, 60, 1),
                    dtype=np.uint8,
                ),
            }
        )
        self.episode_frames = 0

        # Platzhalter für die zuletzt empfangenen Bilder
        self.last_large_img = np.zeros((60, 60, 1), dtype=np.uint8)
        self.last_small_img = np.zeros((60, 60, 1), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        # Setze das Environment zurück und erhalte die Screenshots als Tuple
        large_img, small_img = self.env.reset()
        self.episode_frames = 0

        # Falls die Bilder 2D vorliegen, erweitern wir sie auf (60, 60, 1)
        if large_img.ndim == 2:
            large_img = np.expand_dims(large_img, axis=-1)
        if small_img.ndim == 2:
            small_img = np.expand_dims(small_img, axis=-1)

        # Konvertiere boolesche Bilder in uint8 (0 oder 255), falls nötig
        if large_img.dtype != np.uint8:
            large_img = (large_img.astype(np.uint8)) * 255
        if small_img.dtype != np.uint8:
            small_img = (small_img.astype(np.uint8)) * 255

        self.last_large_img = large_img
        self.last_small_img = small_img

        obs = self._get_state()
        return obs, {}

    def step(self, action):
        (large_img, small_img), _, done = self.env.step(action)
        self.episode_frames += 1

        # Falls die Bilder 2D vorliegen, erweitern wir sie auf (60, 60, 1)
        if large_img.ndim == 2:
            large_img = np.expand_dims(large_img, axis=-1)
        if small_img.ndim == 2:
            small_img = np.expand_dims(small_img, axis=-1)

        # Konvertiere boolesche Bilder in uint8 (0 oder 255), falls nötig
        if large_img.dtype != np.uint8:
            large_img = (large_img.astype(np.uint8)) * 255
        if small_img.dtype != np.uint8:
            small_img = (small_img.astype(np.uint8)) * 255

        self.last_large_img = large_img
        self.last_small_img = small_img

        obs = self._get_state()
        reward = self._get_reward(done, action)  # Unser eigener Reward

        # Rückgabe im Gymnasium-Format: (obs, reward, terminated, truncated, info)
        return obs, reward, done, False, {}

    def _compute_wall_info(self):
        """Gibt für jeden Slot den minimalen Abstand zur Wand zurück (continuous distances)."""
        min_distances = np.full(self.n_slots, np.inf, dtype=np.float32)
        for wall in self.env.get_walls():
            if wall.distance > 0 and wall.enabled:
                slot_idx = wall.slot % self.n_slots
                min_distances[slot_idx] = min(min_distances[slot_idx], wall.distance)

        # Ersetze Inf durch einen hohen Wert (5000.0)
        min_distances[min_distances == np.inf] = 5000.0
        return min_distances

    def _get_state(self):
        wall_info = self._compute_wall_info()
        wall_sum = wall_info.sum()
        if wall_sum == 0:
            normalized_wall_info = np.zeros_like(wall_info)
        else:
            normalized_wall_info = wall_info / wall_sum

        # Berechne den Spielerwinkel
        player_angle = np.array(
            [math.sin(0.5 + 0.5 * math.radians(self.env.get_triangle_angle()))],
            dtype=np.float32,
        )

        # Erstelle One-Hot Encoding für den Spieler-Slot
        player_slot_idx = self.env.get_triangle_slot()
        player_slot_onehot = np.zeros(self.n_slots, dtype=np.float32)
        if 0 <= player_slot_idx < self.n_slots:
            player_slot_onehot[player_slot_idx] = 1.0
        else:
            player_slot_onehot[-1] = 1.0

        # Basiszustand: Level als float
        base_state = np.array([self.env.get_level()], dtype=np.float32)

        # Kombiniere alle Zustandskomponenten
        vector_state = np.concatenate(
            [base_state, normalized_wall_info, player_slot_onehot, player_angle]
        )

        return {
            "vector": vector_state,
            "img_large": self.last_large_img,
            "img_small": self.last_small_img,
        }

    def _get_reward(self, done, action):
        wall_info = self._compute_wall_info()
        wall_sum = wall_info.sum()
        if wall_sum == 0:
            normalized_wall_info = np.zeros_like(wall_info)
        else:
            normalized_wall_info = wall_info / wall_sum

        player_slot = self.env.get_triangle_slot()

        reward = 0.0
        if done:
            return -100.0

        reward += 0.1
        # Eigene Reward-Logik:
        if normalized_wall_info[player_slot] > normalized_wall_info.min():
            if action == 0:
                reward *= 2.5
            else:
                reward *= 2.0
        else:
            reward *= -1.0

        return reward

    def render(self, mode="human"):
        # Optional: Implementiere eine Render-Funktion
        pass


if __name__ == "__main__":
    env = SuperHexagonGymEnv()
    model = PPO(
        policy="MultiInputPolicy",
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

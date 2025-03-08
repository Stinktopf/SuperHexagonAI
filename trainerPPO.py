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
            low=-1.0,
            high=1.0,
            shape=(1 + 3 * self.n_slots,),
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
        reward = self._get_reward(done)

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

        # Ersetze Inf durch 0
        min_distances[min_distances == np.inf] = 0.0
        return min_distances

    def _get_state(self):
        wall_info = self._compute_wall_info()

        # Player slot als One-Hot:
        player_slot_idx = self.env.get_triangle_slot()
        player_slot_onehot = np.zeros(self.n_slots, dtype=np.float32)
        player_slot_onehot[player_slot_idx] = 1.0

        # "Bester Slot" (größter Abstand) ebenfalls One-Hot:
        best_slot_idx = np.argmax(wall_info)
        best_slot_onehot = np.zeros(self.n_slots, dtype=np.float32)
        best_slot_onehot[best_slot_idx] = 1.0

        # Beispiel: Wir wollen den Level als float beibehalten:
        base_state = np.array(
            [
                self.env.get_level(),
                # self.env.get_triangle_angle(),
                # self.env.get_num_slots(),
                # self.env.get_num_walls(),
                # self.env.get_world_angle(),
            ],
            dtype=np.float32,
        )

        # Rohzustand: [base_state, wall_info, player_slot_onehot, best_slot_onehot]
        state_raw = np.concatenate(
            [base_state, wall_info, player_slot_onehot, best_slot_onehot]
        )

        # Normieren/Clippen (optional, hier Tanh-Beispiel)
        norm_factor = np.max(np.abs(state_raw)) + 1e-8
        state_norm = np.clip(np.tanh(state_raw / norm_factor), -1, 1)
        return state_norm

    def _get_reward(self, done):
        """Gibt nur Survival-Reward + moderate Strafe bei 'done'."""
        if done:
            return -1.0  # moderate Strafe für Tod
        else:
            return 0.1  # kleine Belohnung pro überlebtem Schritt

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
        ent_coef=0.005,
        vf_coef=0.7,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    model.learn(total_timesteps=1_000_000)

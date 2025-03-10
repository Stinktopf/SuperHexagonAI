import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from superhexagon import SuperHexagonInterface
from stable_baselines3 import PPO
from utils import Network


class SuperHexagonGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, teacher_net_path, teacher_device="cuda"):
        super().__init__()
        self.env = SuperHexagonInterface(
            frame_skip=4, run_afap=False, allow_game_restart=True
        )
        self.action_space = spaces.Discrete(3)

        self.n_frames = 4
        self.fp = np.zeros(
            (self.n_frames, *SuperHexagonInterface.frame_size), dtype=np.float32
        )
        self.fcp = np.zeros(
            (self.n_frames, *SuperHexagonInterface.frame_size_cropped), dtype=np.float32
        )

        self.observation_space = spaces.Dict(
            {
                "image1": spaces.Box(0, 1, self.fp.shape, dtype=np.float32),
                "image2": spaces.Box(0, 1, self.fcp.shape, dtype=np.float32),
            }
        )

        self.n_atoms, self.support = 51, np.linspace(-1, 0, 51)
        self.teacher_device = torch.device(teacher_device)
        self.teacher_net = Network(
            self.n_frames, SuperHexagonInterface.n_actions, self.n_atoms
        ).to(self.teacher_device)
        self.teacher_net.load_state_dict(
            torch.load(teacher_net_path, map_location=self.teacher_device)
        )
        self.teacher_net.eval()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        f, fc = self.env.reset()
        self.fp.fill(0)
        self.fcp.fill(0)
        self.fp[0], self.fcp[0] = f, fc
        return self._get_state(), {}

    def step(self, action):
        (f, fc), _, done = self.env.step(action)
        self.fp[1:], self.fcp[1:] = self.fp[:-1], self.fcp[:-1]
        self.fp[0], self.fcp[0] = f, fc
        reward = self._compute_reward(action)
        return self._get_state(), reward, done, False, {}

    def _get_state(self):
        return {"image1": self.fp.copy(), "image2": self.fcp.copy()}

    def _compute_reward(self, action):
        # Berechnung der Q-Werte aus dem Lehrer-Netzwerk
        with torch.no_grad():
            q_values = (
                self.teacher_net(
                    torch.tensor(self.fp[None], device=self.teacher_device),
                    torch.tensor(self.fcp[None], device=self.teacher_device),
                )
                .cpu()
                .numpy()
                .squeeze()
            )

        # Bestimmung der Lehrer-Aktion
        teacher_action = (q_values @ self.support).argmax()

        # Lehrer-Bonus: +1, wenn die Aktion korrekt ist, sonst -1
        teacher_reward = 1.0 if action == teacher_action else -1.0

        # Überlebensbonus: Sigmoidartige Funktion, die mit der Zeit steigt und sich einem Maximalwert annähert
        max_bonus = 5.0  # Das maximale Limit des Bonus
        growth_rate = 0.001  # Wie schnell der Bonus wächst
        survival_bonus = max_bonus * (1 - np.exp(-growth_rate * self.env.steps_alive))

        # Gesamte Belohnung: Kombination aus Lehrer-Bonus und Überlebensbonus
        total_reward = teacher_reward + survival_bonus

        # Debug-Ausgabe, wenn debug_mode aktiviert ist
        if getattr(self.env, "debug_mode", False):
            print(
                f"Step {self.env.steps_alive}: Aktion: {action} | Lehreraktion: {teacher_action} | "
                f"Lehrer-Bonus: {teacher_reward} | Überlebensbonus: {survival_bonus:.2f} | Gesamtbelohnung: {total_reward:.2f}"
            )

        return total_reward

    def render(self, mode="human"):
        pass


if __name__ == "__main__":
    env = SuperHexagonGymEnv("super_hexagon_net", teacher_device="cuda")

    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        device="cpu",
        n_steps=1024,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.95,
        gae_lambda=0.95,
        ent_coef=0.001,
        vf_coef=0.25,
        max_grad_norm=0.5,
    )

    model.learn(total_timesteps=1_000_000)

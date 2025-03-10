import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from superhexagon import SuperHexagonInterface
from stable_baselines3 import PPO
from utils import Network  # Teacher-Netz wird aus utils importiert

class SuperHexagonGymEnv(gym.Env):
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, teacher_net_path="super_hexagon_net", teacher_device="cuda",
                 teacher_weight_decay_steps=100000):
        super().__init__()
        self.env = SuperHexagonInterface(frame_skip=1, run_afap=True, allow_game_restart=True)
        self.action_space = spaces.Discrete(3)  # Aktionen: 0 = Stay, 1 = Rechts, 2 = Links

        # Vektorbasierter Zustand (traditioneller Reward)
        self.n_slots = self.env.get_num_slots()
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(2 + 2 * self.n_slots + 3,),
            dtype=np.float32,
        )

        # Teacher-Netz benötigt Bilddaten (wie im Imitationsmodell)
        self.n_frames = 4
        self.fp = np.zeros((self.n_frames, *SuperHexagonInterface.frame_size), dtype=np.float32)
        self.fcp = np.zeros((self.n_frames, *SuperHexagonInterface.frame_size_cropped), dtype=np.float32)
        self.n_atoms = 51
        self.support = np.linspace(-1, 0, self.n_atoms)

        self.episode_frames = 0
        self.last_action = 0
        self.global_step = 0

        # Teacher-Netz initialisieren wie in eval.py
        self.teacher_device = torch.device(teacher_device)
        self.teacher_net = Network(self.n_frames, SuperHexagonInterface.n_actions, self.n_atoms).to(self.teacher_device)
        self.teacher_net.load_state_dict(torch.load(teacher_net_path, map_location=self.teacher_device))
        self.teacher_net.eval()

        self.teacher_weight_decay_steps = teacher_weight_decay_steps

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        f, fc = self.env.reset()
        self.episode_frames = 0
        self.last_action = 0
        self.fp.fill(0)
        self.fcp.fill(0)
        self.fp[0] = f
        self.fcp[0] = fc
        obs = self._get_state()
        if self.env.debug_mode:
            print("=== Reset ===")
            print(f"Initialer Zustand (Vektor): {obs}")
        return obs, {}

    def step(self, action):
        (f, fc), _, done = self.env.step(action)
        self.episode_frames += 1
        self.global_step += 1

        # Update der Bild-Frames für das Teacher-Netz
        self.fp[1:], self.fcp[1:] = self.fp[:-1], self.fcp[:-1]
        self.fp[0] = f
        self.fcp[0] = fc

        # Berechnung der einzelnen Rewards
        traditional_reward = self._compute_traditional_reward(done, action)
        teacher_reward = self._compute_teacher_reward(action)
        
        # Dynamisches Teacher-Gewicht: Zu Beginn dominiert der Teacher, später geht der Fokus auf den traditionellen Reward über.
        teacher_weight = max(0.0, 1.0 - (self.global_step / self.teacher_weight_decay_steps))
        
        # Skalierung der Rewards mit dem Gewichtungsfaktor
        effective_teacher_reward = teacher_weight * teacher_reward
        effective_traditional_reward = (1 - teacher_weight) * traditional_reward
        
        # Softmax-Ansatz zur Kombination
        beta = 10.0  # Temperatur-Parameter; höhere Werte führen zu einer stärkeren Gewichtung des höheren Rewards
        exp_teacher = np.exp(beta * effective_teacher_reward)
        exp_traditional = np.exp(beta * effective_traditional_reward)
        combined_reward = (effective_teacher_reward * exp_teacher + effective_traditional_reward * exp_traditional) / (exp_teacher + exp_traditional)

        if self.env.debug_mode:
            # Berechne One-Hot Vektoren für Spieler und Best-Slot (zur besseren Veranschaulichung)
            player_slot = self.env.get_triangle_slot()
            wall_info = self._compute_wall_info()
            n = len(wall_info)
            candidate_indices = np.nonzero(wall_info > 1000)[0]
            if candidate_indices.size > 0:
                best_slot = int(min(candidate_indices, key=lambda idx: self.circular_distance(player_slot, idx, n)))
            else:
                max_val = np.max(wall_info)
                max_indices = np.nonzero(wall_info == max_val)[0]
                best_slot = int(min(max_indices, key=lambda idx: self.circular_distance(player_slot, idx, n)))
            player_one_hot = [0] * n
            best_one_hot = [0] * n
            player_one_hot[player_slot] = 1
            best_one_hot[best_slot] = 1

            # Zusätzliche Debug-Ausgaben
            print("=== Debug Info ===")
            print(f"Global Step: {self.global_step} | Episode Frame: {self.episode_frames}")
            print(f"Aktion: {action} | Letzte Aktion: {self.last_action}")
            print(f"Traditioneller Reward: {traditional_reward:.2f} | Teacher Reward: {teacher_reward:.2f}")
            print(f"Teacher Weight: {teacher_weight:.4f}")
            print(f"Effective Teacher Reward: {effective_teacher_reward:.2f} | Effective Trad. Reward: {effective_traditional_reward:.2f}")
            print(f"Kombinierter Reward (Softmax): {combined_reward:.2f}")

            # Debug zu Teacher-Netz: Q-Werte und empfohlene Aktion
            fp_tensor = torch.tensor(self.fp[None], dtype=torch.float32, device=self.teacher_device)
            fcp_tensor = torch.tensor(self.fcp[None], dtype=torch.float32, device=self.teacher_device)
            with torch.no_grad():
                q_dist = self.teacher_net(fp_tensor, fcp_tensor).cpu().numpy().squeeze()
            q_values = np.sum(q_dist * self.support, axis=1)
            teacher_action = int(np.argmax(q_values))
            print(f"Q-Werte: {q_values}")
            print(f"Empfohlene Aktion vom Teacher: {teacher_action}")
            print(f"Player One-Hot: {player_one_hot}")
            print(f"Best-Slot One-Hot: {best_one_hot}")
            print("===================")

        self.last_action = action
        obs = self._get_state()
        return obs, combined_reward, done, False, {}

    def _compute_traditional_reward(self, done, action):
        if done:
            return -5.0

        distances = self._compute_wall_info()
        n = len(distances)
        player_slot = self.env.get_triangle_slot()

        candidate_indices = np.nonzero(distances > 1000)[0]
        if candidate_indices.size > 0:
            best_slot = int(min(candidate_indices, key=lambda idx: self.circular_distance(player_slot, idx, n)))
        else:
            max_val = np.max(distances)
            max_indices = np.nonzero(distances == max_val)[0]
            best_slot = int(min(max_indices, key=lambda idx: self.circular_distance(player_slot, idx, n)))

        old_dist = self.circular_distance(player_slot, best_slot, n)
        if action == 1:
            new_slot = (player_slot + 1) % n
        elif action == 2:
            new_slot = (player_slot - 1) % n
        else:
            new_slot = player_slot
        new_dist = self.circular_distance(new_slot, best_slot, n)

        reward = 1.0
        if new_dist < old_dist:
            reward += 2.0 * (old_dist - new_dist)
            if action == self.last_action:
                reward += 1.0
        elif new_dist > old_dist:
            reward -= 2.0 * (new_dist - old_dist)
        elif new_dist > 0:
            reward -= 1.0
        else:
            reward += 1.0
            if action == self.last_action:
                reward += 1.0

        if action != self.last_action and new_dist < old_dist:
            reward += 2.0

        if new_slot != best_slot and distances[new_slot] < 500.0 and distances[player_slot] >= distances[new_slot]:
            reward -= 4.0

        unique_values = np.unique(distances)
        if (len(unique_values) == 2 and np.count_nonzero(distances == np.max(distances)) == 1 
                and player_slot != best_slot):
            reward += 5.0 if new_dist < old_dist else -5.0

        return float(reward)

    def _compute_teacher_reward(self, action):
        fp_tensor = torch.tensor(self.fp[None], dtype=torch.float32, device=self.teacher_device)
        fcp_tensor = torch.tensor(self.fcp[None], dtype=torch.float32, device=self.teacher_device)
        with torch.no_grad():
            q_dist = self.teacher_net(fp_tensor, fcp_tensor).cpu().numpy().squeeze()
        q_values = np.sum(q_dist * self.support, axis=1)
        teacher_action = int(np.argmax(q_values))
        teacher_reward = 1.0 if action == teacher_action else -1.0
        return teacher_reward

    def _compute_wall_info(self):
        min_distances = np.full(self.n_slots, np.inf, dtype=np.float32)
        for wall in self.env.get_walls():
            if wall.distance > 0 and wall.enabled:
                slot_idx = wall.slot % self.n_slots
                min_distances[slot_idx] = min(min_distances[slot_idx], wall.distance)
        min_distances[min_distances == np.inf] = 5000.0
        return min_distances

    def _get_state(self):
        wall_info = self._compute_wall_info()
        wall_info = wall_info / wall_info.sum()
        player_angle = np.array([math.sin(0.5 + 0.5 * math.radians(self.env.get_triangle_angle()))])
        player_slot_idx = self.env.get_triangle_slot()
        player_slot_onehot = np.zeros(self.n_slots, dtype=np.float32)
        player_slot_onehot[player_slot_idx] = 1.0
        base_state = np.array([self.env.get_level()], dtype=np.float32)
        last_action_onehot = np.zeros(3, dtype=np.float32)
        last_action_onehot[self.last_action] = 1.0
        state = np.concatenate([base_state, wall_info, player_slot_onehot, last_action_onehot, player_angle])
        return state

    def circular_distance(self, i, j, n):
        diff = abs(i - j)
        return min(diff, n - diff)

    def render(self, mode="human"):
        pass

if __name__ == "__main__":
    env = SuperHexagonGymEnv()
    total_timesteps = 1_000_000
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
        ent_coef=0.1,
        vf_coef=0.7,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    model.learn(total_timesteps=total_timesteps)

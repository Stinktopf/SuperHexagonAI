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
    
    def __init__(self, teacher_net_path="super_hexagon_net", teacher_device="cuda"):
        super().__init__()
        self.env = SuperHexagonInterface(frame_skip=1, run_afap=True, allow_game_restart=True)
        self.action_space = spaces.Discrete(3)  # Aktionen: 0 = Stay, 1 = Rechts, 2 = Links

        self.n_slots = self.env.get_num_slots()
        # Wandinfo: 3 Werte pro Slot -> 3*n_slots (flach), aber intern 2D (n_slots,3)
        # Spieler-Slot One-Hot: n_slots, Last-Action One-Hot: 3, Winkel: 2
        obs_dim = 3 * self.n_slots + self.n_slots + 3 + 2
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
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

        # Erhalte die empfohlenen Aktionen
        teacher_action = self._get_teacher_recommendation()
        traditional_action = self._get_traditional_recommendation()

        # Einfacher Reward-Aufbau:
        base_reward = 0.25
        if action == teacher_action:
            reward = 1.0  # Basis + Teacherbonus
        elif action == traditional_action:
            reward = 1.0  # Basis + traditioneller Bonus
        else:
            reward = base_reward

        if (action == teacher_action) and (action == traditional_action):
            reward *= 3.0

        if (action == self.last_action) and (action != teacher_action) and (action != traditional_action):
            reward -= 1.0

        if self.env.debug_mode:
            # Debug-Ausgabe: Hier wollen wir die Wall-Info als 2D-Array (n_slots, 3) zeigen
            player_slot = self.env.get_triangle_slot()
            wall_info_2d = self._compute_wall_info_depth()  # 2D: (n_slots, 3)
            # Aggregiere die 3 Werte pro Slot (z. B. das Minimum) für die Best-Slot-Berechnung:
            aggregated = np.array([min(row) for row in wall_info_2d])
            candidate_indices = np.nonzero(aggregated > 1000)[0]
            if candidate_indices.size > 0:
                best_slot = int(min(candidate_indices, key=lambda idx: self.circular_distance(player_slot, idx, self.n_slots)))
            else:
                max_val = np.max(aggregated)
                max_indices = np.nonzero(aggregated == max_val)[0]
                best_slot = int(min(max_indices, key=lambda idx: self.circular_distance(player_slot, idx, self.n_slots)))
            player_one_hot = [0] * self.n_slots
            best_one_hot = [0] * self.n_slots
            player_one_hot[player_slot] = 1
            best_one_hot[best_slot] = 1

            print("=== Debug Info ===")
            print(f"Global Step: {self.global_step} | Episode Frame: {self.episode_frames}")
            print(f"Aktion: {action} | Letzte Aktion: {self.last_action}")
            print(f"Teacher Empfehlung: {teacher_action} | Traditionelle Empfehlung: {traditional_action}")
            print(f"Finaler Reward: {reward:.2f}")
            print(f"Player One-Hot: {player_one_hot}")
            print(f"Best-Slot One-Hot: {best_one_hot}")
            print(f"Wall-Info (2D):\n{wall_info_2d}")
            print("===================")

        self.last_action = action
        obs = self._get_state()
        return obs, reward, done, False, {}

    def _get_teacher_recommendation(self):
        # Berechne die Teacherempfehlung anhand des Teacher-Netzes
        fp_tensor = torch.tensor(self.fp[None], dtype=torch.float32, device=self.teacher_device)
        fcp_tensor = torch.tensor(self.fcp[None], dtype=torch.float32, device=self.teacher_device)
        with torch.no_grad():
            q_dist = self.teacher_net(fp_tensor, fcp_tensor).cpu().numpy().squeeze()
        q_values = np.sum(q_dist * self.support, axis=1)
        teacher_action = int(np.argmax(q_values))
        return teacher_action

    def _get_traditional_recommendation(self):
        distances_2d = self._compute_wall_info_depth()  # 2D: (n_slots, 3)
        # Aggregiere die 3 Werte pro Slot (z. B. Minimum)
        aggregated = np.array([min(row) for row in distances_2d])
        player_slot = self.env.get_triangle_slot()

        candidate_indices = np.nonzero(aggregated > 1000)[0]
        if candidate_indices.size > 0:
            best_slot = int(min(candidate_indices, key=lambda idx: self.circular_distance(player_slot, idx, self.n_slots)))
        else:
            max_val = np.max(aggregated)
            max_indices = np.nonzero(aggregated == max_val)[0]
            best_slot = int(min(max_indices, key=lambda idx: self.circular_distance(player_slot, idx, self.n_slots)))

        best_action = None
        best_new_dist = float('inf')
        for a in range(self.action_space.n):
            if a == 1:
                new_slot = (player_slot + 1) % self.n_slots
            elif a == 2:
                new_slot = (player_slot - 1) % self.n_slots
            else:
                new_slot = player_slot
            new_dist = self.circular_distance(new_slot, best_slot, self.n_slots)
            if new_dist < best_new_dist:
                best_new_dist = new_dist
                best_action = a
        return best_action

    def _compute_wall_info_depth(self):
        # Erzeuge ein Dictionary, das pro Slot eine Liste der Distanzen speichert.
        walls_per_slot = {slot: [] for slot in range(self.n_slots)}
        for wall in self.env.get_walls():
            if wall.distance > 0 and wall.enabled:
                slot_idx = wall.slot % self.n_slots
                walls_per_slot[slot_idx].append(wall.distance)
        
        # Erstelle für jeden Slot eine Liste mit genau 3 Werten (fehlt etwas, wird mit 5000.0 aufgefüllt)
        depth_info = []
        for slot in range(self.n_slots):
            distances = sorted(walls_per_slot[slot])
            while len(distances) < 3:
                distances.append(5000.0)
            depth_info.append(distances[:3])
        
        # Rückgabe als 2D-Array (n_slots, 3)
        return np.array(depth_info, dtype=np.float32)

    def _get_state(self):
        # Hole die 2D-Wandinfo, flache sie aber für den Zustandsvektor ab
        wall_info_2d = self._compute_wall_info_depth()
        wall_info_flat = wall_info_2d.flatten()
        wall_info_flat = wall_info_flat / wall_info_flat.sum()

        angle_rad = math.radians(self.env.get_triangle_angle())
        player_angle_sin = (math.sin(angle_rad) + 1) / 2
        player_angle_cos = (math.cos(angle_rad) + 1) / 2
        
        player_slot_idx = self.env.get_triangle_slot()
        player_slot_onehot = np.zeros(self.n_slots, dtype=np.float32)
        player_slot_onehot[player_slot_idx] = 1.0
        
        last_action_onehot = np.zeros(3, dtype=np.float32)
        last_action_onehot[self.last_action] = 1.0

        # Zustandsvektor zusammenstellen:
        # [Wandinfo (3*n_slots, flach), Spieler-Slot One-Hot (n_slots), Last-Action One-Hot (3), Winkel (2)]
        state = np.concatenate([
            wall_info_flat, 
            player_slot_onehot, 
            last_action_onehot, 
            [player_angle_sin, player_angle_cos]
        ])
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
        learning_rate=1e-4,
        gamma=0.995,
        gae_lambda=0.95,
        ent_coef=0.25,
        vf_coef=0.7,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    model.learn(total_timesteps=total_timesteps)

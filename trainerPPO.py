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

    def circular_distance(self, i, j, n):
        diff = abs(i - j)
        return min(diff, n - diff)

    def _get_reward(self, done, action):
        debug = False
        
        # Falls das Spiel vorbei ist => hoher Malus
        if done:
            if debug:
                print("Game over! Return -100")
            return -100.0

        distances = self._compute_wall_info()  # Array mit n Werten
        n = len(distances)                     # Anzahl Slots
        player_slot = self.env.get_triangle_slot()

        # Zunächst Kandidaten finden, bei denen distance > 1000 gilt
        candidate_indices = np.nonzero(distances > 1000)[0]

        if candidate_indices.size > 0:
            # Aus den Kandidaten: Wähle denjenigen, der den geringsten kreisförmigen Abstand zum Spieler hat.
            best_slot = int(min(candidate_indices, key=lambda idx: self.circular_distance(player_slot, idx, n)))
        else:
            # Falls keine Kandidaten vorhanden sind: Nimm den maximalen Slot (wie bisher)
            max_val = np.max(distances)
            max_indices = np.nonzero(distances == max_val)[0]
            best_slot = int(min(max_indices, key=lambda idx: self.circular_distance(player_slot, idx, n)))

        # Kreisförmige Distanz alt
        old_dist = self.circular_distance(player_slot, best_slot, n)

        # Aktion ausführen (1=links, 2=rechts, 0=bleiben) - modulo n
        if action == 1:
            new_slot = (player_slot - 1) % n
        elif action == 2:
            new_slot = (player_slot + 1) % n
        else:
            new_slot = player_slot

        # Kreisförmige Distanz neu
        new_dist = self.circular_distance(new_slot, best_slot, n)

        # Basierender "Reward" (näher, weiter oder gleich?)
        if new_dist < old_dist:
            reward = +0.3  * new_dist # Richtige Richtung
            direction_info = "Spin good"
        elif new_dist > old_dist:
            reward = -0.1  * new_dist # Falsche Richtung
            direction_info = "Sping bad"
        elif new_dist > 0:
            reward = -0.3  * old_dist # Stehen in einem schlechten Slot
            direction_info = "Stay good"
        else:
            reward = +1.0  # Stehen im best_slot
            direction_info = "Stay bad"

        # Erstelle One-Hot Arrays für Player- und Best-Slot
        player_one_hot = [0] * n
        player_one_hot[player_slot] = 1
        best_one_hot = [0] * n
        best_one_hot[best_slot] = 1

        # Visualisierung im Terminal
        if debug:
            print("-------------------------------------------------------")
            print(f"Distances: {distances}")
            print(f"Player Slot: {player_slot} | Best Slot: {best_slot}")
            print(f"Player One-Hot: {player_one_hot}")
            print(f"Best   One-Hot: {best_one_hot}")
            print(f"Old Distance: {old_dist:.2f} | New Slot: {new_slot} | New Distance: {new_dist:.2f}")
            print(f"Action Taken: {action} | {direction_info}")
            print(f"Calculated Reward: {reward}")
            print("-------------------------------------------------------\n")

        return float(reward)

    def render(self, mode="human"):
        pass


from stable_baselines3.common.callbacks import BaseCallback

class EntropyCoefficientScheduler(BaseCallback):
    """
    Dieser Callback passt den ent_coef-Wert des Modells linear von
    initial_ent_coef auf final_ent_coef über total_timesteps an.
    """
    def __init__(self, total_timesteps, initial_ent_coef, final_ent_coef, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef

    def _on_step(self) -> bool:
        # Berechne den aktuellen Fortschritt (1.0 -> 0.0)
        progress = 1.0 - (self.num_timesteps / self.total_timesteps)
        new_ent_coef = self.final_ent_coef + progress * (self.initial_ent_coef - self.final_ent_coef)
        # Aktualisiere den Wert im Modell
        self.model.ent_coef = new_ent_coef
        return True


if __name__ == "__main__":
    env = SuperHexagonGymEnv()
    total_timesteps = 100_000_000
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cpu",
        n_steps=4096,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.999,
        gae_lambda=0.95,
        ent_coef=0.1,
        vf_coef=0.7,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[128, 128]),
    )

    ent_coef_scheduler = EntropyCoefficientScheduler(
        total_timesteps=total_timesteps,
        initial_ent_coef=0.1,
        final_ent_coef=0.005
    )

    model.learn(total_timesteps=total_timesteps, callback=ent_coef_scheduler)

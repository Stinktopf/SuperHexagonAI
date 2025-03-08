from superhexagon import SuperHexagonInterface  # Vorausgesetzt, diese Schnittstelle ist vorhanden

# -------------------------------
# PPO-Trainer mit dualem Input
# -------------------------------
class PPOTrainer:
    def __init__(self,
                 n_frames=4,  # Anzahl Frames pro Ansicht
                 n_actions=3,
                 total_timesteps=1_000_000,
                 update_timesteps=2048,
                 device='cuda'):
        self.device = device
        self.update_timesteps = update_timesteps
        self.total_timesteps = total_timesteps

        # SuperHexagon-Interface: liefert ein Tuple (weit, zentriert)
        self.env = SuperHexagonInterface(frame_skip=4, run_afap=True, allow_game_restart=True)
        self.env.reset()  # state ist ein Tuple: (wide, crop)
        self.n_frames = n_frames
        self.n_actions = n_actions

    def select_action(self, state):
        # state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, 2*n_frames, 60,60)
        # with torch.no_grad():
        #     logits, value = self.net(state_tensor)
        # probs = F.softmax(logits, dim=-1)
        # dist = torch.distributions.Categorical(probs)
        # action = dist.sample().item()
        # log_prob = dist.log_prob(torch.tensor(action).to(self.device)).item()
        # return action, log_prob, value.item()
        return 1, 1, 1

    # def compute_gae(self, rewards, values, dones):
    #     advantages = []
    #     gae = 0
    #     # Hänge einen letzten Value für die Berechnung an
    #     values = values + [0]
    #     for step in reversed(range(len(rewards))):
    #         delta = rewards[step] + self.gamma * values[step+1] * (1 - dones[step]) - values[step]
    #         gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
    #         advantages.insert(0, gae)
    #     returns = [adv + v for adv, v in zip(advantages, values[:-1])]
    #     return advantages, returns

    def train(self):
        # Verwende self.state_stack als initialen Zustand
        timestep = 0
        episode = 0
        episode_reward = 0
        episode_frames = 0

        while timestep < self.total_timesteps:
            # Sammle update_timesteps Schritte
            for _ in range(self.update_timesteps):

                walls = self.env.get_walls()
                num_slots = self.env.get_num_walls()
                min_distances = [float('inf')] * num_slots

                for wall in walls:
                    print(wall)
                    if wall.distance > 0 and wall.enabled:
                        min_distances[wall.slot % num_slots] = min(min_distances[wall.slot % num_slots], wall.distance)

                target_slot = min_distances.index(max(min_distances))
                print({
                    "Level": self.env.get_level(),
                    "Player Angle": self.env.get_triangle_angle(),
                    "World Angle": self.env.get_world_angle(),
                    "Number Slots": self.env.get_num_slots(),
                    "Number Walls": self.env.get_num_walls(),
                    "Player Slot": self.env.get_triangle_slot(),
                    "Target Slot": target_slot
                })

                action, log_prob, value = self.select_action(None)
                next_obs, _, done = self.env.step(action)

                if done:
                    print(f"Episode {episode} beendet nach {episode_frames} Frames, Reward: {episode_reward:.2f}")
                    self.env.reset()
                    episode += 1
                    episode_reward = 0
                    episode_frames = 0

if __name__ == '__main__':
    trainer = PPOTrainer(
        n_frames=4,
        n_actions=3,
        total_timesteps=1_000_000,
        update_timesteps=2048,
        device='cuda'
    )
    trainer.train()

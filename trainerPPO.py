import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from superhexagon import SuperHexagonInterface  # Vorausgesetzt, diese Schnittstelle ist vorhanden

# -------------------------------
# Einfaches Reward Shaping (Proof-of-Concept)
# -------------------------------
def shape_reward(episode_frames, done):
    if done:
        # Bei Scheitern: Strafe, die mit mehr überlebten Frames weniger hart bestraft
        return -100 * math.log(episode_frames + 1)
    else:
        # Überlebensbonus mit Sättigungseffekt
        return 100 * math.tanh(0.01 * episode_frames)

# -------------------------------
# Definition eines Residual-Blocks
# -------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut-Verbindung anpassen, falls nötig
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# -------------------------------
# Noisy Linear Layer für verbesserte Exploration
# -------------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        self.weight_epsilon.normal_()
        self.bias_epsilon.normal_()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

# -------------------------------
# Dual-View ResNet-ähnliches Actor-Critic-Netzwerk
# -------------------------------
class DualViewResNetActorCritic(nn.Module):
    def __init__(self, n_frames, n_actions):
        """
        n_frames: Anzahl der Frames pro Ansicht (z.B. 4)
        n_actions: Anzahl möglicher Aktionen
        """
        super(DualViewResNetActorCritic, self).__init__()
        # Verarbeitung der Weitansicht
        self.conv1_wide = nn.Conv2d(n_frames, 16, kernel_size=3, stride=1, padding=1)
        self.bn1_wide = nn.BatchNorm2d(16)
        self.layer1_wide = ResidualBlock(16, 32, stride=2)    # -> (32, 30, 30)
        self.layer2_wide = ResidualBlock(32, 64, stride=2)    # -> (64, 15, 15)
        self.fc_image_wide = nn.Linear(64 * 15 * 15, 120)

        # Verarbeitung der zentrierten Ansicht
        self.conv1_crop = nn.Conv2d(n_frames, 16, kernel_size=3, stride=1, padding=1)
        self.bn1_crop = nn.BatchNorm2d(16)
        self.layer1_crop = ResidualBlock(16, 32, stride=2)    # -> (32, 30, 30)
        self.layer2_crop = ResidualBlock(32, 64, stride=2)    # -> (64, 15, 15)
        self.fc_image_crop = nn.Linear(64 * 15 * 15, 120)

        # Fusion der beiden Feature-Vektoren
        self.fc_fusion = nn.Linear(120 * 2, 84)

        # Policy- und Value-Heads
        self.policy_head = NoisyLinear(84, n_actions)
        self.value_head = nn.Linear(84, 1)

    def forward(self, x):
        """
        x: Tensor der Form (Batch, 2*n_frames, 60, 60)
           Die ersten n_frames Kanäle entsprechen der Weitansicht,
           die nächsten n_frames Kanäle der zentrierten Ansicht.
        """
        B = x.size(0)
        n = x.size(1) // 2
        wide = x[:, :n, :, :]
        crop = x[:, n:, :, :]

        # Weitansicht verarbeiten
        x_w = F.relu(self.bn1_wide(self.conv1_wide(wide)))
        x_w = self.layer1_wide(x_w)
        x_w = self.layer2_wide(x_w)
        x_w = x_w.view(B, -1)
        x_w = F.relu(self.fc_image_wide(x_w))

        # Zentrierte Ansicht verarbeiten
        x_c = F.relu(self.bn1_crop(self.conv1_crop(crop)))
        x_c = self.layer1_crop(x_c)
        x_c = self.layer2_crop(x_c)
        x_c = x_c.view(B, -1)
        x_c = F.relu(self.fc_image_crop(x_c))

        # Fusion der beiden Ströme
        x_fused = torch.cat([x_w, x_c], dim=1)
        x_fused = F.relu(self.fc_fusion(x_fused))

        logits = self.policy_head(x_fused)
        value = self.value_head(x_fused)
        return logits, value

# -------------------------------
# PPO-Trainer mit dualem Input
# -------------------------------
class PPOTrainer:
    def __init__(self,
                 n_frames=4,  # Anzahl Frames pro Ansicht
                 n_actions=3,
                 lr=2e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_epsilon=0.2,
                 ppo_epochs=4,
                 minibatch_size=64,
                 total_timesteps=1_000_000,
                 update_timesteps=2048,
                 device='cuda'):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.update_timesteps = update_timesteps
        self.total_timesteps = total_timesteps

        # Adaptive Entropie: Startwert und Abklingfaktor
        self.entropy_coef = 0.01
        self.entropy_decay = 0.9999
        self.min_entropy_coef = 0.001

        # SuperHexagon-Interface: liefert ein Tuple (weit, zentriert)
        self.env = SuperHexagonInterface(frame_skip=4, run_afap=True, allow_game_restart=True)
        state = self.env.reset()  # state ist ein Tuple: (wide, crop)
        wide, crop = state  # beide mit Form (60,60)
        self.n_frames = n_frames
        # Erzeuge separate Frame-Stapel für beide Ansichten
        state_stack_wide = np.stack([wide] * n_frames, axis=0)   # (n_frames, 60,60)
        state_stack_crop = np.stack([crop] * n_frames, axis=0)     # (n_frames, 60,60)
        # Zusammenführen entlang der Kanal-Dimension -> (2*n_frames, 60,60)
        self.state_stack = np.concatenate([state_stack_wide, state_stack_crop], axis=0)

        self.input_shape = self.state_stack.shape  # (2*n_frames, 60,60)
        self.n_actions = n_actions

        # Nutze das duale Netzwerk
        self.net = DualViewResNetActorCritic(n_frames, n_actions).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        # LR-Scheduler: Reduziert die Lernrate schrittweise
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5000, gamma=0.99)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, 2*n_frames, 60,60)
        with torch.no_grad():
            logits, value = self.net(state_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action).to(self.device)).item()
        return action, log_prob, value.item()

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        # Hänge einen letzten Value für die Berechnung an
        values = values + [0]
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = [adv + v for adv, v in zip(advantages, values[:-1])]
        return advantages, returns

    def train(self):
        # Verwende self.state_stack als initialen Zustand
        state_stack = self.state_stack
        timestep = 0
        episode = 0
        episode_reward = 0
        episode_frames = 0

        # Speicher für Trajektorien
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

        while timestep < self.total_timesteps:
            # Sammle update_timesteps Schritte
            for _ in range(self.update_timesteps):

                walls = self.env.get_walls()
                num_slots = self.env.get_num_walls()
                min_distances = [float('inf')] * num_slots

                for wall in walls:
                    print(wall)
                    if wall['distance'] > 0 and wall['enabled']:
                        min_distances[wall['slot'] % num_slots] = min(min_distances[wall['slot'] % num_slots], wall['distance'])

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

                current_state = state_stack.copy()  # (2*n_frames, 60,60)

                action, log_prob, value = self.select_action(current_state)
                next_obs, _, done = self.env.step(action)
                # next_obs: Tuple (wide, crop)
                wide_next, crop_next = next_obs
                # Neues Frame: Zusammenführung beider Ansichten (Form: (2,60,60))
                next_frame = np.concatenate([np.expand_dims(wide_next, axis=0),
                                             np.expand_dims(crop_next, axis=0)], axis=0)

                reward = shape_reward(episode_frames, done)

                states.append(current_state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(float(done))
                values.append(value)

                episode_reward += reward
                episode_frames += 1
                # Aktualisiere den State-Stack: Entferne die ältesten 2 Kanäle und füge next_frame oben ein
                state_stack = np.concatenate([next_frame, state_stack[:-2]], axis=0)
                timestep += 1

                if done:
                    print(f"Episode {episode} beendet nach {episode_frames} Frames, Reward: {episode_reward:.2f}")
                    state = self.env.reset()  # tuple (wide, crop)
                    wide, crop = state
                    state_stack_wide = np.stack([wide] * self.n_frames, axis=0)
                    state_stack_crop = np.stack([crop] * self.n_frames, axis=0)
                    state_stack = np.concatenate([state_stack_wide, state_stack_crop], axis=0)
                    episode += 1
                    episode_reward = 0
                    episode_frames = 0

            # Berechne den letzten Value
            _, _, last_value = self.select_action(state_stack)
            values.append(last_value)

            # Berechne Advantages und Returns mit GAE
            advantages, returns = self.compute_gae(rewards, values, dones)

            # Konvertiere gesammelte Daten in Tensors
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
            returns_tensor = torch.FloatTensor(returns).to(self.device)
            advantages_tensor = torch.FloatTensor(advantages).to(self.device)

            # Normiere Advantages
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

            dataset_size = states_tensor.size(0)
            for _ in range(self.ppo_epochs):
                indices = np.arange(dataset_size)
                np.random.shuffle(indices)
                for start in range(0, dataset_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_idx = indices[start:end]

                    mb_states = states_tensor[mb_idx]
                    mb_actions = actions_tensor[mb_idx]
                    mb_old_log_probs = log_probs_tensor[mb_idx]
                    mb_returns = returns_tensor[mb_idx]
                    mb_advantages = advantages_tensor[mb_idx]

                    logits, values_pred = self.net(mb_states)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    new_log_probs = dist.log_prob(mb_actions)

                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(values_pred.squeeze(-1), mb_returns)
                    entropy_loss = -dist.entropy().mean()

                    loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            # Update des Lernraten-Schedulers und Anpassung des Entropie-Koeffizienten
            self.scheduler.step()
            self.entropy_coef = max(self.min_entropy_coef, self.entropy_coef * self.entropy_decay)
            self.net.policy_head.reset_noise()

            # Leere Trajektorien-Speicher für den nächsten Update-Zyklus
            states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

if __name__ == '__main__':
    trainer = PPOTrainer(
        n_frames=4,
        n_actions=3,
        lr=2e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        ppo_epochs=4,
        minibatch_size=256,
        total_timesteps=1_000_000,
        update_timesteps=2048,
        device='cuda'
    )
    trainer.train()

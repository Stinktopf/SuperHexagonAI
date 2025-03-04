# trainerC.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from superhexagon import SuperHexagonInterface  # Vorausgesetzt, diese Schnittstelle ist vorhanden

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
# ResNet-ähnliches Actor-Critic-Netzwerk (mit Input-Normalisierung)
# -------------------------------
class ResNetActorCritic(nn.Module):
    """
    Erwartet: in_channels = 2 * n_frames.
    Bsp. bei n_frames=4 => in_channels=8
    """
    def __init__(self, in_channels, n_actions):
        super(ResNetActorCritic, self).__init__()
        
        # Erste Convolution-Schicht: Input (in_channels, 120, 120) 
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Zwei Residual-Blöcke mit Stride=2 -> Reduktion 120x120 -> 30x30 (hier ggf. an neue Dimension anpassen)
        self.layer1 = ResidualBlock(16, 32, stride=2)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        
        # Fully-Connected-Schichten für Bildfeatures
        # Angepasst an die reduzierte Größe (hier: 30x30 -> ggf. 15x15, falls weitere Anpassung nötig)
        self.fc_image = nn.Linear(64 * 15 * 15, 120)
        self.fc_fusion = nn.Linear(120, 84)
        
        # Policy-Head als Noisy Linear Layer für bessere Exploration
        self.policy_head = NoisyLinear(84, n_actions)
        self.value_head = nn.Linear(84, 1)
    
    def forward(self, x):
        # Angenommen x ist in [0..255]
        x = x / 255.0
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc_image(x))
        x = F.relu(self.fc_fusion(x))
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

# -------------------------------
# PPO-Trainer (mit 2-Kanal-Frames + Streak-Strafe)
# -------------------------------
class PPOTrainer:
    def __init__(self,
                 n_frames=4,       # Wir stacken 4 Zeitschritte
                 n_actions=3,
                 lr=1e-3,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_epsilon=0.2,
                 ppo_epochs=4,
                 minibatch_size=32,
                 total_timesteps=1_000_000,
                 update_timesteps=1024,
                 device='cuda'):
        
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.update_timesteps = update_timesteps
        self.total_timesteps = total_timesteps
        
        # Entropie-Koeffizient (adaptiv)
        self.entropy_coef = 0.01
        self.entropy_decay = 0.9999
        self.min_entropy_coef = 0.001
        
        # Richtungs-Streak
        self.last_action = None
        self.current_streak = 0
        self.max_streak_penalty = 30  # ab dieser Streak-Höhe wird Straf-Bonus angewendet
        
        # Starte Environment
        self.env = SuperHexagonInterface(frame_skip=4, run_afap=True, allow_game_restart=True)
        
        # Environment-Reset => (frame, frame_cropped)
        state_tuple, _ = self.env.reset()  
        # Debug: Shape der vorverarbeiteten Frames prüfen
        # print("State tuple shapes:", state_tuple[0].shape, state_tuple[1].shape)
        
        # Mache aus zwei Bildern (2, Height, Width)
        combined = np.stack([state_tuple[0], state_tuple[1]], axis=0)  # erwartete shape: (2, 120, 120)
        
        # Lege den Frame-Stack an: (4, 2, 120, 120) => (8, 120, 120)
        self.n_frames = n_frames
        stack_4x2 = np.stack([combined] * n_frames, axis=0)  # shape: (4, 2, 120, 120)
        self.state_stack = stack_4x2.reshape(2 * n_frames, *combined.shape[1:])  # shape: (8, 120, 120)
        
        # Netzwerk: 2 * n_frames = 8 Kanäle
        in_channels = 2 * n_frames
        self.n_actions = n_actions
        self.net = ResNetActorCritic(in_channels, n_actions).to(device)
        
        # Optimizer & Scheduler
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_timesteps)
        
        # Logging
        self.episode_rewards_history = []
        self.episode_numbers = []
        plt.ion()
        self.fig, self.ax = plt.subplots()

    def update_plot(self, episode, reward):
        self.episode_numbers.append(episode)
        self.episode_rewards_history.append(reward)
        self.ax.clear()
        self.ax.plot(self.episode_numbers, self.episode_rewards_history, label='Reward pro Episode')
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Reward')
        self.ax.legend()
        plt.draw()
        plt.pause(0.001)

    def shape_reward(self, done, action):
        """
        +1 pro Step, -50 bei Tod.
        Strafbonus bei zu langem "Streak" in gleicher Richtung.
        """
        reward = -50.0 if done else 1.0

        if action in [1, 2]:
            if action == self.last_action:
                self.current_streak += 1
            else:
                self.current_streak = 1
        else:
            self.current_streak = 0

        self.last_action = action

        if self.current_streak > self.max_streak_penalty:
            reward -= 5.0

        return reward

    def select_action(self, state_stack):
        """
        state_stack: shape (8, H, W) -> unsqueeze(0) -> (1, 8, H, W)
        """
        state_tensor = torch.FloatTensor(state_stack).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(state_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action).to(self.device)).item()
        return action, log_prob, value.item()

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0.0
        values = values + [0.0]  # letztes Value
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step+1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        returns = [adv + v for adv, v in zip(advantages, values[:-1])]
        return advantages, returns

    def train(self):
        timestep = 0
        episode = 0
        episode_reward = 0.0
        episode_frames = 0
        
        # Speicher für Trajektorien
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        
        while timestep < self.total_timesteps:
            for _ in range(self.update_timesteps):
                current_state = self.state_stack.copy()  # erwartet: (8, 120, 120)
                
                action, log_prob, value = self.select_action(current_state)
                next_obs, _, done = self.env.step(action)  # (frame, frame_cropped)
                next_frame, next_frame_cropped = next_obs  # (120,120), (120,120)
                combined_next = np.stack([next_frame, next_frame_cropped], axis=0)  # (2, 120, 120)

                reward = self.shape_reward(done, action)
                
                states.append(current_state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(float(done))
                values.append(value)
                
                episode_reward += reward
                episode_frames += 1
                timestep += 1
                
                # Update des Frame-Stacks
                old_stack_4x2 = self.state_stack.reshape(self.n_frames, 2, *combined_next.shape[1:])
                new_stack_4x2 = np.concatenate([combined_next[np.newaxis, :], old_stack_4x2[:-1]], axis=0)
                self.state_stack = new_stack_4x2.reshape(2 * self.n_frames, *combined_next.shape[1:])
                
                if done:
                    print(f"Episode {episode} beendet nach {episode_frames} Frames, Reward: {episode_reward:.2f}")
                    self.update_plot(episode, episode_reward)

                    state_tuple, _ = self.env.reset()  
                    combined = np.stack([state_tuple[0], state_tuple[1]], axis=0)
                    stack_init_4x2 = np.stack([combined] * self.n_frames, axis=0)
                    self.state_stack = stack_init_4x2.reshape(2 * self.n_frames, *combined.shape[1:])
                    
                    episode += 1
                    episode_reward = 0.0
                    episode_frames = 0
                    self.current_streak = 0
                    self.last_action = None
            
            # GAE: Letztes Value
            _, _, last_value = self.select_action(self.state_stack)
            values.append(last_value)
            advantages, returns = self.compute_gae(rewards, values, dones)
            
            # Konvertiere in Tensors
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            log_probs_tensor = torch.FloatTensor(log_probs).to(self.device)
            returns_tensor = torch.FloatTensor(returns).to(self.device)
            advantages_tensor = torch.FloatTensor(advantages).to(self.device)
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
            
            self.scheduler.step()
            self.entropy_coef = max(self.min_entropy_coef, self.entropy_coef * self.entropy_decay)
            self.net.policy_head.reset_noise()
            
            states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

if __name__ == '__main__':
    trainer = PPOTrainer(
        n_frames=4,
        n_actions=3,
        lr=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        ppo_epochs=4,
        minibatch_size=64,
        total_timesteps=1_000_000,
        update_timesteps=1024,
        device='cuda'
    )
    trainer.train()

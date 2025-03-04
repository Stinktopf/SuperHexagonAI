import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import matplotlib
matplotlib.use('TkAgg')  # Wechsel zum TkAgg-Backend
import matplotlib.pyplot as plt
import torchvision.models as models
from superhexagon import SuperHexagonInterface  # Vorausgesetzt, diese Schnittstelle ist vorhanden

# -------------------------------
# Reward Shaping: konstante Belohnung pro Frame
# -------------------------------
def shape_reward(episode_frames, done):
    if done:
        return -50  # Feste Strafbelohnung beim Scheitern
    else:
        return 1    # Konstanter Überlebensbonus

# -------------------------------
# MobileNet-basierte Actor-Critic-Architektur
# -------------------------------
class MobileNetActorCritic(nn.Module):
    def __init__(self, n_frames, n_actions):
        super(MobileNetActorCritic, self).__init__()
        self.n_actions = n_actions
        
        # Lade MobileNetV2 (ohne vortrainierte Gewichte)
        mobilenet = models.mobilenet_v2(pretrained=False)
        # Passe den ersten Convolution-Layer an: statt 3 Kanälen jetzt n_frames (z.B. 4 gestapelte Graustufenbilder)
        mobilenet.features[0][0] = nn.Conv2d(n_frames, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        # MobileNetV2 liefert einen Feature-Vektor der Größe 1280
        self.fc_policy = nn.Linear(1280, n_actions)
        self.fc_value = nn.Linear(1280, 1)
    
    def forward(self, x):
        # x: [Batch, n_frames, 60, 60]. Annahme: Pixel im Bereich 0-255.
        x = x / 255.0
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc_policy(x)
        value = self.fc_value(x)
        return logits, value

# -------------------------------
# PPO-Trainer mit MobileNet
# -------------------------------
class PPOMobileNetTrainer:
    def __init__(self,
                 n_frames=4,
                 n_actions=3,
                 lr=1e-4,
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
        
        # Entropie-Koeffizient (zur Förderung von Exploration)
        self.entropy_coef = 0.01
        
        # Initialisiere die Umgebung
        self.env = SuperHexagonInterface(frame_skip=4, run_afap=True, allow_game_restart=True)
        state, _ = self.env.reset()
        self.input_shape = (n_frames,) + state.shape
        self.n_actions = n_actions
        
        # Erstelle das MobileNet-basierte Actor-Critic-Netzwerk
        self.net = MobileNetActorCritic(n_frames, n_actions).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        
        # Live-Plotting zur Überwachung des Trainingsfortschritts
        self.episode_rewards_history = []
        self.episode_numbers = []
        plt.ion()
        self.fig, self.ax = plt.subplots()
    
    def update_plot(self, episode, reward):
        # Aktualisiere den Plot nur alle 10 Episoden
        if episode % 10 == 0:
            self.episode_numbers.append(episode)
            self.episode_rewards_history.append(reward)
            self.ax.clear()
            self.ax.plot(self.episode_numbers, self.episode_rewards_history, label='Reward pro Episode')
            self.ax.set_xlabel('Episode')
            self.ax.set_ylabel('Reward')
            self.ax.legend()
            plt.draw()
            plt.pause(0.001)
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits, value = self.net(state_tensor)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)  # Tensor, um den Gradientenfluss zu erhalten
        return action.item(), log_prob, value, probs
    
    def compute_gae(self, rewards, values, dones, last_value):
        advantages = []
        gae = 0
        # Hänge last_value an, um die Bootstrapping-Berechnung zu ermöglichen
        values = values + [last_value]
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i+1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        returns = [adv + v for adv, v in zip(advantages, values[:-1])]
        return advantages, returns
    
    def train(self):
        # Initialisiere den Frame-Stack
        state, _ = self.env.reset()
        state_stack = np.stack([state] * 4, axis=0)
        
        timestep = 0
        episode = 0
        episode_reward = 0
        episode_frames = 0
        
        # Speicher für Trajektorien
        states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []
        
        while timestep < self.total_timesteps:
            # Sammle eine Trajektorie über update_timesteps Schritte
            for _ in range(self.update_timesteps):
                current_state = state_stack.copy()
                action, log_prob, value, _ = self.select_action(current_state)
                next_obs, _, done = self.env.step(action)
                next_state, _ = next_obs
                
                reward = shape_reward(episode_frames, done)
                
                states.append(current_state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(float(done))
                values.append(value.item())
                
                episode_reward += reward
                episode_frames += 1
                timestep += 1
                
                state_stack = np.concatenate(([next_state], state_stack[:-1]), axis=0)
                
                if done:
                    print(f"Episode {episode} beendet nach {episode_frames} Frames, Reward: {episode_reward:.2f}")
                    self.update_plot(episode, episode_reward)
                    state, _ = self.env.reset()
                    state_stack = np.stack([state] * 4, axis=0)
                    episode += 1
                    episode_reward = 0
                    episode_frames = 0
            
            # Berechne den letzten Wert zum Bootstrapping
            _, _, last_value, _ = self.select_action(state_stack)
            last_value = last_value.item()
            
            # Berechne Advantages und Returns mit Generalized Advantage Estimation (GAE)
            advantages, returns = self.compute_gae(rewards, values, dones, last_value)
            
            # Konvertiere gesammelte Daten in Tensors
            states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            actions_tensor = torch.LongTensor(actions).to(self.device)
            old_log_probs_tensor = torch.stack(log_probs).to(self.device)
            returns_tensor = torch.FloatTensor(returns).to(self.device)
            advantages_tensor = torch.FloatTensor(advantages).to(self.device)
            
            # Normiere Advantages
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
            
            dataset_size = states_tensor.size(0)
            indices = np.arange(dataset_size)
            
            # Mehrere PPO-Epochen
            for _ in range(self.ppo_epochs):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_idx = indices[start:end]
                    
                    mb_states = states_tensor[mb_idx]
                    mb_actions = actions_tensor[mb_idx]
                    mb_old_log_probs = old_log_probs_tensor[mb_idx]
                    mb_returns = returns_tensor[mb_idx]
                    mb_advantages = advantages_tensor[mb_idx]
                    
                    logits, values_pred = self.net(mb_states)
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    new_log_probs = dist.log_prob(mb_actions)
                    
                    ratio = torch.exp(new_log_probs - mb_old_log_probs)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    value_loss = F.mse_loss(values_pred.squeeze(-1), mb_returns)
                    entropy_loss = -dist.entropy().mean()
                    
                    loss = policy_loss + 0.5 * value_loss + self.entropy_coef * entropy_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            
            # Leere die Trajektorien-Speicher für den nächsten Update-Zyklus
            states, actions, log_probs, rewards, dones, values = [], [], [], [], [], []

if __name__ == '__main__':
    trainer = PPOMobileNetTrainer(
        n_frames=1,         # z.B. 4 gestapelte Graustufenbilder
        n_actions=3,        # Aktionen: Halten, Rechts, Links
        lr=1e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        ppo_epochs=4,
        minibatch_size=64,
        total_timesteps=1_000_000,
        update_timesteps=2048,
        device='cuda'
    )
    trainer.train()

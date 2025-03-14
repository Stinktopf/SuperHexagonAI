import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from superhexagon import SuperHexagonInterface

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class SuperHexagonGymEnv(gym.Env):
    """
    Gym env for SuperHexagon.
    Actions: 0=Stay, 1=Left, 2=Right.
    Observation: [norm wall distances, player angle, direction].
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        """Initialize env and define spaces."""
        super().__init__()
        self.env = SuperHexagonInterface(
            frame_skip=1, run_afap=True, allow_game_restart=True
        )
        self.action_space = spaces.Discrete(3)
        self.n_slots = self.env.get_num_slots()
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2 + self.n_slots,), dtype=np.float32
        )
        self.episode_frames = 0

    def reset(self, seed=None, options=None):
        """Reset env and return initial observation."""
        super().reset(seed=seed)
        self.env.reset()
        self.episode_frames = 0
        obs = self._get_state()
        return obs, {}

    def step(self, action):
        """Take action and return (obs, reward, done, truncated, info)."""
        _, _, done = self.env.step(action)
        self.episode_frames += 1
        obs = self._get_state()
        reward = self._get_reward(done, action)
        return obs, reward, done, False, {}

    def _get_wall_distances(self):
        """
        Compute minimal distance to wall for each slot; if there is no wall, set it to 5000.0.
        """
        min_distances = np.full(self.n_slots, np.inf, dtype=np.float32)
        for wall in self.env.get_walls():
            if wall.distance > 0 and wall.enabled:
                slot_idx = wall.slot % self.n_slots
                min_distances[slot_idx] = min(min_distances[slot_idx], wall.distance)
        min_distances[min_distances == np.inf] = 5000.0
        return min_distances

    def _get_state(self):
        """
        Build state vector: [normalized wall distances, player angle, direction].
        """
        wall_info = self._get_wall_distances()
        wall_info = wall_info / wall_info.sum()
        player_angle = np.array(
            [0.5 + 0.5 * math.sin(math.radians(self.env.get_triangle_angle()))]
        )
        direction = np.array([self.get_exit_direction()], dtype=np.float32)
        state = np.concatenate([wall_info, player_angle, direction])
        return state

    def _get_reward(self, done, action):
        """
        Return reward; -10 if done, else adjust by angular distance.
        """
        debug = self.env.debug_mode
        if done:
            if debug:
                print("Game over! Return -10.0")
            return -10.0
        wall_info = self._get_wall_distances()
        wall_info = wall_info / wall_info.sum()
        player_slot_idx = self.env.get_triangle_slot()
        player_slot_onehot = np.zeros(self.n_slots, dtype=np.float32)
        player_slot_onehot[player_slot_idx] = 1.0
        reward = 1
        distance = self.get_exit_distance()
        distance_factor = distance / 360
        if distance_factor == 0:
            reward *= 1 if action == 0 else 0.7
        else:
            reward *= -distance_factor
        if debug:
            print("-------------------------------------------------------")
            print(f"Player Angle: {self.env.get_triangle_angle()}")
            print(f"Player Slot: {self.env.get_triangle_slot()}")
            print(f"Player One-Hot: {player_slot_onehot}")
            print(f"Distance: {distance} and Factor {distance_factor}")
            print(f"Action Taken: {action}")
            print(f"State: {self._get_state()}")
            print(f"Calculated Reward: {reward}")
            print("-------------------------------------------------------\n")
        return float(reward)

    def get_exit_distance(self):
        """
        Return minimal angular distance to a wall gap.
        """
        num_slots = self.env.get_num_slots()
        wall_info = self._get_wall_distances()
        wall_info = wall_info / wall_info.sum()
        player_angle = self.env.get_triangle_angle()
        player_slot = self.env.get_triangle_slot()
        left_border = 0
        right_border = 0
        if wall_info[player_slot] > wall_info.min():
            return 0
        for i in range(player_slot, player_slot + num_slots):
            index = i % num_slots
            if wall_info[index] > wall_info.min():
                left_border = index * (360 / num_slots)
                break
        for i in range(player_slot, player_slot - num_slots, -1):
            index = i % num_slots
            if wall_info[index] > wall_info.min():
                right_border = (index + 1) * (360 / num_slots)
                break
        left_distance = min(
            abs(player_angle - left_border), 360 - abs(player_angle - left_border)
        )
        right_distance = min(
            abs(player_angle - right_border), 360 - abs(player_angle - right_border)
        )
        return min(left_distance, right_distance)

    def get_exit_direction(self):
        """
        Return direction indicator (0.0 for left, 1.0 for right) to the next wall gap.
        """
        num_slots = self.env.get_num_slots()
        wall_info = self._get_wall_distances()
        wall_info = wall_info / wall_info.sum()
        player_angle = self.env.get_triangle_angle()
        player_slot = self.env.get_triangle_slot()
        left_border = 0
        right_border = 0
        if wall_info[player_slot] > wall_info.min():
            return 0.5
        for i in range(player_slot, player_slot + num_slots):
            index = i % num_slots
            if wall_info[index] > wall_info.min():
                left_border = index * (360 / num_slots)
                break
        for i in range(player_slot, player_slot - num_slots, -1):
            index = i % num_slots
            if wall_info[index] > wall_info.min():
                right_border = (index + 1) * (360 / num_slots)
                break
        left_distance = min(
            abs(player_angle - left_border), 360 - abs(player_angle - left_border)
        )
        right_distance = min(
            abs(player_angle - right_border), 360 - abs(player_angle - right_border)
        )
        return 0.0 if left_distance < right_distance else 1.0

    def render(self, mode="human"):
        """Render method (not implemented)."""
        pass


class ActorCritic(nn.Module):
    def __init__(self, input_dim, num_actions, net_arch):
        """Initialize ActorCritic network."""
        super(ActorCritic, self).__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in net_arch:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        self.shared = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, num_actions)
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, x):
        """Forward pass."""
        x = self.shared(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value


class PPOAgent:
    def __init__(
        self,
        env,
        net_arch=[512, 512, 256],
        learning_rate=2e-4,
        gamma=0.999,
        gae_lambda=0.95,
        clip_ratio=0.2,
        ent_coef=0.01,
        vf_coef=0.6,
        max_grad_norm=0.5,
        n_steps=4096,
        batch_size=32,
        total_timesteps=1_000_000,
        initial_ent_coef=0.1,
        final_ent_coef=0.005,
        update_epochs=4,
    ):
        """Initialize PPOAgent."""
        self.env = env
        self.obs_shape = env.observation_space.shape
        self.num_actions = env.action_space.n

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        self.n_steps = n_steps
        self.batch_size = batch_size
        self.update_epochs = update_epochs

        self.total_timesteps = total_timesteps
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef

        self.device = torch.device("cpu")
        self.model = ActorCritic(
            input_dim=self.obs_shape[0],
            num_actions=self.num_actions,
            net_arch=net_arch,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def compute_gae(self, rewards, values, dones, last_value):
        """Compute GAE."""
        advantages = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[-1]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            lastgaelam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * lastgaelam
            )
            advantages[t] = lastgaelam
        returns = advantages + values
        return advantages, returns

    def update_ent_coef(self, current_step):
        """Update entropy coefficient."""
        progress = 1.0 - (current_step / self.total_timesteps)
        self.ent_coef = self.final_ent_coef + progress * (
            self.initial_ent_coef - self.final_ent_coef
        )

    def learn(self):
        """Train PPOAgent."""
        obs, _ = self.env.reset()
        ep_rewards = 0
        episode_lengths = []
        current_episode_length = 0
        current_step = 0

        while current_step < self.total_timesteps:
            observations = []
            actions = []
            logprobs = []
            rewards = []
            dones = []
            values = []

            for _ in range(self.n_steps):
                current_episode_length += 1
                obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
                logits, value = self.model(obs_tensor)
                probs = torch.softmax(logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()
                logprob = dist.log_prob(action)

                observations.append(obs)
                actions.append(action.item())
                logprobs.append(logprob.item())
                values.append(value.item())

                obs, reward, done, _, _ = self.env.step(action.item())
                rewards.append(reward)
                dones.append(done)

                ep_rewards += reward
                current_step += 1

                if done:
                    episode_lengths.append(current_episode_length)
                    current_episode_length = 0
                    obs, _ = self.env.reset()

                self.update_ent_coef(current_step)

            obs_tensor = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            _, last_value = self.model(obs_tensor)
            last_value = last_value.item()

            observations = np.array(observations, dtype=np.float32)
            actions = np.array(actions)
            logprobs = np.array(logprobs, dtype=np.float32)
            rewards = np.array(rewards, dtype=np.float32)
            dones = np.array(dones, dtype=np.float32)
            values = np.array(values, dtype=np.float32)

            advantages, returns = self.compute_gae(rewards, values, dones, last_value)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            observations = torch.FloatTensor(observations).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            old_logprobs = torch.FloatTensor(logprobs).to(self.device)
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)

            dataset_size = observations.shape[0]
            indices = np.arange(dataset_size)
            for epoch in range(self.update_epochs):
                np.random.shuffle(indices)
                for start in range(0, dataset_size, self.batch_size):
                    end = start + self.batch_size
                    mb_idx = indices[start:end]

                    mb_obs = observations[mb_idx]
                    mb_actions = actions[mb_idx]
                    mb_old_logprobs = old_logprobs[mb_idx]
                    mb_returns = returns[mb_idx]
                    mb_advantages = advantages[mb_idx]

                    logits, values_pred = self.model(mb_obs)
                    probs = torch.softmax(logits, dim=-1)
                    dist = Categorical(probs)
                    new_logprobs = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_logprobs - mb_old_logprobs)
                    surr1 = ratio * mb_advantages
                    surr2 = (
                        torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                        * mb_advantages
                    )
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = ((mb_returns - values_pred.squeeze()) ** 2).mean()

                    loss = (
                        policy_loss
                        + self.vf_coef * value_loss
                        - self.ent_coef * entropy
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

            if current_step % 10000 < self.n_steps:
                avg_episode_length = np.mean(episode_lengths) if episode_lengths else 0
                print(
                    f"Step: {current_step}, Avg. Episode Length: {avg_episode_length:.2f}, Episode Reward: {ep_rewards:.2f}"
                )
                ep_rewards = 0
                episode_lengths = []


if __name__ == "__main__":
    env = SuperHexagonGymEnv()
    agent = PPOAgent(
        env=env,
        net_arch=[512, 512, 256],
        learning_rate=2e-4,
        gamma=0.999,
        gae_lambda=0.95,
        clip_ratio=0.2,
        ent_coef=0.01,
        vf_coef=0.6,
        max_grad_norm=0.5,
        n_steps=4096,
        batch_size=32,
        total_timesteps=1_000_000,
        initial_ent_coef=0.1,
        final_ent_coef=0.005,
        update_epochs=4,
    )
    agent.learn()

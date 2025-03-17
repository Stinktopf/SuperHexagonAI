from environment import SuperHexagonGymEnv
from stable_baselines3 import DQN

if __name__ == "__main__":
    env = SuperHexagonGymEnv()
    total_timesteps = 1_000_000

    model = DQN(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        device="cpu",
        learning_rate=1e-4,
        gamma=0.99,
        buffer_size=100000,
        learning_starts=5000,
        batch_size=64,
        train_freq=8,
        target_update_interval=250,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
    )

    model.learn(total_timesteps=total_timesteps)

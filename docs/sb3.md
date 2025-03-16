The following section presents the implementation using Stable-Baselines3.

### Setup

[Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/) provides a direct implementation of the PPO algorithm, which we use for training our RL agent. By leveraging this framework, we can seamlessly integrate our Gymnasium environment and train the agent.

The usage is as follows:
```python
env = SuperHexagonGymEnv()
total_timesteps = 1_000_000
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    device="cpu",
    n_steps=4096,
    batch_size=32,
    learning_rate=2e-4,
    gamma=0.999,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.6,
    max_grad_norm=0.5,
    policy_kwargs=dict(net_arch=[512, 512, 256]),
)
ent_coef_scheduler = EntropyCoefficientScheduler(
    total_timesteps=total_timesteps, initial_ent_coef=0.1, final_ent_coef=0.005
)
model.learn(total_timesteps=total_timesteps, callback=ent_coef_scheduler)
```

### Settings

The PPO agent is configured with:

- **Policy:** A multi-layer perceptron is used.
- **Learning Rate:** `2e-4` - Stabilizes learning.
- **Gamma:** `0.999` - Prioritizes long-term survival over short-term gains.
- **Entropy Coefficient:** `0.01 → 0.005` - Encourages early exploration, then stabilizes decision-making.
- **Policy Network Architecture:** `[512, 512, 256]` - Optimised by trial and error.

## Entropy Scheduling

To ensure a balance between exploration and exploitation, we linearly scale the entropy coefficient during training. 

This is achieved using the `EntropyCoefficientScheduler` callback:

- ::: trainer_PPO_GYM_SB3.EntropyCoefficientScheduler.__init__
- ::: trainer_PPO_GYM_SB3.EntropyCoefficientScheduler._on_step
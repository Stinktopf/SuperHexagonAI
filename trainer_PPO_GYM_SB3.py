from environment import SuperHexagonGymEnv

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class EntropyCoefficientScheduler(BaseCallback):
    """
    Callback to linearly schedule the entropy coefficient during training.
    """

    def __init__(
        self,
        total_timesteps: int,
        initial_ent_coef: float,
        final_ent_coef: float,
        verbose: int = 0,
    ):
        """
        **\_\_init\_\_**
        """
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef

    def _on_step(self) -> bool:
        """
        **_on_step**
        """
        progress = 1.0 - (self.num_timesteps / self.total_timesteps)
        new_ent_coef = self.final_ent_coef + progress * (
            self.initial_ent_coef - self.final_ent_coef
        )
        self.model.ent_coef = new_ent_coef
        return True


if __name__ == "__main__":
    """
    Trains a PPO agent on the SuperHexagonGymEnv using an entropy coefficient scheduler.
    """
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

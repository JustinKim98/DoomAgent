import vizdoom as vzd
import os
import numpy as np
import matplotlib.pyplot as plt
from models import model
from doomenv import env
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common import policies
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.env_util import make_vec_env


class SaveModelCallBack(BaseCallback):
    def __init__(self, freq, path, log_dir, verbose=1):
        super(SaveModelCallBack, self).__init__(verbose)
        self.frequency = freq
        self.log_frequency = freq / 10
        self.model_path = path
        self.log_dir = log_dir
        self.writer = torch.utils.tensorboard.SummaryWriter()

    def _on_step(self):
        if self.n_calls % self.frequency == 0:
            self.model.save(
                os.path.join(self.model_path, "model_iter_{}".format(self.n_calls))
            )

        return True


if __name__ == "__main__":
    steps = 10000

    vec_env = make_vec_env(
        lambda: env.ContinuousEnv(
            "scenarios/deathmatch.cfg",
            frame_buffer_size=6,
            living_reward=0.01,
            kill_opponent_reward=200,
            shoot_opponent_reward=70,
        ),
        1,
    )

    policy_kwargs = dict(
        features_extractor_class=model.PolicyModel,
        features_extractor_kwargs=dict(features_dim=1024),
        net_arch=dict(
            activation_fn=torch.nn.LeakyReLU,
            net_arch=dict(
                pi=[1024, 512, 512, 256, 256, 128, 64],
                vf=[1024, 512, 512, 256, 256, 128, 64],
            ),
        ),
    )

    torch.set_default_dtype(torch.float32)

    deathmatch_model = PPO(
        policy=policies.ActorCriticCnnPolicy,
        policy_kwargs=policy_kwargs,
        env=vec_env,
        verbose=True,
        learning_rate=1e-5 * 5,
        batch_size=64,
        gamma=0.99,
        n_steps=steps,
        tensorboard_log="logs/deathmatch_continuous",
        device="cuda",
    )

    callback = SaveModelCallBack(
        freq=10000,
        log_dir="logs/deathmatch_continuous",
        path="deathmatch_continuous_models",
    )
    deathmatch_model.learn(total_timesteps=steps * 10000, callback=callback)

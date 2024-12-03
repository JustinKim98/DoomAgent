import vizdoom as vzd
import os
import numpy as np
import matplotlib.pyplot as plt
from models import model
from doomenv import multiplayer_env
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
                os.path.join(
                    self.model_path, "model_iter_{}".format(self.n_calls / 100)
                )
            )

        return True


if __name__ == "__main__":
    steps = 5000
    allowed_actions = [
        vzd.Button.ATTACK,
        vzd.Button.MOVE_RIGHT,
        vzd.Button.MOVE_LEFT,
        vzd.Button.MOVE_FORWARD,
        vzd.Button.MOVE_BACKWARD,
        vzd.Button.TURN_LEFT,
        vzd.Button.TURN_RIGHT,
        vzd.Button.RELOAD,
    ]

    game = vzd.DoomGame()

    env = multiplayer_env.BaseEnv(
        "scenarios/multi.cfg",
        allowed_actions=allowed_actions,
        frame_buffer_size=6,
        living_reward=-0.01,
        kill_opponent_reward=100,
        shoot_opponent_reward=70,
        exploration_rate=0.05,
        game=game,
        configure_as_host=True,
    )

    policy_kwargs = dict(
        features_extractor_class=model.PolicyModel,
        features_extractor_kwargs=dict(features_dim=1024),
        net_arch=dict(
            activation_fn=torch.nn.Tanh,
            net_arch=dict(
                vf=[512, 512, 256, 256, 128, 64],
                pi=[512, 512, 256, 256, 128, 64],
            ),
        ),
    )

    model = PPO(
        policy=policies.ActorCriticCnnPolicy,
        policy_kwargs=policy_kwargs,
        env=env,
        verbose=True,
        learning_rate=1e-6 * 5,
        batch_size=64,
        gamma=0.99,
        n_steps=steps,
        tensorboard_log="logs/multiplayer",
        device="mps",
    )

    callback = SaveModelCallBack(
        freq=10000, log_dir="logs/multiplayer", path="multiplayer"
    )

    model.learn(total_timesteps=steps * 10000, callback=callback)

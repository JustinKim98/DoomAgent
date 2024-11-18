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
        # if self.n_calls % self.log_frequency == 0:
        #     x, y = ts2xy(load_results(self.log_dir), "timesteps")
        #     average = np.mean(y[-100:0])
        #     self.writer.add_scalar()

        return True


if __name__ == "__main__":
    steps = 2500
    allowed_actions = [
        vzd.Button.ATTACK,
        vzd.Button.MOVE_RIGHT,
        vzd.Button.MOVE_LEFT,
        vzd.Button.MOVE_UP,
        vzd.Button.MOVE_DOWN,
        vzd.Button.TURN_LEFT,
        vzd.Button.TURN_RIGHT,
        # vzd.Button.ALTATTACK,
        vzd.Button.RELOAD,
    ]

    vec_env = make_vec_env(
        lambda: env.BaseEnv(
            "scenarios/deathmatch.cfg",
            allowed_actions=allowed_actions,
            frame_buffer_size=4,
            living_reward=-0,
            kill_opponent_reward=200,
            shoot_opponent_reward=70,
        ),
        2,
    )

    policy_kwargs = dict(
        features_extractor_class=model.PolicyModel,
        features_extractor_kwargs=dict(features_dim=1024),
        net_arch=dict(
            activation_fn=torch.nn.ReLU,
            net_arch=dict(pi=[512, 512, 256, 128, 64], vf=[512, 512, 256, 128, 64]),
        ),
    )

    model = PPO(
        policy=policies.ActorCriticCnnPolicy,
        policy_kwargs=policy_kwargs,
        env=vec_env,
        verbose=True,
        learning_rate=1e-5 * 5,
        batch_size=64,
        gamma=0.99,
        n_steps=steps,
        tensorboard_log="logs/deathmatch4",
        device="cuda",
    )

    callback = SaveModelCallBack(
        freq=2500, log_dir="logs/deathmatch4", path="deathmatch_models3"
    )
    model.learn(total_timesteps=steps * 1000, callback=callback)

    # print("inference!")
    # env.reset()
    # is_done = False
    # action = 0

    # for i in range (0, 10):
    #     print(f"episode : {i}")
    #     while(not is_done):
    #         state, reward, is_done, _ = env.step(action)
    #         action = model.predict(state)
    #     env.reset()
    #     is_done = False

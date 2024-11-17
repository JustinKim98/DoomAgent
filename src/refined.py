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
        self.frequency =freq 
        self.log_frequency = freq/10
        self.model_path = path
        self.log_dir = log_dir
        self.writer = torch.utils.tensorboard.SummaryWriter()

    def _on_step(self):
        if self.n_calls % self.frequency == 0:
            self.model.save(os.path.join(self.model_path, 'model_iter_{}'.format(self.n_calls)))
        # if self.n_calls % self.log_frequency == 0:
        #     x, y = ts2xy(load_results(self.log_dir), "timesteps")
        #     average = np.mean(y[-100:0])
        #     self.writer.add_scalar()


        return True

if __name__ == '__main__':
    steps = 2500
    vec_env = make_vec_env(lambda: env.BaseEnv("scenarios/deathmatch.cfg"), 2)

    policy_kwargs = dict(
        features_extractor_class=model.PolicyModel,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[512, 256, 128, 32], vf=[512, 256, 128, 32]))
    )

    model = PPO(policy=policies.ActorCriticCnnPolicy, policy_kwargs=policy_kwargs, env=vec_env,
                verbose=True, learning_rate=1e-4, batch_size = 64, gamma = 0.99, n_steps=steps, tensorboard_log = "logs/progress_log", device="mps")
    # model = PPO.load("model_outputs_nov17-refined-1/model_iter_10000.zip", env = vec_env,  learning_ratedevice="mps")

    callback = SaveModelCallBack(freq=2500, log_dir="logs/progress_log", path="model_outputs_nov17-refined-2")
    model.learn(total_timesteps=steps*100, callback=callback)

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

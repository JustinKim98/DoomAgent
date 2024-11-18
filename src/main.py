import vizdoom as vzd
import random
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common import policies
from stable_baselines3.common.env_util import make_vec_env

from gym import Env
from gym.spaces import Discrete, Box
import cv2


class PolicyModel(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim):
        super(PolicyModel, self).__init__(observation_space, features_dim)
        input_channels = observation_space.shape[0]

        base_channel_size = 16

        self.conv2d1 = torch.nn.Conv2d(input_channels, base_channel_size, kernel_size=4)
        self.bn1 = torch.nn.BatchNorm2d(base_channel_size)
        self.maxpool1 = torch.nn.MaxPool2d(4, 2)
        self.relu1 = torch.nn.LeakyReLU()

        self.conv2d2 = torch.nn.Conv2d(
            base_channel_size, base_channel_size * 2, kernel_size=4
        )
        self.bn2 = torch.nn.BatchNorm2d(base_channel_size * 2)
        self.maxpool2 = torch.nn.MaxPool2d(4, 2)
        self.relu2 = torch.nn.LeakyReLU()

        self.conv2d3 = torch.nn.Conv2d(
            base_channel_size * 2, base_channel_size * 4, kernel_size=4
        )
        self.bn3 = torch.nn.BatchNorm2d(base_channel_size * 4)
        self.maxpool3 = torch.nn.MaxPool2d(4, 2)
        self.relu3 = torch.nn.LeakyReLU()

        self.conv2d4 = torch.nn.Conv2d(
            base_channel_size * 4, base_channel_size * 8, kernel_size=4
        )
        self.bn4 = torch.nn.BatchNorm2d(base_channel_size * 8)
        self.maxpool4 = torch.nn.MaxPool2d(4, 2)
        self.relu4 = torch.nn.LeakyReLU()

        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(11520, 1024)
        self.relu4 = torch.nn.LeakyReLU()
        self.linear2 = torch.nn.Linear(1024, features_dim)
        self.relu5 = torch.nn.LeakyReLU()

    def forward(self, state_input: torch.Tensor):
        h1 = self.conv2d1(state_input)
        h1 = self.bn1(h1)
        h1 = self.maxpool1(h1)
        h1 = self.relu1(h1)

        h2 = self.conv2d2(h1)
        h2 = self.bn2(h2)
        h2 = self.maxpool2(h2)
        h2 = self.relu2(h2)

        h3 = self.conv2d3(h2)
        h3 = self.bn3(h3)
        h3 = self.maxpool3(h3)
        h3 = self.relu3(h3)

        h4 = self.conv2d4(h3)
        h4 = self.bn4(h4)
        h4 = self.maxpool4(h4)
        h4 = self.relu4(h4)

        h5 = self.flatten(h4)
        h5 = self.linear1(h5)
        h5 = self.relu4(h5)

        h6 = self.linear2(h5)
        out = self.relu5(h6)
        return out


class DoomEnv(Env):
    def __init__(self, scenario):
        super().__init__()

        # Initialize game environment
        self.game = vzd.DoomGame()
        self.game.load_config(scenario)
        self.game.set_sound_enabled(True)
        self.game.set_console_enabled(True)
        self.game.set_render_all_frames(True)

        self.game.set_living_reward(0)
        self.game.set_death_penalty(100.0)
        self.game.set_available_buttons(
            [
                vzd.Button.ATTACK,
                vzd.Button.MOVE_RIGHT,
                vzd.Button.MOVE_LEFT,
                vzd.Button.MOVE_UP,
                vzd.Button.MOVE_DOWN,
                # vzd.Button.LOOK_UP,
                # vzd.Button.LOOK_DOWN,
                # vzd.Button.CROUCH,
                vzd.Button.TURN_LEFT,
                vzd.Button.TURN_RIGHT,
                vzd.Button.JUMP,
                # vzd.Button.USE,
                vzd.Button.ALTATTACK,
                vzd.Button.RELOAD,
            ]
        )

        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X180)
        self.game.set_screen_format(vzd.ScreenFormat.CRCGCB)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_automap_buffer_enabled(True)
        self.game.set_objects_info_enabled(True)
        self.game.set_sectors_info_enabled(True)
        self.game.set_labels_buffer_enabled(True)

        self.game.set_render_hud(True)
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_particles(False)
        self.game.set_render_decals(True)

        self.game.init()

        self.observation_space = Box(
            low=0, high=255, shape=(4, 180, 320), dtype=np.uint8
        )
        self.action_space = Discrete(len(self.game.get_available_buttons()))
        self.actions = np.zeros(len(self.game.get_available_buttons()), dtype=np.uint8)

        self.maximum_steps = 50000
        self.num_hits = 0
        self.num_taken_hits = 0
        self.total_reward = 0
        self.prev_damage = 0
        self.num_kills = 0

    def step(self, action):
        self.actions[action] = 1
        self.game.set_action(self.actions)
        self.game.advance_action()
        reward = self.game.get_last_reward()
        self.actions[action] = 0

        state = self.game.get_state()

        cur_hits = self.game.get_game_variable(vzd.GameVariable.HITCOUNT)
        cur_hits_taken = self.game.get_game_variable(vzd.GameVariable.HITS_TAKEN)
        cur_damage = self.game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN)
        cur_kills = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        dead = self.game.get_game_variable(vzd.GameVariable.DEAD)

        damage = cur_damage - self.prev_damage
        self.prev_damage = cur_damage
        reward -= damage

        if dead:
            reward -= 100

        if cur_kills > self.num_kills and cur_hits > self.num_hits:
            print("Killed opponent!")
            self.num_kills = cur_kills
            reward += 300

        if cur_hits > self.num_hits:
            print("Shot oppnenet!")
            reward += 100
        elif action == 0:
            reward -= 1

        self.num_hits = cur_hits

        if damage > 0:
            print(f"Damaged! {damage}")
        self.num_taken_hits = cur_hits_taken

        # print(f"depth buffer : {depth_buf}")
        # print(f"objects : {objects}")

        # print(f"action : {action} reward : {reward}")

        is_terminated = self.game.is_episode_finished()
        is_truncated = self.step_cnt > self.maximum_steps

        if is_terminated or is_truncated:
            if is_truncated:
                print("Truncated!")
            print(f"total reward : {self.total_reward}")
            return np.zeros([4, 480, 640]), self.total_reward, True, dict()

        self.step_cnt += 1

        if self.step_cnt == self.maximum_steps:
            reward = -10000

        self.total_reward += reward
        return self.wrap_state(state), reward, False, dict()

    def reset(self):
        self.game.new_episode()
        self.step_cnt = 0
        self.total_reward = 0
        self.num_hits = 0
        self.num_taken_hits = 0
        self.num_kills = 0
        self.prev_damage = 0

        state = self.game.get_state()
        return self.wrap_state(state)

    def wrap_state(self, state: vzd.GameState):
        screen_buffer = state.screen_buffer
        objects = state.objects

        # Depth buffer
        depth_buffer = state.depth_buffer
        # print(f"depth buffer : {depth_buffer}")
        # print(f"screen buffer : {screen_buffer}")
        # Objects in current state (including enemies)
        # objects = state.objects
        cur_state = np.concatenate(
            (screen_buffer, np.expand_dims(depth_buffer, 0)), axis=0
        )
        return cur_state

    def seed(self, val):
        self.game.set_seed(val)

    def close(self):
        self.game.close()


class SaveModelCallBack(BaseCallback):
    def __init__(self, freq, path, verbose=1):
        super(SaveModelCallBack, self).__init__(verbose)
        self.frequency = freq
        self.model_path = path

    def _on_step(self):
        if self.n_calls % self.frequency == 0:
            self.model.save(
                os.path.join(self.model_path, "model_iter_{}".format(self.n_calls))
            )

        return True


steps = 5000
env = DoomEnv("scenarios/deathmatch.cfg")
vec_env = make_vec_env(lambda: DoomEnv("scenarios/deathmatch.cfg"), 2)
policy_kwargs = dict(
    features_extractor_class=PolicyModel,
    features_extractor_kwargs=dict(features_dim=512),
)
# model = PPO(policy='CnnPolicy', policy_kwargs=policy_kwargs, env=env, verbose=True, learning_rate=1e-5*5, n_steps=steps, device="mps")
model = PPO.load(
    "model_outputs_nov17-3/model_iter_90000.zip", env=vec_env, device="mps"
)

callback = SaveModelCallBack(freq=10000, path="model_outputs_nov17-3")
model.learn(total_timesteps=steps * 100, callback=callback)

print("inference!")
env.reset()
is_done = False
action = 0

for i in range(0, 10):
    print(f"episode : {i}")
    while not is_done:
        state, reward, is_done, _ = env.step(action)
        action = model.predict(state)
    env.reset()
    is_done = False

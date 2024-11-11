from vizdoom import *
import vizdoom as vzd
import random
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from gym import Env
from gym.spaces import Discrete, Box
import cv2

class DoomEnv(Env):
    def __init__(self, scenario):
        super().__init__()
        self.game = DoomGame()
        self.game.load_config(scenario)
        self.game.set_sound_enabled(True)
        self.game.set_console_enabled(True)
        self.game.set_render_all_frames(True)

        self.game.set_living_reward(-1)
        self.game.set_death_penalty(100.0)
        self.game.set_available_buttons([
                            vzd.Button.MOVE_RIGHT,
                            vzd.Button.MOVE_LEFT,
                            vzd.Button.LOOK_UP,
                            vzd.Button.MOVE_UP, 
                            vzd.Button.MOVE_DOWN,
                            vzd.Button.JUMP,
                            vzd.Button.CROUCH,
                            vzd.Button.TURN_LEFT,
                            vzd.Button.TURN_RIGHT,
                            vzd.Button.ALTATTACK,
                            vzd.Button.LOOK_DOWN, 
                            vzd.Button.ATTACK, 
                            vzd.Button.USE, 
                            vzd.Button.ALTATTACK,
                            vzd.Button.RELOAD])

        self.game.init()
        self.observation_space = Box(low=0,high=255,shape=(256, 256, 1), dtype=np.uint8)
        self.action_space = Discrete(len(self.game.get_available_buttons()))
        self.actions = np.zeros(len(self.game.get_available_buttons()), dtype=np.uint8)

    def step(self, action):
        self.actions[action] = 1
        self.game.set_action(self.actions)
        self.game.advance_action()
        reward = self.game.get_last_reward()
        self.actions[action] = 0

        if self.game.get_state():
            state = self.game.get_state().screen_buffer
            state = self.grayscale(state)
            info = self.game.get_state().game_variables[0]
        else:
            state = np.zeros(self.observation_space.shape)
            info = 0
        
        print(f"action : {action} reward : {reward}")
        info = {"info":info}
        done = self.game.is_episode_finished()
        if done:
            total_reward = self.game.get_total_reward()
            print(f"total reward : {total_reward}")
        return state, reward, done, info

    def reset(self):
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        state = self.grayscale(state)
        return state

    def grayscale(self,observation):
        """Grayscale, trim the bottom infos and reduce the number of pixels"""
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (256,256), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (256,256,1))
        state = state[:, :]
        return state

    def close(self):
        self.game.close()


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

steps = 2048
env = DoomEnv("scenarios/basic.cfg")
model = PPO('CnnPolicy', env, verbose=True, learning_rate=0.001, n_steps=steps, device="mps")

callback = TrainAndLoggingCallback(check_freq=10000, save_path="output")
model.learn(total_timesteps=steps*1000, callback=callback)

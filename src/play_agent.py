import time
from vizdoom import *
import vizdoom as vzd
import os
import numpy as np
from stable_baselines3 import PPO
from gym import Env
from gym.spaces import Discrete, Box
import cv2
import matplotlib.pyplot as plt


class DoomEnv(Env):
    def __init__(self, scenario, initial_difficulty=1):
        super().__init__()
        self.scenario = scenario
        self.difficulty = initial_difficulty
        self.init_game()

        self.observation_space = Box(
            low=0, high=255, shape=(3, 100, 160), dtype=np.uint8
        )
        self.action_space = Discrete(len(self.game.get_available_buttons()))

    def init_game(self):
        self.game = DoomGame()
        self.game.load_config(self.scenario)
        self.game.set_sound_enabled(True)
        self.game.set_console_enabled(False)
        self.game.set_render_all_frames(False)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_doom_skill(self.difficulty)
        self.game.set_available_buttons(
            [
                vzd.Button.MOVE_FORWARD,
                vzd.Button.MOVE_BACKWARD,
                vzd.Button.TURN_LEFT,
                vzd.Button.TURN_RIGHT,
                vzd.Button.MOVE_LEFT,
                vzd.Button.MOVE_RIGHT,
                vzd.Button.ATTACK,
            ]
        )
        self.game.init()

    def step(self, action):
        actions = np.eye(7, dtype=np.uint8)
        self.game.make_action(actions[action], 4)

        reward = self.game.get_last_reward()
        done = self.game.is_episode_finished()

        if self.game.get_state():
            state = self._process_observation(self.game.get_state().screen_buffer)
        else:
            state = np.zeros(self.observation_space.shape)

        return state, reward, done, {}

    def reset(self):
        self.game.new_episode()
        if self.game.get_state():
            return self._process_observation(self.game.get_state().screen_buffer)
        else:
            return np.zeros(self.observation_space.shape)

    def _process_observation(self, observation):
        resized = cv2.resize(
            np.moveaxis(observation, 0, -1), (160, 100), interpolation=cv2.INTER_AREA
        )
        return np.moveaxis(resized, -1, 0)

    def close(self):
        self.game.close()


def play_agent(env, model, num_episodes=5):
    """
    A function to play the agent in the environment for a number of episodes.
    """
    for episode in range(num_episodes):
        print(f"--- Starting episode {episode + 1} ---")
        obs = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

        print(f"Episode finished with a total reward of {total_reward}.")
    print("End of episodes.")


if __name__ == "__main__":
    scenario_path = "deadly_corridor.cfg"
    model_path = "../trained_agent.zip"

    # Charger l'environnement
    env = DoomEnv(scenario_path)

    # Charger le modèle pré-entraîné
    if os.path.exists(model_path):
        model = PPO.load(model_path, env=env)
        print(f"Model {model_path} loaded successfully.")
    else:
        raise FileNotFoundError(f"The model file {model_path} was not found.")

    # Faire jouer l'agent
    play_agent(env, model, num_episodes=5)

    # Fermer l'environnement
    env.close()

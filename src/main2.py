from vizdoom import *
import vizdoom as vzd
import os
import numpy as np
import cv2
from gym import Env
from gym.spaces import Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt


class DoomEnv(Env):
    def __init__(self, scenario, initial_difficulty=1):
        super().__init__()
        self.scenario = scenario
        self.difficulty = initial_difficulty
        self.init_game()

        self.observation_space = Box(low=0, high=255, shape=(100, 160, 1), dtype=np.uint8)
        self.action_space = Discrete(len(self.game.get_available_buttons()))
        self.previous_health = 100
        self.hitcount = 0
        self.ammo = 52
        self.no_damage_steps = 0  # Counter for steps without taking damage
        self.safe_to_move_forward = False

    def init_game(self):
        self.game = DoomGame()
        self.game.load_config(self.scenario)
        self.game.set_sound_enabled(True)
        self.game.set_console_enabled(False)
        self.game.set_render_all_frames(False)
        self.game.set_doom_skill(self.difficulty)
        self.game.set_available_buttons([
            vzd.Button.MOVE_FORWARD,
            vzd.Button.MOVE_BACKWARD,
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT,
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.ATTACK
        ])
        self.game.set_living_reward(0)
        self.game.set_death_penalty(-100)
        self.game.init()

    def step(self, action):
        actions = np.eye(7, dtype=np.uint8)
        movement_reward = self.game.make_action(actions[action], 4) # Repeat each action for 4 frames
        reward = movement_reward 

        if self.game.get_state(): # Check if episode is still running
            state = self.game.get_state().screen_buffer # Get current game state
            state = self._process_observation(state) # Process observation

            # Get available game variables
            game_vars = self.game.get_state().game_variables # Health, Hits, Ammo
            health = game_vars[0] if len(game_vars) > 0 else self.previous_health # Get health if available
            hitcount = game_vars[1] if len(game_vars) > 1 else self.hitcount # Get hitcount if available
            ammo = game_vars[2] if len(game_vars) > 2 else self.ammo # Get ammo

            # Calculate changes in health, hits, and ammo
            damage_taken_delta = self.previous_health - health # Calculate damage taken
            self.previous_health = health # Update previous health

            hitcount_delta = hitcount - self.hitcount # Calculate hits
            self.hitcount = hitcount # Update hitcount

            ammo_delta = self.ammo - ammo # Calculate ammo
            self.ammo = ammo # Update ammo

            # Rewards and penalties based on actions
            reward += damage_taken_delta * -10  # Penalty for health loss
            reward += hitcount_delta * 100  # Reward for hitting enemies
            reward += ammo_delta * -10  # Penalty for wasting ammo

            # Track no damage steps to enable moving forward
            if damage_taken_delta > 0:
                self.no_damage_steps = 0  # Reset counter if damage is taken
                self.safe_to_move_forward = False
            else:
                self.no_damage_steps += 1

            if self.no_damage_steps > 10:
                self.safe_to_move_forward = True

            # Reward or penalize moving forward based on safety
            if action == 0:  # MOVE_FORWARD
                if self.safe_to_move_forward:
                    reward += 500  # Reward for moving forward when safe
                else:
                    reward -= 20  # Penalty for moving forward prematurely

            info = {"health": health, "hitcount": hitcount, "safe_to_move": self.safe_to_move_forward}
        else:
            state = np.zeros(self.observation_space.shape)
            info = {"health": 0, "hitcount": 0, "safe_to_move": False}

        done = self.game.is_episode_finished()
        return state, reward, done, info

    def reset(self):
        self.previous_health = 100
        self.hitcount = 0
        self.ammo = 52
        self.no_damage_steps = 0
        self.safe_to_move_forward = False
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self._process_observation(state)

    def increase_difficulty(self):
        if self.difficulty < 5:
            self.difficulty += 1
            self.game.close()
            self.init_game()

    def _process_observation(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=-1)

    def close(self):
        self.game.close()


class DifficultyProgressCallback(BaseCallback):
    def __init__(self, env, verbose=1):
        super(DifficultyProgressCallback, self).__init__(verbose)
        self.env = env
        self.consecutive_wins = 0
        self.epoch_rewards = []
        self.epoch_timesteps = []
        self.epoch = 0

    def _on_step(self):
        n_steps = self.locals["self"].n_steps
        if self.n_calls % n_steps == 0:
            self.epoch += 1
            rewards = self.locals["rollout_buffer"].rewards
            mean_reward = np.mean(rewards)
            self.epoch_rewards.append(mean_reward)
            self.epoch_timesteps.append(self.n_calls)
            if self.verbose:
                print(f"[Epoch {self.epoch}] Timesteps: {self.n_calls}, Mean Reward: {mean_reward:.2f}")

            # Check if agent succeeded
            ep_info = self.locals.get("infos", [])
            if ep_info and ep_info[-1].get("health", 0) > 0:
                self.consecutive_wins += 1
            else:
                self.consecutive_wins = 0

            # Increase difficulty if agent wins 5 times in a row
            if self.consecutive_wins >= 5:
                print(f"Agent mastered difficulty {self.env.difficulty}. Increasing difficulty!")
                self.env.increase_difficulty()
                self.consecutive_wins = 0

        return True

    def _on_training_end(self):
        if self.verbose:
            print("Training Finished!")
            print(f"Total Epochs: {self.epoch}, Final Mean Reward: {self.epoch_rewards[-1]:.2f}")


if __name__ == "__main__":
    scenario_path = "scenarios/deadly_corridor.cfg"
    env = DoomEnv(scenario_path)

    model = PPO("CnnPolicy", env, verbose=1, learning_rate=0.00001, n_steps=8192, batch_size=64, n_epochs=10, clip_range=0.1, gamma=0.95, gae_lambda=0.9, device="cpu")

    progress_callback = DifficultyProgressCallback(env, verbose=1)

    model.learn(total_timesteps=100000, callback=progress_callback)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    plt.plot(progress_callback.epoch_timesteps, progress_callback.epoch_rewards)
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Learning Progress Over Epochs")
    plt.show()

    env.close()

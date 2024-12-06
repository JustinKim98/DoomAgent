import time
from vizdoom import *
import vizdoom as vzd
import os
import numpy as np
import cv2
from gym import Env
from gym.spaces import Discrete, Box
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import torch
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import shutil
import matplotlib
matplotlib.use("Agg")


class DoomEnv(Env):
    def __init__(self, scenario, initial_difficulty=2):
        super().__init__()
        self.scenario = scenario
        self.difficulty = initial_difficulty
        self.init_game()

        self.observation_space = Box(low=0, high=255, shape=(3, 100, 160), dtype=np.uint8)
        self.action_space = Discrete(len(self.game.get_available_buttons()))
        self.previous_health = 100
        self.hitcount = 0
        self.ammo = 52
        self.previous_vest_distance = None
        self.detected_enemies = 0
        self.consecutive_wins = 0  # Ajout du compteur de victoires consécutives
        self.enemy_labels = ["Zombieman", "ShotgunGuy", "ChaingunGuy", "Imp", "Demon", "Spectre"]

    def init_game(self):
        self.game = DoomGame()
        self.game.load_config(self.scenario)
        self.game.set_sound_enabled(True)
        self.game.set_console_enabled(False)
        self.game.set_render_all_frames(False)
        self.game.set_labels_buffer_enabled(True)
        self.game.set_depth_buffer_enabled(True)
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
        movement_reward = self.game.make_action(actions[action], 4)
        reward = movement_reward

        if self.game.get_state():
            state = self.game.get_state()
            screen_buffer = self._process_observation(state.screen_buffer)
            labels = state.labels

            vest_distance = None
            self.detected_enemies = False

            for label in labels:
                if label.object_name in self.enemy_labels:
                    self.detected_enemies = True
                    # print(f"Enemy detected: {label.object_name}")
                    break

            # GreenArmor detection
            for label in labels:
                if label.object_name == "GreenArmor":
                    vest_distance = np.sqrt(label.x ** 2 + label.y ** 2)
                    # print(f"GreenArmor detected at distance: {vest_distance}")
                    break

            if self.detected_enemies:
                if action == 6:  # ATTACK
                    reward += 50 
                    # print("Attacking enemies!")
                else:
                    reward -= 25 
                    # print("Ignoring enemies!")
            elif action == 6:
                reward -= 10
                # print("No enemies to attack!")

            if vest_distance is not None:
                if self.previous_vest_distance is not None:
                    distance_delta = self.previous_vest_distance - vest_distance
                    reward += 100 if distance_delta > 0 else -100
                self.previous_vest_distance = vest_distance

                # Reward for reaching the GreenArmor
                threshold = 25
                if vest_distance < threshold:
                    reward += 200
                    print("GreenArmor reached!")
            else:
                reward -= 40  # Penalty for not detecting the GreenArmor

            # Health, hitcount, ammo
            game_vars = state.game_variables
            health = game_vars[0] if len(game_vars) > 0 else self.previous_health
            damage_taken_delta = self.previous_health - health
            self.previous_health = health

            hitcount = game_vars[1] if len(game_vars) > 1 else self.hitcount
            ammo = game_vars[2] if len(game_vars) > 2 else self.ammo
            hitcount_delta = hitcount - self.hitcount
            ammo_delta = self.ammo - ammo

            reward += hitcount_delta * 80  
            reward += ammo_delta * -10  
            if damage_taken_delta > 0:
                reward -= damage_taken_delta * 10  
            if health <= 0:
                reward -= 150  

            self.hitcount = hitcount
            self.ammo = ammo

            info = {
                "health": health,
                "hitcount": hitcount,
                "ammo": ammo,
                "vest_distance": vest_distance,
                "detected_enemies": self.detected_enemies,
            }
        else:
            screen_buffer = np.zeros(self.observation_space.shape)
            info = {"health": 0, "hitcount": 0, "ammo": 0, "vest_distance": None, "detected_enemies": False}

        done = self.game.is_episode_finished()
        return screen_buffer, reward, done, info




    def reset(self):

        self.previous_health = 100
        self.hitcount = 0
        self.ammo = 52
        self.previous_vest_distance = None
        self.detected_enemies = 0
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self._process_observation(state)


    def _process_observation(self, observation):
        resized = cv2.resize(np.moveaxis(observation, 0, -1), (160, 100), interpolation=cv2.INTER_AREA)
        return np.moveaxis(resized, -1, 0)  # Convert back to channels-first





    def close(self):
        self.game.close()

class EpochLoggerCallback(BaseCallback):
    def __init__(self, verbose=1, n_steps_per_epoch=2048, env=None):
        super(EpochLoggerCallback, self).__init__(verbose)
        self.n_steps_per_epoch = n_steps_per_epoch
        self.current_epoch = 0
        self.env = env  
        self.rewards = []
        self.healths = []
        self.hitcounts = []

        self.epoch_rewards = []
        self.epoch_healths = []
        self.epoch_hitcounts = []

        self.writer = SummaryWriter("runs/doom_agent")

    def _on_step(self) -> bool:

        if self.env.game.get_state():
            state = self.env.game.get_state()
            game_vars = state.game_variables

            self.epoch_rewards.append(self.env.game.get_last_reward())
            self.epoch_healths.append(game_vars[0])
            self.epoch_hitcounts.append(game_vars[1])

        self.current_epoch = self.num_timesteps // self.n_steps_per_epoch

        if self.num_timesteps % self.n_steps_per_epoch == 0:
            if self.verbose:
                print(f"Epoch {self.current_epoch} completed.")

            if self.epoch_rewards:
                avg_reward = np.mean(self.epoch_rewards)
                avg_health = np.mean(self.epoch_healths)
                avg_hitcount = np.mean(self.epoch_hitcounts)

                self.rewards.append(avg_reward)
                self.healths.append(avg_health)
                self.hitcounts.append(avg_hitcount)

                self.writer.add_scalar("Average Reward", avg_reward, self.current_epoch)
                self.writer.add_scalar("Average Health", avg_health, self.current_epoch)
                self.writer.add_scalar("Enemies Eliminated", avg_hitcount, self.current_epoch)

            self.epoch_rewards = []
            self.epoch_healths = []
            self.epoch_hitcounts = []

        try:
            if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
                approx_kl = self.model.logger.name_to_value.get("rollout/approx_kl", None)
                value_loss = self.model.logger.name_to_value.get("train/value_loss", None)

                if approx_kl is not None:
                    self.writer.add_scalar("Approx KL", approx_kl, self.num_timesteps)
                    # print(f"Approx KL: {approx_kl}")
                if value_loss is not None:
                    self.writer.add_scalar("Value Loss", value_loss, self.num_timesteps)
                    # print(f"Value Loss: {value_loss}")

        except Exception as e:
            print(f"Warning: Could not log PPO metrics due to {e}")

        return True


    def plot_metrics(self):

        epochs = range(len(self.rewards))

        plt.figure()
        plt.plot(epochs, self.rewards, label="Average Reward")
        plt.xlabel("Epochs")
        plt.ylabel("Average Reward")
        plt.title("Evolution of Average Rewards")
        plt.legend()
        plt.grid()
        plt.savefig("average_rewards.png") 
        print("Saved average_rewards.png")

        plt.figure()
        plt.plot(epochs, self.healths, label="Average Health", color="green")
        plt.xlabel("Epochs")
        plt.ylabel("Average Health")
        plt.title("Evolution of Average Health")
        plt.legend()
        plt.grid()
        plt.savefig("average_health.png")
        print("Saved average_health.png")

        plt.figure()
        plt.plot(epochs, self.hitcounts, label="Enemies Eliminated", color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Enemies Eliminated")
        plt.title("Evolution of Enemies Eliminated")
        plt.legend()
        plt.grid()
        plt.savefig("enemies_eliminated.png") 
        print("Saved enemies_eliminated.png")


    def on_training_end(self):
        self.writer.close()







def record_video(env, model, video_path, video_fps=30, num_episodes=6):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, video_fps, (320, 200), isColor=True)

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            frame = np.moveaxis(obs, 0, -1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.resize(frame, (320, 200), interpolation=cv2.INTER_AREA)
            out.write(frame)

            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)

    out.release()
    print(f"Video recorded at: {video_path}")

    if os.path.exists(video_path):
        size = os.path.getsize(video_path)
        if size > 0:
            print(f"Video successfully created with size: {size} bytes.")
        else:
            print("Error: Video file is empty.")
    else:
        print("Error: Video file was not created.")

class AgentWrapper:
    def __init__(self, policy="CnnPolicy", env=None, model_path=None, device="cuda"):
        """
        Wrapper to handle the agent.
        """
        self.env = env
        self.policy = policy
        self.device = device
        if model_path and os.path.exists(model_path):
            print(f"Loading the model from : {model_path}")
            self.model = PPO.load(model_path, env=self.env, device=self.device)
        else:
            print("Creation of a new model.")
            self.model = PPO(
                policy=self.policy,
                env=self.env,
                verbose=1,
                learning_rate=0.00001,
                n_steps=2048,
                batch_size=256,
                n_epochs=4,
                clip_range=0.1,
                gamma=0.95,
                gae_lambda=0.9,
                device=self.device,
            )

    def train(self, total_timesteps, callback=None):
        """
        Train the agent for a number of timesteps.
        """
        print(f"Training the model for {total_timesteps} timtesteps...")
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, path):
        """
        Save the model to a file.
        """
        print(f"Model saved at : {path}")
        self.model.save(path)

    def predict(self, observation):
        """
        Predict the action to take given an observation.
        """
        return self.model.predict(observation)

    @staticmethod
    def load(path, env=None, device="cuda"):
        """
        Load the model from a file.
        """
        print(f"Loading the model from : {path}")
        return PPO.load(path, env=env, device=device)


if __name__ == "__main__":
    scenario_path = "scenarios/deadly_corridor.cfg"
    new_model_path = "trained_agent_intermediate.zip" 

    env = DoomEnv(scenario_path)

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. The code will run on the CPU.")


    agent_wrapper = AgentWrapper(env=env, model_path=None)  


    epoch_logger = EpochLoggerCallback(verbose=1, n_steps_per_epoch=2048, env=env)


    total_timesteps = 2048 * 900

    print("Training the agent...")
    agent_wrapper.train(total_timesteps=total_timesteps, callback=epoch_logger)

 
    print(f"Saving the agent at : {new_model_path}")
    agent_wrapper.save(new_model_path)

    # Évaluation
    mean_reward, std_reward = evaluate_policy(agent_wrapper.model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

   
    epoch_logger.plot_metrics()

  
    # video_path = os.path.abspath("trained_agent_video.mp4")
    # record_video(env, agent_wrapper.model, video_path)

    env.close()


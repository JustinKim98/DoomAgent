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
import shutil

class DoomEnv(Env):
    def __init__(self, scenario, initial_difficulty=1):
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
            screen_buffer = state.screen_buffer
            screen_buffer = self._process_observation(screen_buffer)

            labels = state.labels
            depth_buffer = state.depth_buffer

            vest_distance = None
            for label in labels:
                if label.object_name == "GreenArmor":
                    # print(f"Vest detected at: {label.x}, {label.y}")
                    vest_distance = np.sqrt(label.x ** 2 + label.y ** 2)
                    # print(f"Distance: {vest_distance}")
                    break
                

            if vest_distance is not None:
                if self.previous_vest_distance is not None:
                    distance_delta = self.previous_vest_distance - vest_distance
                    if distance_delta > 0:
                        reward += 10
                    elif distance_delta < 0:
                        reward += -10
                self.previous_vest_distance = vest_distance
            else :
                # print("Vest not detected")
                reward += -40

            game_vars = state.game_variables
            health = game_vars[0] if len(game_vars) > 0 else self.previous_health
            hitcount = game_vars[1] if len(game_vars) > 1 else self.hitcount
            ammo = game_vars[2] if len(game_vars) > 2 else self.ammo
            damage_taken_delta = self.previous_health - health
            self.previous_health = health
            hitcount_delta = hitcount - self.hitcount
            self.hitcount = hitcount
            ammo_delta = self.ammo - ammo
            self.ammo = ammo

            reward += damage_taken_delta * -10
            reward += hitcount_delta * 100
            reward += ammo_delta * -40

            if health <= 0:
                reward += -100

            info = {
                "health": health,
                "hitcount": hitcount,
                "ammo": ammo,
                "vest_distance": vest_distance,
                "detected_enemies": self.detected_enemies
            }
        else:
            screen_buffer = np.zeros(self.observation_space.shape)
            info = {
                "health": 0,
                "hitcount": 0,
                "ammo": 0,
                "vest_distance": None,
                "detected_enemies": self.detected_enemies
            }

        done = self.game.is_episode_finished()
        return screen_buffer, reward, done, info

    def check_victory(self):
        if self.game.is_episode_finished():
            state = self.game.get_state()
            if state:
                game_vars = state.game_variables
                health = game_vars[0] if len(game_vars) > 0 else 0  # HEALTH
                ammo = game_vars[2] if len(game_vars) > 2 else 0  # AMMO

                # Critères de victoire
                if health > 0 and ammo > 0:
                    return True  # Victoire
        return False  # Défaite



    def reset(self):
        # Vérifiez si l'agent a gagné
        if self.check_victory():
            self.consecutive_wins += 1
            print(f"Victory! Consecutive wins: {self.consecutive_wins}")
            if self.consecutive_wins >= 5:
                print(f"Agent has won 5 consecutive episodes. Increasing difficulty.")
                self.increase_difficulty()
                self.consecutive_wins = 0
        else:
            # print("Defeat. Resetting consecutive wins.")
            self.consecutive_wins = 0

        # Réinitialisez l'environnement
        self.previous_health = 100
        self.hitcount = 0
        self.ammo = 52
        self.previous_vest_distance = None
        self.detected_enemies = 0
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self._process_observation(state)



    def increase_difficulty(self):
        if self.difficulty < 5:
            self.difficulty += 1
            self.game.close()
            self.init_game()

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
        self.env = env  # Référence à l'environnement

    def _on_step(self) -> bool:
        # Calculer l'époque actuelle
        self.current_epoch = self.num_timesteps // self.n_steps_per_epoch

        # Si une époque est terminée
        if self.num_timesteps % self.n_steps_per_epoch == 0:
            if self.verbose:
                print(f"Epoch {self.current_epoch} completed.")

                # Incrémenter la difficulté toutes les 500 époques
                if self.current_epoch % 500 == 0 and self.env.difficulty < 3:
                    self.env.difficulty += 1
                    self.env.init_game()  # Recharger le jeu avec la nouvelle difficulté
                    print(f"Difficulty increased to {self.env.difficulty}")

        return True





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

    # Vérifiez si la vidéo a une taille correcte
    if os.path.exists(video_path):
        size = os.path.getsize(video_path)
        if size > 0:
            print(f"Video successfully created with size: {size} bytes.")
        else:
            print("Error: Video file is empty.")
    else:
        print("Error: Video file was not created.")


if __name__ == "__main__":
    scenario_path = "scenarios/deadly_corridor.cfg"
    env = DoomEnv(scenario_path)

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. The code will run on the CPU.")

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=0.00001,
        n_steps=2048,  # Aligné avec n_steps_per_epoch pour un suivi cohérent
        batch_size=256,  # Divise exactement n_steps=2048
        n_epochs=4,  # Nombre d'itérations sur chaque lot
        clip_range=0.1,
        gamma=0.95,
        gae_lambda=0.9,
        device="cuda"
    )

    # Initialiser le callback avec l'environnement
    epoch_logger = EpochLoggerCallback(verbose=1, n_steps_per_epoch=2048, env=env)

    # Total timesteps pour 1500 époques
    total_timesteps = 2048 * 1500  

    model.learn(total_timesteps=total_timesteps, callback=epoch_logger)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    video_path = os.path.abspath("trained_agent_video.mp4")
    record_video(env, model, video_path)

    env.close()
 

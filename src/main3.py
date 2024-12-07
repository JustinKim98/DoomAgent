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
    def __init__(self, scenario, initial_difficulty=3):
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

            if self.game.is_player_dead():
                self.game.respawn_player()

            reward += damage_taken_delta * -30
            reward += hitcount_delta * 220
            reward += ammo_delta * -1

            if health <= 0:
                reward += -150

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
        self.rewards = []
        self.healths = []
        self.hitcounts = []

        # Variables temporaires pour l'accumulation
        self.epoch_rewards = []
        self.epoch_healths = []
        self.epoch_hitcounts = []

    def _on_step(self) -> bool:
        # Collecter les informations sur l'étape actuelle
        if self.env.game.get_state():
            state = self.env.game.get_state()
            game_vars = state.game_variables

            # Récompense cumulée, santé et hitcount
            self.epoch_rewards.append(self.env.game.get_last_reward())
            self.epoch_healths.append(game_vars[0])  # Santé
            self.epoch_hitcounts.append(game_vars[1])  # Ennemis touchés

        # Calculer l'époque actuelle
        self.current_epoch = self.num_timesteps // self.n_steps_per_epoch

        # Si une époque est terminée
        if self.num_timesteps % self.n_steps_per_epoch == 0:
            if self.verbose:
                print(f"Epoch {self.current_epoch} completed.")

            # Enregistrer les moyennes pour l'époque
            if self.epoch_rewards:
                self.rewards.append(np.mean(self.epoch_rewards))
                self.healths.append(np.mean(self.epoch_healths))
                self.hitcounts.append(np.mean(self.epoch_hitcounts))

            # Réinitialiser les variables temporaires
            self.epoch_rewards = []
            self.epoch_healths = []
            self.epoch_hitcounts = []

            # Augmenter la difficulté toutes les 300 époques
            if self.current_epoch % 300 == 0 and self.env.difficulty < 3:
                self.env.difficulty += 1
                self.env.init_game()  # Recharger le jeu avec la nouvelle difficulté
                print(f"Difficulty increased to {self.env.difficulty}")

        return True

    def plot_metrics(self):
        """Plot the recorded metrics."""
        epochs = range(len(self.rewards))

        # Average rewards
        plt.figure()
        plt.plot(epochs, self.rewards, label="Average Reward")
        plt.xlabel("Epochs")
        plt.ylabel("Average Reward")
        plt.title("Evolution of Average Rewards")
        plt.legend()
        plt.grid()
        plt.show()

        # Average health
        plt.figure()
        plt.plot(epochs, self.healths, label="Average Health", color="green")
        plt.xlabel("Epochs")
        plt.ylabel("Average Health")
        plt.title("Evolution of Average Health")
        plt.legend()
        plt.grid()
        plt.show()

        # Enemies eliminated
        plt.figure()
        plt.plot(epochs, self.hitcounts, label="Enemies Eliminated", color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Enemies Eliminated")
        plt.title("Evolution of Enemies Eliminated")
        plt.legend()
        plt.grid()
        plt.show()






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

class CorridorAgent:
    def __init__(self, policy="CnnPolicy", env=None, model_path=None, device="cuda"):
        """
        Wrapper pour gérer l'entraînement, la sauvegarde et le chargement d'un agent.
        """
        self.env = env

        if env == None:
            self.env = DoomEnv("deadly_corridor.cfg")

        self.policy = policy
        self.device = device
        self.action = 0
        if model_path and os.path.exists(model_path):
            print(f"Chargement du modèle depuis : {model_path}")
            self.model = PPO.load(model_path, env=self.env, device=self.device)
        else:
            print("Création d'un nouveau modèle.")
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
        Entraîne l'agent pour un certain nombre de timesteps.
        """
        print(f"Entraînement de l'agent pour {total_timesteps} étapes...")
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, path):
        """
        Sauvegarde le modèle entraîné.
        """
        print(f"Sauvegarde du modèle à : {path}")
        self.model.save(path)

    def predict(self, observation):
        """
        Prédit une action à partir d'une observation.
        """
        return self.model.predict(observation)

    def step(self):
        state, reward, is_done, _ = self.env.step(self.action)
        self.action = self.model.predict(state)[0].item()

        if is_done:
            return 0, is_done

        return reward, is_done

    @staticmethod
    def load(path, env=None, device="cuda"):
        """
        Charge un modèle existant.
        """
        print(f"Chargement du modèle depuis : {path}")
        return PPO.load(path, env=env, device=device)

    def close(self):
        self.env.close()


if __name__ == "__main__":
    scenario_path = "scenarios/deadly_corridor.cfg"
    env = DoomEnv(scenario_path)

    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. The code will run on the CPU.")

    # Initialiser l'agent avec ou sans modèle existant
    agent_wrapper = AgentWrapper(env=env, model_path="trained_agent.zip")

    # Initialiser le callback avec l'environnement
    epoch_logger = EpochLoggerCallback(verbose=1, n_steps_per_epoch=2048, env=env)

    # Total timesteps pour 1500 époques
    total_timesteps = 2048 * 1500

    # Entraîner l'agent
    agent_wrapper.train(total_timesteps=total_timesteps, callback=epoch_logger)

    # Sauvegarder le modèle entraîné
    agent_wrapper.save("trained_agent.zip")

    # Évaluation
    mean_reward, std_reward = evaluate_policy(agent_wrapper.model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # Tracer les métriques enregistrées
    epoch_logger.plot_metrics()

    # Enregistrement de la vidéo
    video_path = os.path.abspath("trained_agent_video.mp4")
    record_video(env, agent_wrapper.model, video_path)

    env.close()


 
 # Notes
 # We need to do a function deadly_corridor_daemon.predict(state) to get the action to take
 # Plot some graphs to see the evolution of the agent
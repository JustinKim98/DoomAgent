import gymnasium as gym  # Use gymnasium instead of gym
import numpy as np
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
from vizdoom import DoomGame, Mode, ScreenResolution, GameVariable


# Custom VizDoom environment
class DoomDefendCenterEnv(gym.Env):  # Inherit from gymnasium.Env
    def __init__(self):
        super(DoomDefendCenterEnv, self).__init__()

        # Initialize the game
        self.game = DoomGame()
        self.game.load_config("defend_the_center.cfg")
        self.game.set_doom_scenario_path("defend_the_center.wad")
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.set_mode(Mode.PLAYER)
        self.game.set_window_visible(False)
        self.game.init()

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(len(self.game.get_available_buttons()))
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
        )

        # Custom reward variables
        self.last_health = self.game.get_game_variable(GameVariable.HEALTH)
        self.last_ammo = self.game.get_game_variable(GameVariable.AMMO2)
        self.last_enemy_count = 0  # Initialize enemy count
        self.timestep = 0

    def step(self, action):
        # Perform action in the game
        self.timestep += 1  # Increment timestep
        print(f"Timestep: {self.timestep}")  # Log to console
        self.game.make_action(self._convert_action(action))

        # Check if the episode is finished
        done = self.game.is_episode_finished()
        state = self.game.get_state()

        if state:
            frame = state.screen_buffer
            processed_state = self.preprocess_frame(frame)
        else:
            processed_state = np.zeros(self.observation_space.shape, dtype=np.uint8)

        reward = self._calculate_reward()
        return processed_state, reward, done, False, {}

    def preprocess_frame(self, frame):
        """Preprocess Image: Resize, crop, and convert to grayscale."""
        frame = np.transpose(frame, (1, 2, 0))  # (C, H, W) -> (H, W, C)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
        frame = cv2.resize(frame, (640, 480))  # Resize to match observation space
        return np.stack([frame] * 3, axis=-1)  # Expand grayscale to 3 channels

    def _calculate_reward(self):
        """Reward shaping based on game variables."""
        reward = 0

        # Get current game variables
        current_health = self.game.get_game_variable(GameVariable.HEALTH)
        current_ammo = self.game.get_game_variable(GameVariable.AMMO2)

        # Check the number of enemies in the current state
        current_enemy_count = self.game.get_game_variable(GameVariable.KILLCOUNT)
        print("Enemy count:", current_enemy_count)

        # Reward for killing enemies
        if current_enemy_count < self.last_enemy_count:
            reward += (self.last_enemy_count - current_enemy_count) * 10
        self.last_enemy_count = current_enemy_count

        # Penalty for health loss
        if current_health < self.last_health:
            reward -= (self.last_health - current_health) * 0.1
        self.last_health = current_health

        # Penalty for ammo usage
        if current_ammo < self.last_ammo:
            reward -= (self.last_ammo - current_ammo) * 0.2
        self.last_ammo = current_ammo

        # Small survival reward
        reward += 1

        print(f"Step Reward: {reward}")
        return reward

    def _convert_action(self, action):
        """Convert discrete action to VizDoom-compatible format."""
        action_list = [0] * len(self.game.get_available_buttons())
        action_list[action] = 1
        return action_list

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            self.game.set_seed(seed)
        self.game.new_episode()
        state = self.game.get_state().screen_buffer
        return self.preprocess_frame(state), {}

    def render(self, mode="rgb_array"):
        """Render the game."""
        if mode == "rgb_array":
            return self.game.get_state().screen_buffer
        elif mode == "human":
            self.game.set_window_visible(True)
        else:
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")

    def close(self):
        """Close the environment and release resources."""
        self.game.close()


# Callback for saving the model
class SaveModelCallback(BaseCallback):
    """
    Callback for saving a model every `save_freq` steps.
    """

    def __init__(self, save_freq, save_path, verbose=1):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            save_file = f"{self.save_path}_step_{self.n_calls}.zip"
            # self.model.save(save_file)
            if self.verbose > 0:
                print(f"Model saved to {save_file}")
        return True


# Initialize the environment
env = DoomDefendCenterEnv()
check_env(env)

# Initialize the PPO agent
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log="./ppo_doom_tensorboard/")


save_freq = 10000  # Save every 10,000 steps ( for debugging)
save_path = "ppo_defend_center_checkpoint"

# Create and use the callback
save_model_callback = SaveModelCallback(save_freq=save_freq, save_path=save_path)

# Train the model with the callback
model.learn(total_timesteps=2048 * 1000, callback=save_model_callback)

# Save the final model
model.save("ppo_defend_center_final")

import vizdoom as vzd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from gym import Env
from gym.spaces import Discrete, Box


class BaseEnv(Env):
    def __init__(
        self,
        scenario,
        allowed_actions,
        frame_buffer_size=1,
        living_reward=0,
        shoot_opponent_reward=50,
        kill_opponent_reward=100,
        exploration_rate=0.1,
    ):
        super().__init__()

        # Set game options
        self.game = vzd.DoomGame()
        self.game.load_config(scenario)
        self.game.set_sound_enabled(False)
        self.game.set_console_enabled(True)
        self.game.set_render_all_frames(True)

        self.game.set_living_reward(living_reward)
        self.game.set_death_penalty(0)
        self.game.set_available_buttons(allowed_actions)

        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_automap_buffer_enabled(True)
        self.game.set_objects_info_enabled(True)
        self.game.set_sectors_info_enabled(True)
        self.game.set_labels_buffer_enabled(True)

        self.game.set_render_hud(True)
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_particles(False)
        self.game.set_render_decals(False)
        self.game.set_doom_skill(1)

        # Initialize vizddom environment
        self.game.init()
        img = cv2.resize(
            self.game.get_state().screen_buffer,
            None,
            fx=0.5,
            fy=0.5,
            interpolation=cv2.INTER_LINEAR,
        )
        plt.imsave("state.png", img)

        # Space configurations
        self.observation_space = Box(
            low=0, high=255, shape=(3 * frame_buffer_size, 240, 320), dtype=np.uint8
        )

        self.action_space_size = len(self.game.get_available_buttons())
        self.action_space = Discrete(self.action_space_size)
        self.actions = np.zeros(self.action_space_size, dtype=np.uint8)
        self.tics = 4

        self.shoot_opponent_reward = shoot_opponent_reward
        self.kill_opponent_reward = kill_opponent_reward
        self.exploration_rate = exploration_rate

        # Game variable configurations
        self.maximum_steps = 500000
        self.step_cnt = 0
        self.num_hits = 0
        self.num_taken_hits = 0
        self.total_reward = 0
        self.prev_damage = 0
        self.prev_damage_given = 0
        self.num_kills = 0
        self.prev_fragcount = 0

        self.frame_buffer_size = frame_buffer_size
        self.frame_buffer = [
            np.zeros([3, 240, 320], dtype=np.uint8)
            for i in range(0, self.frame_buffer_size)
        ]

    def step(self, action):
        if np.random.uniform(0, 1) < self.exploration_rate:
            action = np.random.choice(self.action_space_size, 1)[0]

        self.actions[action] = 1
        self.game.set_action(self.actions)
        self.game.advance_action(self.tics)
        reward = self.game.get_last_reward()
        self.actions[action] = 0

        state = self.game.get_state()

        cur_hits = self.game.get_game_variable(vzd.GameVariable.HITCOUNT)
        cur_hits_taken = self.game.get_game_variable(vzd.GameVariable.HITS_TAKEN)
        cur_damage = self.game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN)
        cur_damage_given = self.game.get_game_variable(vzd.GameVariable.DAMAGECOUNT)
        cur_kills = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)

        damage = cur_damage - self.prev_damage
        damage_given = cur_damage_given - self.prev_damage_given
        reward -= damage

        if cur_kills > self.num_kills and cur_hits > self.num_hits:
            print("Killed opponent!")
            self.num_kills = cur_kills
            reward += self.kill_opponent_reward

        if damage > 0:
            reward -= damage
            print(f"Damaged! {damage}")

        if damage_given > 0:
            reward += damage_given * 5
            print(f"Shot : {damage_given*5}")

        if action == 0 and damage_given == 0:
            reward -= 1

        self.num_hits = cur_hits
        self.num_taken_hits = cur_hits_taken
        self.prev_damage = cur_damage
        self.prev_damage_given = cur_damage_given

        is_terminated = self.game.is_episode_finished()
        is_truncated = self.step_cnt > self.maximum_steps

        if is_terminated or is_truncated:
            if is_truncated:
                print("Truncated!")
            print(f"total reward : {self.total_reward}")
            return (
                np.zeros([3 * self.frame_buffer_size, 240, 320]),
                self.total_reward,
                True,
                dict(),
            )

        self.step_cnt += 1

        if self.game.is_player_dead() and (not is_terminated):
            self.game.respawn_player()

        self.total_reward += reward
        return self.wrap_state(state), reward, False, dict()

    def reset(self):
        # reset all variables
        self.step_cnt = 0
        self.total_reward = 0
        self.num_hits = 0
        self.num_taken_hits = 0
        self.prev_damage = 0
        self.prev_damage_given = 0
        self.num_kills = 0
        self.prev_fragcount = 0

        self.game.new_episode()
        state = self.game.get_state()
        return self.wrap_state(state)

    def update_frame_buffer(self, state: vzd.GameState):
        screen_buffer = state.screen_buffer
        img = cv2.resize(
            screen_buffer, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR
        )
        img = np.transpose(img, [2, 0, 1])

        for i in range(0, self.frame_buffer_size - 1):
            self.frame_buffer[i] = self.frame_buffer[i + 1]
        self.frame_buffer[self.frame_buffer_size - 1] = img

        return np.concatenate(self.frame_buffer, axis=0)

    def wrap_state(self, state: vzd.GameState):
        wrapped_state = self.update_frame_buffer(state)
        return wrapped_state

    def seed(self, val):
        self.game.set_seed(val)

    def close(self):
        self.game.close()


# Env with continuous action space
class ContinuousEnv(Env):
    def __init__(
        self,
        scenario,
        frame_buffer_size=1,
        living_reward=0,
        shoot_opponent_reward=50,
        kill_opponent_reward=100,
        exploration_rate=0.1,
    ):
        super().__init__()

        # Set game options
        self.game = vzd.DoomGame()
        self.game.load_config(scenario)
        self.game.set_sound_enabled(False)
        self.game.set_console_enabled(True)
        self.game.set_render_all_frames(True)

        self.game.set_living_reward(living_reward)
        self.game.set_death_penalty(100.0)

        # Use continuous action space
        self.game.clear_available_buttons()
        self.game.add_available_button(vzd.Button.ATTACK)  # 0
        self.game.add_available_button(vzd.Button.MOVE_FORWARD_BACKWARD_DELTA, 10)  # 1
        self.game.add_available_button(vzd.Button.MOVE_LEFT_RIGHT_DELTA, 5)  # 2
        self.game.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA, 5)  # 3
        self.game.add_available_button(vzd.Button.MOVE_FORWARD)  # 4
        self.game.add_available_button(vzd.Button.MOVE_BACKWARD)  # 5
        self.game.add_available_button(vzd.Button.TURN_RIGHT)  # 6
        self.game.add_available_button(vzd.Button.TURN_LEFT)  # 7

        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_automap_buffer_enabled(True)
        self.game.set_objects_info_enabled(True)
        self.game.set_sectors_info_enabled(True)
        self.game.set_labels_buffer_enabled(True)

        self.game.set_render_hud(True)
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_particles(False)
        self.game.set_render_decals(False)
        self.game.set_doom_skill(1)

        # Initialize vizddom environment
        self.game.init()

        # Space configurations
        self.observation_space = Box(
            low=0, high=255, shape=(3 * frame_buffer_size, 240, 320), dtype=np.uint8
        )

        self.action_space_size = len(self.game.get_available_buttons())
        self.action_space = Box(low=-1.0, high=1.0, shape=(self.action_space_size,))
        self.actions = np.zeros(self.action_space_size, dtype=np.uint8)
        self.tics = 4

        self.shoot_opponent_reward = shoot_opponent_reward
        self.kill_opponent_reward = kill_opponent_reward
        self.exploration_rate = exploration_rate

        # Game variable configurations
        self.maximum_steps = 500000
        self.step_cnt = 0
        self.num_hits = 0
        self.num_taken_hits = 0
        self.total_reward = 0
        self.prev_damage = 0
        self.prev_damage_given = 0
        self.num_kills = 0
        self.prev_fragcount = 0

        self.frame_buffer_size = frame_buffer_size
        self.frame_buffer = [
            np.zeros([3, 240, 320], dtype=np.uint8)
            for i in range(0, self.frame_buffer_size)
        ]

    def step(self, action):

        action[0] = 1 if action[0] > 0 else 0
        action[1] *= 10
        action[2] *= 5
        action[3] *= 5
        action[4] = 1 if action[4] > 0 else 0
        action[5] = 1 if action[5] > 0 else 0
        action[6] = 1 if action[6] > 0 else 0
        action[7] = 1 if action[7] > 0 else 0

        self.actions = action
        reward = self.game.make_action(self.actions, tics=2)
        state = self.game.get_state()

        cur_hits = self.game.get_game_variable(vzd.GameVariable.HITCOUNT)
        cur_hits_taken = self.game.get_game_variable(vzd.GameVariable.HITS_TAKEN)
        cur_damage = self.game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN)
        cur_damage_given = self.game.get_game_variable(vzd.GameVariable.DAMAGECOUNT)
        cur_kills = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)

        damage = cur_damage - self.prev_damage
        damage_given = cur_damage_given - self.prev_damage_given
        reward -= damage

        if action[0] == 1 and cur_kills > self.num_kills and cur_hits > self.num_hits:
            print("Killed opponent!")
            reward += self.kill_opponent_reward

        if damage > 0:
            reward -= damage
            print(f"Damaged! {damage}")

        if action[0] == 1 and damage_given > 0:
            reward += damage_given * 5
            print(f"Shot : {damage_given*5}")

        if action[0] == 1 and damage_given == 0:
            print("missed shot")
            reward -= 1

        self.num_hits = cur_hits
        self.num_taken_hits = cur_hits_taken
        self.prev_damage = cur_damage
        self.prev_damage_given = cur_damage_given
        self.num_kills = cur_kills

        is_terminated = self.game.is_episode_finished()
        is_truncated = self.step_cnt > self.maximum_steps

        if is_terminated or is_truncated:
            if is_truncated:
                print("Truncated!")
            print(f"total reward : {self.total_reward}")
            return (
                np.zeros([3 * self.frame_buffer_size, 240, 320]),
                self.total_reward,
                True,
                dict(),
            )

        self.step_cnt += 1

        if self.game.is_player_dead() and (not is_terminated):
            self.game.respawn_player()

        self.total_reward += reward
        return self.wrap_state(state), reward, False, dict()

    def reset(self):
        self.game.new_episode()
        self.step_cnt = 0
        self.num_hits = 0
        self.num_taken_hits = 0
        self.total_reward = 0
        self.prev_damage = 0
        self.prev_damage_given = 0
        self.num_kills = 0
        self.prev_fragcount = 0

        state = self.game.get_state()
        return self.wrap_state(state)

    def update_frame_buffer(self, state: vzd.GameState):
        screen_buffer = state.screen_buffer
        img = cv2.resize(
            screen_buffer, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR
        )
        img = np.transpose(img, [2, 0, 1])

        for i in range(0, self.frame_buffer_size - 1):
            self.frame_buffer[i] = self.frame_buffer[i + 1]
        self.frame_buffer[self.frame_buffer_size - 1] = img

        return np.concatenate(self.frame_buffer, axis=0)

    def wrap_state(self, state: vzd.GameState):
        wrapped_state = self.update_frame_buffer(state)
        return wrapped_state

    def seed(self, val):
        self.game.set_seed(val)

    def close(self):
        self.game.close()

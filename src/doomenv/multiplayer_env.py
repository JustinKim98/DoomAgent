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
        infinite_run=False,
        game=None,
        configure_as_host=False,
    ):
        super().__init__()

        # Set game options
        if game is None:
            self.game = vzd.DoomGame()
        else:
            self.game = game

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
        self.game.set_render_corpses(False)
        self.game.set_render_weapon(True)
        self.game.set_render_particles(False)
        self.game.set_render_decals(False)
        self.game.set_doom_skill(1)

        self.game.add_game_args(
            "+viz_connect_timeout 60 "
            "-deathmatch "
            "+timelimit 10.0 "
            "+sv_forcerespawn 1 "
            "+sv_noautoaim 1 "
            "+sv_respawnprotect 1 "
            "+sv_spawnfarthest 1 "
            "+sv_nocrouch 1 "
            "+viz_respawn_delay 10 "
            "+viz_nocheat 1"
            "+name Host +colorset 0"
        )

        if configure_as_host:
            print("Multiplayer configurations applied")
            self.game.add_game_args("-host 2" "-port 5029 ")

        # Initialize vizddom environment
        self.game.init()

        # Frame buffer settings
        self.frame_storage_size = (
            int((frame_buffer_size * (frame_buffer_size - 1)) / 2) + 1
        )
        self.frame_buffer_size = frame_buffer_size
        self.frame_buffer = [
            np.zeros([3, 240, 320], dtype=np.uint8)
            for i in range(0, self.frame_storage_size)
        ]

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
        self.infinite_run = infinite_run

        # Game variable configurations
        self.maximum_steps = 1500
        self.step_cnt = 0
        self.num_hits = 0
        self.num_taken_hits = 0
        self.total_reward = 0
        self.prev_damage = 0
        self.prev_damage_given = 0
        self.num_kills = 0

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
        is_truncated = not self.infinite_run and self.step_cnt > self.maximum_steps

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

        if self.step_cnt % 100 == 0:
            print(f"steps : {self.step_cnt}")

        self.step_cnt += 1

        if self.game.is_player_dead() and (not is_terminated):
            self.game.respawn_player()

        self.total_reward += reward
        return self.wrap_state(state), reward, False, dict()

    def reset(self):
        # reset all variables
        self.step_cnt = 0
        self.total_reward = 0
        self.game.new_episode()
        self.num_hits = self.game.get_game_variable(vzd.GameVariable.HITCOUNT)
        self.num_taken_hits = self.game.get_game_variable(vzd.GameVariable.HITS_TAKEN)
        self.prev_damage = self.game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN)
        self.prev_damage_given = self.game.get_game_variable(
            vzd.GameVariable.DAMAGECOUNT
        )
        self.num_kills = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        state = self.game.get_state()
        return self.wrap_state(state)

    def update_frame_buffer(self, state: vzd.GameState):
        screen_buffer = state.screen_buffer
        img = cv2.resize(
            screen_buffer, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR
        )
        img = np.transpose(img, [2, 0, 1])

        for i in range(0, self.frame_storage_size - 1):
            self.frame_buffer[i] = self.frame_buffer[i + 1]
        self.frame_buffer[self.frame_storage_size - 1] = img

        cnt = 0
        env_frames = list()
        for i in range(0, self.frame_buffer_size):
            cnt += i
            env_frames.append(self.frame_buffer[len(self.frame_buffer) - cnt - 1])

        return np.concatenate(env_frames, axis=0)

    def wrap_state(self, state: vzd.GameState):
        wrapped_state = self.update_frame_buffer(state)
        return wrapped_state

    def seed(self, val):
        self.game.set_seed(val)

    def close(self):
        self.game.close()

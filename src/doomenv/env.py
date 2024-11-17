import vizdoom as vzd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from gym import Env
from gym.spaces import Discrete, Box

class BaseEnv(Env):
    def __init__(self, scenario):
        super().__init__()

        # Set game options
        self.game = vzd.DoomGame()
        self.game.load_config(scenario)
        self.game.set_sound_enabled(False)
        self.game.set_console_enabled(True)
        self.game.set_render_all_frames(True)

        self.game.set_living_reward(0)
        self.game.set_death_penalty(100.0)
        self.game.set_available_buttons([
                            vzd.Button.ATTACK, 
                            vzd.Button.MOVE_RIGHT,
                            vzd.Button.MOVE_LEFT,
                            vzd.Button.MOVE_UP, 
                            vzd.Button.MOVE_DOWN,
                            vzd.Button.TURN_LEFT,
                            vzd.Button.TURN_RIGHT,
                            # vzd.Button.ALTATTACK,
                            vzd.Button.RELOAD
                            ])
        
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
        img = cv2.resize(self.game.get_state().screen_buffer, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        plt.imsave('state.png', img)

        # Space configurations
        self.observation_space = Box(low=0,high=255,shape=(12, 240, 320), dtype=np.uint8)
        self.action_space = Discrete(len(self.game.get_available_buttons()))
        self.actions = np.zeros(len(self.game.get_available_buttons()), dtype=np.uint8)
        self.tics = 4
        
        # Game variable configurations
        self.maximum_steps = 50000
        self.num_hits = 0
        self.num_taken_hits = 0
        self.total_reward = 0
        self.prev_damage = 0
        self.num_kills = 0
        self.prev_fragcount = 0

        self.frame_buffer = [
            np.zeros([3, 240, 320], dtype=np.uint8) for i in range(0, 4)
        ]

    def step(self, action):
        self.actions[action] = 1
        self.game.set_action(self.actions)
        self.game.advance_action(4)
        reward = self.game.get_last_reward()
        self.actions[action] = 0

        state = self.game.get_state()

        cur_hits = self.game.get_game_variable(vzd.GameVariable.HITCOUNT)
        cur_hits_taken = self.game.get_game_variable(vzd.GameVariable.HITS_TAKEN)
        cur_damage = self.game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN)
        cur_kills = self.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        cur_fragcount = self.game.get_game_variable(vzd.GameVariable.FRAGCOUNT)
        dead = self.game.get_game_variable(vzd.GameVariable.DEAD)

        damage = cur_damage - self.prev_damage
        self.prev_damage = cur_damage
        reward -= damage

        if dead:
            reward -= 100

        if cur_kills > self.num_kills and cur_hits > self.num_hits:
            print("Killed opponent!")
            self.num_kills = cur_kills
            reward += 100

        reward += (cur_fragcount - self.prev_fragcount)*10
        if cur_fragcount != self.prev_fragcount:
            print(f"fragcount: {cur_fragcount - self.prev_fragcount}")
        self.prev_fragcount = cur_fragcount
            
            
        if cur_hits > self.num_hits:
            print("Shot oppnenet!")
            reward += 50 
        elif action == 0:
            reward -= 1

        self.num_hits = cur_hits

        if damage > 0:
            print(f"Damaged! {damage}")
        self.num_taken_hits = cur_hits_taken 

        is_terminated = self.game.is_episode_finished()
        is_truncated = self.step_cnt > self.maximum_steps

        if is_terminated or is_truncated :
            if is_truncated:
                print("Truncated!")
            print(f"total reward : {self.total_reward}")
            return np.zeros([12, 240, 320]), self.total_reward, True, dict()
        
        self.step_cnt += 1

        if self.step_cnt == self.maximum_steps:
            reward = -10000

        self.total_reward += reward

        if self.game.is_player_dead() and (not is_terminated):
            self.game.respawn_player()

        return self.wrap_state(state), reward, False, dict()

    def reset(self):
        self.game.new_episode()
        self.step_cnt = 0
        self.total_reward = 0
        self.num_hits = 0
        self.num_taken_hits = 0
        self.num_kills = 0
        self.prev_damage = 0
        self.prev_fragcount = 0

        state = self.game.get_state()
        return self.wrap_state(state)
    
    def update_frame_buffer(self, state : vzd.GameState):
        screen_buffer = state.screen_buffer
        img = cv2.resize(screen_buffer, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        img = np.transpose(img, [2, 0, 1])

        for i in range(0, 3):
            self.frame_buffer[i] = self.frame_buffer[i + 1]
        self.frame_buffer[3] = img

        img= np.concatenate(self.frame_buffer, 0)
        return img
    
    def wrap_state(self, state : vzd.GameState): 
        wrapped_state = self.update_frame_buffer(state)
        return wrapped_state
        # Depth buffer
        # depth_buffer = state.depth_buffer

        # print(f"depth buffer : {depth_buffer}")
        # print(f"screen buffer : {screen_buffer}")
        # Objects in current state (including enemies)
        # objects = state.objects
        
        return img 

    def seed(self, val):
        self.game.set_seed(val)

    def close(self):
        self.game.close()

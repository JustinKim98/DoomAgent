import vizdoom as vzd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from gym import Env
from gym.spaces import Discrete, Box

class DoomEnv(Env):
    def __init__(self, scenario):
        super().__init__()

        # Initialize game environment
        self.game = vzd.DoomGame()
        self.game.load_config(scenario)
        self.game.set_sound_enabled(True)
        self.game.set_console_enabled(True)
        self.game.set_render_all_frames(True)

        self.game.set_living_reward(-0.2)
        self.game.set_death_penalty(100.0)
        self.game.set_available_buttons([
                            vzd.Button.ATTACK, 
                            vzd.Button.MOVE_RIGHT,
                            vzd.Button.MOVE_LEFT,
                            vzd.Button.MOVE_UP, 
                            vzd.Button.MOVE_DOWN,
                            # vzd.Button.LOOK_UP,
                            # vzd.Button.LOOK_DOWN, 
                            # vzd.Button.CROUCH,
                            vzd.Button.TURN_LEFT,
                            vzd.Button.TURN_RIGHT,
                            vzd.Button.JUMP,
                            # vzd.Button.USE, 
                            vzd.Button.ALTATTACK,
                            vzd.Button.RELOAD])
        
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_depth_buffer_enabled(True)
        self.game.set_automap_buffer_enabled(True)
        self.game.set_objects_info_enabled(True)
        self.game.set_sectors_info_enabled(True)

        self.game.set_render_hud(False)
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_particles(False)
        self.game.set_render_decals(True)

        self.game.init()

        self.observation_space = Box(low=0,high=255,shape=(4, 480, 640), dtype=np.uint8)
        self.action_space = Discrete(len(self.game.get_available_buttons()))
        self.actions = np.zeros(len(self.game.get_available_buttons()), dtype=np.uint8)


        self.maximum_steps = 50000
        self.num_hits = 0
        self.num_taken_hits = 0
        self.total_reward = 0
        self.step_cnt = 0

    def step(self, action):
        self.actions[action] = 1
        self.game.set_action(self.actions)
        self.game.advance_action()
        reward = self.game.get_last_reward()
        self.actions[action] = 0

        # if action == 0:
        #     reward -= 1 

        state = self.game.get_state()

        cur_hits = self.game.get_game_variable(vzd.GameVariable.HITCOUNT)
        cur_hits_taken = self.game.get_game_variable(vzd.GameVariable.HITS_TAKEN)

        if cur_hits > self.num_hits:
            print("Shot oppnenet!")
            reward += 50

        if cur_hits_taken > self.num_taken_hits:
            print("Damaged!")
            reward -= 10

        self.num_hits = cur_hits
        self.num_taken_hits =cur_hits_taken 

        # print(f"screen buffer : {screen_buffer}")
        # print(f"depth buffer : {depth_buf}")
        # print(f"objects : {objects}")
        
        # print(f"action : {action} reward : {reward}")

        is_terminated = self.game.is_episode_finished()
        is_truncated = self.step_cnt > self.maximum_steps

        if is_terminated or is_truncated :
            if is_truncated:
                print("Truncated!")
            print(f"total reward : {self.total_reward}")
            return np.zeros([4, 480, 640]), self.total_reward, True, dict()
        
        self.step_cnt += 1
        if self.step_cnt == self.maximum_steps:
            reward -= 200

        self.total_reward += reward
        return self.wrap_state(state), reward, False, dict()

    def reset(self):
        self.game.new_episode()
        self.step_cnt = 0
        self.total_reward = 0
        self.num_hits = 0
        self.num_taken_hits = 0
        state = self.game.get_state()
        return self.wrap_state(state)
    
    def wrap_state(self, state : vzd.GameState): 
        screen_buffer = state.screen_buffer
        # Depth buffer
        depth_buffer = state.depth_buffer
        # print(f"depth buffer : {depth_buffer}")
        # print(f"screen buffer : {screen_buffer}")
        # Objects in current state (including enemies)
        # objects = state.objects
        cur_state = np.concatenate((screen_buffer.transpose(2, 0, 1), np.expand_dims(depth_buffer, 0)), axis=0)
        return cur_state

    def close(self):
        self.game.close()


if __name__ == "__main__":
    print("loaded model!")
    env = DoomEnv("scenarios/deathmatch.cfg")
    model = PPO.load("model_outputs/model_iter_200000.zip", env=env)

    env.reset()
    is_done = False
    action = 0

    for i in range (0, 10):
        print(f"episode : {i}")
        while(not is_done):
            state, reward, is_done, _ = env.step(action)
            action = model.predict(state)
        env.reset()
        is_done = False
#!/usr/bin/env python3

#####################################################################
# This script presents how to host a deathmatch game.
#####################################################################

import os
from random import choice
import sys
import numpy as np
import vizdoom as vzd
from infer_deathmatch import DeathmatchAgent

game = vzd.DoomGame()

def set_game_env(mode, model_path):

    model = None
    if mode == "corridor":
        model = DeathmatchAgent(model_path)
        #model = load_model(CORRIDOR_MODEL)
        #game.load_config(os.path.join(vzd.scenarios_path, "deadly_corridor.cfg"))
        actions = np.eye(7)

    elif mode == "dtc":
        #model = load_model(DTC_MODEL)
        #game.load_config(os.path.join(vzd.scenarios_path, "multi_duel.cfg"))
        actions = np.eye(3)


    elif mode == "deathmatch":
        #model = load_model(DEATHMATCH_MODEL)
        model = DeathmatchAgent(model_path, game)
        #game.load_config(os.path.join(vzd.scenarios_path, "multi.cfg"))
        actions = np.eye(8)

    return model, actions


#game.set_doom_map("map01")
model, actions = set_game_env(sys.argv[1], sys.argv[2])

game.add_game_args(
    "-host 2 "
    "-port 5029 "  
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
)

# game.add_game_args("+viz_spectator 1")

# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game.add_game_args("+name Host +colorset 0")

#game.set_mode(vzd.Mode.ASYNC_PLAYER)

# game.set_window_visible(False)

game.init()


total_reward = 0

while not game.is_episode_finished():

    s = game.get_state()

    #action = model.predict(s)
    #action = choice(actions)
    reward, _ = model.step()
    #game.make_action(action)
    total_reward += reward
    if game.is_player_dead():
        game.respawn_player()

print("Total Reward: " + str(total_reward))
game.close()

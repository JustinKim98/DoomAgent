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
from main3 import AgentWrapper
from defend_the_center_env import DoomDefendCenterEnv
game = vzd.DoomGame()


def set_game_env(mode, model_path):
    model = None
    if mode == "corridor":
        model = AgentWrapper(model_path= model_path)
        # model = load_model(CORRIDOR_MODEL)
        # game.load_config(os.path.join(vzd.scenarios_path, "deadly_corridor.cfg"))
        actions = np.eye(7)

    elif mode == "dtc":
        model = DoomDefendCenterEnv(model_path, game = game)
        # game.load_config(os.path.join(vzd.scenarios_path, "multi_duel.cfg"))
        actions = np.eye(3)

    elif mode == "deathmatch":
        model = DeathmatchAgent(model_path, game=game)
        # game.load_config(os.path.join(vzd.scenarios_path, "multi.cfg"))
        actions = np.eye(8)

    elif mode =

    return model, actions


# game.add_game_args("+viz_spectator 1")

# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
# game.set_doom_map("map01")
model, actions = set_game_env(sys.argv[1], sys.argv[2])


# game.set_mode(vzd.Mode.ASYNC_PLAYER)

# game.set_window_visible(False)

# game.init()


total_reward = 0

while not game.is_episode_finished():

    s = game.get_state()

    # action = model.predict(s)
    # action = choice(actions)
    reward, _ = model.step()
    # game.make_action(action)

    total_reward += reward
    if game.is_player_dead():
        game.respawn_player()

print("Total Reward: " + str(total_reward))
game.close()

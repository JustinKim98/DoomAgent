#!/usr/bin/env python3

#####################################################################
# This script presents how to host a deathmatch game.
#####################################################################

import os
from random import choice
import sys
import vizdoom as vzd
from deathmatch.infer_deathmatch import DeathmatchAgent
from corridor.corridor import CorridorAgent
from dtc.dtc import DefendTheCenterAgent
from multi.infer_multiplayer import MultiplayerAgent


def set_game_env(mode, model_path):
    model = None
    print(f"Mode : {mode}")
    if mode == "corridor":
        model = CorridorAgent(model_path=model_path)

    elif mode == "dtc":
        model = DefendTheCenterAgent(model_path)

    elif mode == "deathmatch":
        print("Invoking deathmatch")
        model = DeathmatchAgent(model_path)

    else:
        print("Invoke multi mode on host")
        game = vzd.DoomGame()
        game.set_window_visible(True)
        game.set_doom_map("map02")

        # Use CIG example config or your own.
        game.load_config(os.path.join(vzd.scenarios_path, "multi.cfg"))
        game.add_game_args(
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

        game.add_game_args("+name Host +colorset 0")
        game.add_game_args("-join 127.0.0.1 -port 5029")

        model = MultiplayerAgent(model_path, game=game)

    return model


model = set_game_env(sys.argv[1], sys.argv[2])


total_reward = 0
is_done = False

for i in range(0, 10):
    while not is_done:
        reward, is_done = model.step()
        total_reward += reward
    model.reset()

print("Total Reward: " + str(total_reward))
model.close()

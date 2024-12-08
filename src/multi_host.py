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
from main3 import CorridorAgent
from defend_the_center_env import DefendTheCenterAgent
from infer_multi import MultiAgent

def set_game_env(mode, model_path):
    if mode == "corridor":
        model = CorridorAgent(model_path= model_path)

    elif mode == "dtc":
        model = DefendTheCenterAgent(model_path)

    elif mode == "deathmatch":
        model = DeathmatchAgent(model_path)

    else:
        model = MultiAgent(model_path)

    return model

model = set_game_env(sys.argv[1], sys.argv[2])


total_reward = 0
is_done = False

while not is_done:

    reward, is_done = model.step()

    total_reward += reward


print("Total Reward: " + str(total_reward))
model.close()

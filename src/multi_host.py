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


def set_game_env(mode, model_path):
    model = None
    if mode == "corridor":
        model = CorridorAgent(model_path= model_path)
        actions = np.eye(7)

    elif mode == "dtc":
        model = DefendTheCenterAgent(model_path)
        actions = np.eye(3)

    elif mode == "deathmatch":
        model = DeathmatchAgent(model_path)
        actions = np.eye(8)

    #elif mode =

    return model

model = set_game_env(sys.argv[1], sys.argv[2])


total_reward = 0
is_done = False

while not is_done:

    reward, is_done = model.step()

    total_reward += reward


print("Total Reward: " + str(total_reward))
model.close()

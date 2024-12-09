#!/usr/bin/env python3

import os
import vizdoom as vzd
from pynput import keyboard
from time import sleep
import sys

# Initialize the Doom game
game = vzd.DoomGame()

game.set_window_visible(True)


def set_player_env(mode):
    if mode == "corridor":
        game.load_config(os.path.join(vzd.scenarios_path, "deadly_corridor.cfg"))

        current_action = [0] * 9
        key_mapping = {
            "space": [0, 0, 1, 0, 0, 0, 0],
            "z": [0, 0, 0, 0, 0, 1, 0],
            "c": [0, 0, 0, 0, 0, 0, 1],
            "d": [0, 1, 0, 0, 0, 0, 0],
            "a": [1, 0, 0, 0, 0, 0, 0],
            "w": [0, 0, 0, 1, 0, 0, 0],
            "s": [0, 0, 0, 0, 1, 0, 0],
        }
        return current_action, key_mapping

    if mode == "dtc":
        game.load_config(os.path.join(vzd.scenarios_path, "defend_the_center.cfg"))
        current_action = [0] * 3
        key_mapping = {"a": [1, 0, 0], "d": [0, 1, 0], "space": [0, 0, 1]}
        return current_action, key_mapping

    if mode == "deathmatch":
        game.load_config(os.path.join(vzd.scenarios_path, "deathmatch.cfg"))
        current_action = [0] * 8
        key_mapping = {
            "space": [1, 0, 0, 0, 0, 0, 0, 0],
            "d": [0, 1, 0, 0, 0, 0, 0, 0],
            "a": [0, 0, 1, 0, 0, 0, 0, 0],
            "w": [0, 0, 0, 1, 0, 0, 0, 0],
            "s": [0, 0, 0, 0, 1, 0, 0, 0],
            "z": [0, 0, 0, 0, 0, 1, 0, 0],
            "c": [0, 0, 0, 0, 0, 0, 1, 0],
            "r": [0, 0, 0, 0, 0, 0, 0, 1],
        }

        return current_action, key_mapping

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

current_action, key_mapping = set_player_env(sys.argv[1])

game.init()


def on_press(key):
    global current_action
    try:
        # check if the pressed key is in the mapping
        key_str = key.char if hasattr(key, "char") else key.name
        print(f"key : {key_str}")
        if key_str in key_mapping:
            current_action = key_mapping[key_str]

    except AttributeError:
        key_str = key.name
        if key_str in key_mapping:
            current_action = key_mapping[key_str]


def on_release(key):
    global current_action
    current_action = [0] * len(current_action)  # reset


listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

while not game.is_episode_finished():

    if game.is_player_dead():
        game.respawn_player()

    s = game.get_state()

    game.make_action(current_action)

    # add a small delay to reduce CPU usage
    sleep(0.05)


game.close()

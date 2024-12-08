#!/usr/bin/env python3

import os
import vizdoom as vzd
from time import sleep
import keyboard

# Initialize the Doom game
game = vzd.DoomGame()
game.set_window_visible(True)

current_action = [0] * 10
key_mapping = {
    "left": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # look left
    "right": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # look right
    "space": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # shoot
    "d": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # move right
    "a": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # move right
    "w": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # move front
    "s": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # move back
    "r": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # reload
    "up": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # look up
    "down": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # look down
}


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

game.add_game_args("+name Player +colorset 0")
game.add_game_args("-join 127.0.0.1 -port 5029")
game.set_doom_map("map02")


game.init()

def keyboard_listener():
    def handle_event(key):
        global current_action
        if key.event_type == "down":
            try:
                # check if the pressed key is in the mapping
                key_str = key.char if hasattr(key, "char") else key.name
                if key_str in key_mapping:
                    current_action = key_mapping[key_str]

            except AttributeError:
                key_str = key.name
                if key_str in key_mapping:
                    current_action = key_mapping[key_str]

        else:
            current_action = [0] * len(current_action)  # reset

    keyboard.hook(handle_event)


keyboard_listener()

while not game.is_episode_finished():

    if game.is_player_dead():
        game.respawn_player()

    s = game.get_state()

    game.make_action(current_action)

    # add a small delay to reduce CPU usage
    sleep(0.05)

keyboard.unhook_all()
game.close()

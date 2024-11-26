#!/usr/bin/env python3

import os
import vizdoom as vzd
import keyboard  # For handling key presses

# Initialize the Doom game
game = vzd.DoomGame()

# Use CIG example config or your own.
game.load_config(os.path.join(vzd.scenarios_path, "cig.cfg"))

game.set_doom_map("map01")  # Limited deathmatch.
# game.set_doom_map("map02")  # Full deathmatch.

# Host game with options that will be used in the competition.
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

game.add_game_args("+name Host +colorset 0")

game.set_mode(vzd.Mode.ASYNC_PLAYER)

game.init()

# Define key bindings for actions
key_mapping = {
    "w": [1, 0, 0, 0, 0, 0, 0, 0, 0],  # Move forward
    "s": [0, 1, 0, 0, 0, 0, 0, 0, 0],  # Move backward
    "a": [0, 0, 1, 0, 0, 0, 0, 0, 0],  # Turn left
    "d": [0, 0, 0, 1, 0, 0, 0, 0, 0],  # Turn right
    "space": [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Attack
    "shift": [0, 0, 0, 0, 0, 1, 0, 0, 0],  # Use
}

# Play until the game (episode) is over.
while not game.is_episode_finished():
    if game.is_player_dead():
        game.respawn_player()

    # Get the state.
    s = game.get_state()

    # Wait for user input and map it to actions
    action = [0] * 9  # Default action is no-op
    for key, mapped_action in key_mapping.items():
        if keyboard.is_pressed(key):
            action = mapped_action
            break

    # Perform the action
    game.make_action(action)

game.close()

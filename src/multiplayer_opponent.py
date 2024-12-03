#!/usr/bin/env python3

import os
import vizdoom as vzd

# Initialize the Doom game
game = vzd.DoomGame()

# Use CIG example config or your own.
game.load_config(os.path.join(vzd.scenarios_path, "cig.cfg"))
# game.set_doom_map("map02")  # Full deathmatch.

# Host game with options that will be used in the competition.
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
game.set_mode(vzd.Mode.ASYNC_PLAYER)
allowed_actions = [
    vzd.Button.ATTACK,
    vzd.Button.MOVE_RIGHT,
    vzd.Button.MOVE_LEFT,
    vzd.Button.MOVE_FORWARD,
    vzd.Button.MOVE_BACKWARD,
    vzd.Button.TURN_LEFT,
    vzd.Button.TURN_RIGHT,
    vzd.Button.RELOAD,
]
game.set_available_buttons(allowed_actions)
game.init()


# Play until the game (episode) is over.
while not game.is_episode_finished():
    if game.is_player_dead():
        game.respawn_player()

    # Get the state.
    s = game.get_state()

    # Wait for user input and map it to actions
    action = [0] * 9  # Default action is no-op
    action[0] = 1

    # Perform the action
    game.make_action(action)

game.close()


import os
from random import choice
import vizdoom as vzd

# Initialize the Doom game
game = vzd.DoomGame()

# Use the defend_the_center config and WAD
game.load_config("defend_the_center.cfg")
game.set_doom_scenario_path("defend_the_center.wad")

# Multiplayer setup
game.add_game_args(
    "-host 2 "  # Host for 2 players
    "-port 5029 "  # Communication port
    "+viz_connect_timeout 60 "  # Timeout for connecting players
    "+timelimit 5.0 "  # End after 5 minutes
    "+sv_forcerespawn 1 "  # Enable automatic respawn
    "+sv_spawnfarthest 1 "  # Spawn players far apart
    "+viz_respawn_delay 10 "  # Delay between respawns
    "+viz_nocheat 1"  # Disable cheats
)

# Name the agent and set color
game.add_game_args("+name Host +colorset 0")

# Set game mode for multiplayer
game.set_mode(vzd.Mode.ASYNC_PLAYER)
game.init()

# Example actions
actions = [
    [1, 0, 0],  # Attack
    [0, 1, 0],  # Move Right
    [0, 0, 1],  # Move Left
]

# Main loop
while not game.is_episode_finished():
    state = game.get_state()
    if state:
        # Perform random action for demonstration
        game.make_action(choice(actions))

    if game.is_player_dead():
        game.respawn_player()

game.close()

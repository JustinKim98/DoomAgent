#!/usr/bin/env python3

import os
import vizdoom as vzd
import numpy as np
import torch
from doomenv import multiplayer_env
from models import model
from stable_baselines3 import PPO


class MultiplayerAgent:
    def __init__(self, model_path, frame_buffer_size=3, game=None):
        self.model_path = model_path
        self.allowed_buttons = [
            vzd.Button.ATTACK,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_UP,
            vzd.Button.MOVE_DOWN,
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT,
            vzd.Button.LOOK_UP,
            vzd.Button.LOOK_DOWN,
            vzd.Button.RELOAD,
        ]
        self.game = game

        # policy_kwargs = dict(
        #     features_extractor_class=model.PolicyModel,
        #     features_extractor_kwargs=dict(features_dim=1024),
        #     net_arch=dict(
        #         activation_fn=torch.nn.Tanh,
        #         net_arch=dict(
        #             vf=[1024, 512, 512, 256, 256, 256, 128, 64],
        #             pi=[1024, 512, 512, 256, 256, 256, 128, 64],
        #         ),
        #     ),
        # )
        self.env = multiplayer_env.BaseEnv(
            "multi.cfg",
            self.allowed_buttons,
            frame_buffer_size,
            game=self.game,
            configure_as_host=True,
        )
        self.model = PPO.load(self.model_path, env=self.env, device="cpu")
        self.is_done = False
        self.action = 0

    def step(self):
        if self.is_done:
            return (0, self.is_done)
        state, reward, is_done, _ = self.env.step(self.action)
        self.action = self.model.predict(state)
        return (reward, is_done)

    def reset(self):
        self.env.reset()
        self.is_done = False
        self.action = 0


if __name__ == "__main__":
    # Initialize the Doom game
    game = vzd.DoomGame()
    game.set_window_visible(True)

    # Use CIG example config or your own.
    game.load_config(os.path.join(vzd.scenarios_path, "multi.cfg"))
    game.set_doom_map("map02")  # Full deathmatch.

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
    game.add_game_args("-join 127.0.0.1 -port 5029")

    game.set_mode(vzd.Mode.ASYNC_PLAYER)

    agent = MultiplayerAgent(
        "downloaded_models/multiplayer7/model_iter_17800.0",
        game=game,
        frame_buffer_size=3,
    )

    # Play until the game (episode) is over.
    while True:
        is_done = False
        while not is_done:
            reward, is_done = agent.step()
        agent.reset()

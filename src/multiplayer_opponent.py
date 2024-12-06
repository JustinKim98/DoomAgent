#!/usr/bin/env python3

import os
import vizdoom as vzd
import numpy as np
import torch
from doomenv import multiplayer_env
from models import model
from stable_baselines3 import PPO


class MultiplayerAgent:
    def __init__(self, model_path, frame_buffer_size=6, game=None):
        self.model_path = model_path
        self.allowed_buttons = [
            vzd.Button.ATTACK,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_FORWARD,
            vzd.Button.MOVE_BACKWARD,
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT,
            # vzd.Button.LOOK_UP,
            # vzd.Button.LOOK_DOWN,
            vzd.Button.RELOAD,
        ]
        self.game = game

        policy_kwargs = dict(
            features_extractor_class=model.PolicyModel,
            features_extractor_kwargs=dict(features_dim=1024),
            net_arch=dict(
                activation_fn=torch.nn.Tanh,
                net_arch=dict(
                    vf=[512, 512, 256, 256, 128, 64],
                    pi=[512, 512, 256, 256, 128, 64],
                ),
            ),
        )
        self.env = multiplayer_env.BaseEnv(
            "multi_duel.cfg", self.allowed_buttons, frame_buffer_size, game=self.game
        )
        self.model = PPO.load(
            self.model_path, env=self.env, custom_object=policy_kwargs, device="cpu"
        )
        self.is_done = False
        self.action = 0

    def step(self):
        if self.is_done:
            return (0, is_done)
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

    game.set_doom_map("map02")  # Full deathmatch.

    game.add_game_args("+name Host +colorset 0")
    game.add_game_args("-join 127.0.0.1 -port 5029")

    game.set_mode(vzd.Mode.ASYNC_PLAYER)

    agent = MultiplayerAgent("deathmatch_models8/model_iter_4000.0", game=game)

    # Play until the game (episode) is over.
    while True:
        is_done = False
        while not is_done:
            reward, is_done = agent.step()
        agent.reset()

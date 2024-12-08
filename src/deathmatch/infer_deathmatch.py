import vizdoom as vzd
import torch
from doomenv import env
from models import model
from stable_baselines3 import PPO


class DeathmatchAgent:
    def __init__(self, model_path, frame_buffer_size=6):
        self.model_path = model_path
        self.allowed_buttons = [
            vzd.Button.ATTACK,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_UP,
            vzd.Button.MOVE_DOWN,
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT,
            vzd.Button.RELOAD,
        ]
        self.game = vzd.DoomGame()

        policy_kwargs = dict(
            features_extractor_class=model.PolicyModel,
            features_extractor_kwargs=dict(features_dim=1024),
            net_arch=dict(
                activation_fn=torch.nn.Tanh,
                net_arch=dict(
                    vf=[1024, 512, 512, 256, 256, 256, 128, 64],
                    pi=[1024, 512, 512, 256, 256, 256, 128, 64],
                ),
            ),
        )
        self.env = env.BaseEnv(
            "deathmatch.cfg", self.allowed_buttons, frame_buffer_size, game=self.game
        )
        self.model = PPO.load(
            self.model_path, env=self.env, custom_object=policy_kwargs, device="auto"
        )
        self.is_done = False
        self.action = 0
        self.game.init()

    def step(self):
        state, reward, is_done, _ = self.env.step(self.action)
        self.action = self.model.predict(state)

        if is_done:
            return (0, is_done)

        return (reward, is_done)

    def reset(self):
        self.env.reset()
        self.is_done = False
        self.action = 0

    def close(self):
        self.env.close()


if __name__ == "__main__":
    agent = DeathmatchAgent("models/deathmatch/hard_deathmatch")

    for i in range(0, 10):
        print(f"episode : {i}")
        is_done = False
        while not is_done:
            reward, is_done = agent.step()
        agent.reset()

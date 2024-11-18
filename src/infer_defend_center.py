import vizdoom as vzd
from doomenv import env
from models import model
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


if __name__ == "__main__":
    allowed_actions = [
        vzd.Button.ATTACK,
        vzd.Button.TURN_LEFT,
        vzd.Button.TURN_RIGHT,
        vzd.Button.RELOAD,
    ]

    env = env.BaseEnv(
        "scenarios/defend_the_center.cfg", allowed_actions, frame_buffer_size=4
    )

    policy_kwargs = dict(
        features_extractor_class=model.PolicyModel,
        features_extractor_kwargs=dict(features_dim=512),
    )

    model = PPO.load(
        "downloaded_models/defend_center2/model_iter_57500.zip",
        env=env,
        custom_objects=policy_kwargs,
    )

    env.reset()
    is_done = False
    action = 0

    for i in range(0, 10):
        print(f"episode : {i}")
        while not is_done:
            state, reward, is_done, _ = env.step(action)
            action = model.predict(state)
        env.reset()
        is_done = False

import vizdoom as vzd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from defend_the_center_env import DoomDefendCenterEnv 

if __name__ == "__main__":
    # Load trained model
    print("Loading model...")
    model = PPO.load("ppo_defend_center_final", device="auto")

    # Initialize the environment and wrap in VecEnv
    env = DummyVecEnv([lambda: DoomDefendCenterEnv()])
    state = env.reset()
    done = False
    total_reward = 0

    print("Starting inference...")
    while not done:
        # Predict the action (ensure state is in correct format)
        action, _states = model.predict(state, deterministic=True)

        # Perform the action
        state, reward, done, info = env.step(action)
        total_reward += reward

        # Render the environment (optional)
        env.render()

    print(f"Total reward: {total_reward}")
    env.close()

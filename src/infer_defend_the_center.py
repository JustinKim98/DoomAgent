import vizdoom as vzd
from stable_baselines3 import PPO
from defend_the_center_env import DoomDefendCenterEnv
from stable_baselines3.common.vec_env import DummyVecEnv

class DefendCenterAgent:
    def __init__(self, model_path):
        self.model_path = model_path
        self.env = DummyVecEnv([lambda: DoomDefendCenterEnv()])
        self.model = PPO.load(self.model_path, env=self.env, device="auto")

    def step(self, state):
        action, _ = self.model.predict(state, deterministic=True)
        next_state, reward, done, info = self.env.step(action)
        return next_state, reward, done, info

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()


if __name__ == "__main__":
    # Path to the trained model
    model_path = "ppo_defend_center_final"
    agent = DefendCenterAgent(model_path)

    total_episodes = 10
    for episode in range(total_episodes):
        print(f"Starting episode {episode + 1}...")
        state = agent.reset()
        done = False
        total_reward = 0

        while not done:
            state, reward, done, info = agent.step(state)
            total_reward += reward
            agent.env.render() 

        print(f"Episode {episode + 1} finished with total reward: {total_reward}")

    agent.close()

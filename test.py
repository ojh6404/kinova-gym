# import gymnasium as gym
import gym
import panda_gym

# env = gym.make('kinova-v0', render_mode="human")
env = gym.make('kinova-v0', render_mode="human")

observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # random action
    observation, reward, terminated, truncated, info = env.step(action)
    # print(observation)

    if terminated or truncated:
        observation, info = env.reset()


env.close()

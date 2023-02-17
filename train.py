import gym
from stable_baselines3 import DDPG, HerReplayBuffer
from stable_baselines3 import PPO

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

import kinova_gym

env = gym.make("kinova-v0")
# env = gym.make("kinova-v0", render_mode="human")
# check_env(env)

# model = DDPG(policy="MultiInputPolicy", env=env,
#              replay_buffer_class=HerReplayBuffer, verbose=1)
model = PPO(policy="MlpPolicy", env=env, verbose=1)
model.learn(total_timesteps=1000000)

model.save('kinova')
del model

model = PPO.load('kinova', env=env)

mean_reward, std_reward = evaluate_policy(
    model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
# vec_env = model.get_env()
vec_env = gym.make("kinova-v0", render_mode="human")
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()

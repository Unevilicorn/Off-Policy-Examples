#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env


# In[2]:


class DiscretePendulum(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, n_bins=9):
        super(DiscretePendulum, self).__init__()
        self.env = gym.make('Pendulum-v1')
        
        self.action_space = gym.spaces.Discrete(n_bins)
        self.observation_space = self.env.observation_space

        self.nbins = n_bins

    def reset(self, seed=None):
        return self.env.reset(seed=seed)

    def step(self, action):
        # Map action to continuous action space
        action = (4 * action / (self.nbins - 1)) - 2
        return self.env.step([action])

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()
    
check_env(DiscretePendulum())


# In[3]:


policy_kwargs = dict(activation_fn=nn.LeakyReLU, net_arch=[32, 32, 32])

n_training = 200
n_total = 200
ep_size = 200
batch_size = 1

env = DiscretePendulum()

model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, gradient_steps=-1, batch_size=batch_size, max_grad_norm=1)
model.learn(total_timesteps=n_training * ep_size)


import wandb

wandb.init(
    project="dqn-pendulum",
    config={
        "type": "StableBaselines3 DQN",
        "batch_size": batch_size,
        "buffer_type" : "Built-in",

    }
)


obs, info = env.reset()
i = 0
rewards = 0
all_rewards = []
while i < n_total:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    rewards += reward
    if terminated or truncated:
        obs, info = env.reset()
        print(f"Episode {i} finished, with reward {rewards}", flush=True)
        wandb.log({"reward": rewards})
        all_rewards.append(rewards)
        rewards = 0
        i += 1
wandb.finish()


# In[4]:


import matplotlib.pyplot as plt
# plot all rewards as it is and also the average of the last 10 rewards
plt.plot(all_rewards)
plt.plot([sum(all_rewards[i:i+10])/10 for i in range(len(all_rewards)-10)])
plt.show()


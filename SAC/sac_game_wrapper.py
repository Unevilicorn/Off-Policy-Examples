import gymnasium as gym
import numpy as np

class Float32Wrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self):
        observation, info = self.env.reset()
        # convert observation to float32
        observation = observation.astype(np.float32)
        return observation, info

    def step(self, action):
        next_observation, reward, done, trunc, info = self.env.step(action)
        # convert next_observation to float32
        next_observation = next_observation.astype(np.float32)
        reward = reward.astype(np.float32)
        return next_observation, reward, done, trunc, info
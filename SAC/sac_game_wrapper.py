import gymnasium as gym
import numpy as np

class Float32Wrapper(gym.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)

    def reset(self, *args, **kwargs):
        observation, info = self.env.reset(*args, **kwargs)
        # convert observation to float32
        observation = observation.astype(np.float32)
        return observation, info

    def step(self, action):
        next_observation, reward, done, trunc, info = self.env.step(action)
        # convert next_observation to float32
        next_observation = next_observation.astype(np.float32)
        reward = reward.astype(np.float32).tolist()
        return next_observation, reward, done, trunc, info


def gym_pendulum():
    return Float32Wrapper(gym.make("Pendulum-v1"))

def gym_swimmer():
    return Float32Wrapper(gym.make("Swimmer-v4"))

def gym_halfcheetah():
    return Float32Wrapper(gym.make("HalfCheetah-v4"))
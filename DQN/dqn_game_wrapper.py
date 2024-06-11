import gymnasium as gym
import numpy as np

def divmod(x, y):
    return x // y, x % y

def remap_interval(x, f1, f2, t1, t2):
    # map x from [a, b] to [c, d]
    return t1 + (x - f1) * (t2 - t1) / (f2 - f1)

class DescreteToContinuousWrapper(gym.Wrapper):
    def __init__(self, env, action_remapping, maxAction=0) -> None:
        super().__init__(env)
        self.action_remapping = action_remapping
        self.action_space = gym.spaces.Discrete(maxAction)

    def reset(self, *args, **kwargs):
        observation, info = self.env.reset(*args, **kwargs)
        # convert observation to float32
        observation = observation.astype(np.float32)
        return observation, info
    
    def step(self, action):
        action_remapped = self.action_remapping(action)
        next_observation, reward, done, trunc, info = self.env.step(action_remapped)
        # convert next_observation to float32
        next_observation = next_observation.astype(np.float32)
        reward = reward.astype(np.float32).tolist()
        return next_observation, reward, done, trunc, info



def discrete_gym_pendulum(num_actions):    
    env = gym.make("Pendulum-v1")
    def action_remapping(action):
        return [remap_interval(action, 0, num_actions-1, -2, 2)]
    return DescreteToContinuousWrapper(env, action_remapping, maxAction=num_actions)


def discrete_gym_swimmer(num_actions):
    env = gym.make("Swimmer-v4")
    sqrt_actions = num_actions ** 0.5
    assert int(sqrt_actions) == sqrt_actions
    def action_remapping(action):
        a1, a2 = divmod(action, sqrt_actions)
        real_a1 = remap_interval(a1, 0, sqrt_actions-1, -1, 1)
        real_a2 = remap_interval(a2, 0, sqrt_actions-1, -1, 1)
        return [real_a1, real_a2]
    return DescreteToContinuousWrapper(env, action_remapping, maxAction=num_actions)

def discrete_gym_reacher(num_actions):
    env = gym.make("Reacher-v4")
    sqrt_actions = num_actions ** 0.5
    assert int(sqrt_actions) == sqrt_actions
    def action_remapping(action):
        a1, a2 = divmod(action, sqrt_actions)
        real_a1 = remap_interval(a1, 0, sqrt_actions-1, -1, 1)
        real_a2 = remap_interval(a2, 0, sqrt_actions-1, -1, 1)
        return [real_a1, real_a2]
    return DescreteToContinuousWrapper(env, action_remapping, maxAction=num_actions)

def discrete_gym_cheetah(num_actions):
    env = gym.make("HalfCheetah-v4")
    root = num_actions ** (1/6)
    assert int(root) == root
    def action_remapping(action):
        actions = []
        for i in range(6):
            action, a = divmod(action, root)
            real_a = remap_interval(a, 0, root-1, -1, 1)
            actions.append(real_a)
        return actions
    return DescreteToContinuousWrapper(env, action_remapping, maxAction=num_actions)
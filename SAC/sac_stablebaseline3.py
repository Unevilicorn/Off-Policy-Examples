# hack to import from parent directory
import cProfile
import sys
import os.path
import time

# get the current path
path = os.path.dirname(os.path.abspath(__name__))
# add the directory to the path
sys.path.insert(0, path)


from sac_helper import sac_cli, plot_and_save_average_plots
from sac_env_config import env_to_configs
from maybe_wandb import get_wandb


import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
import os

import gymnasium as gym


def logging_callback_creator(reward_arr, loss_arr, wdb):
    class CustomCallback(BaseCallback):
        """
        A custom callback that derives from ``BaseCallback``.

        :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        """
        def __init__(self, verbose: int = 0):
            super().__init__(verbose)
            self.reward_this_episode = 0
            self.ep_num = 1

        def _on_step(self) -> bool:
            """
            This method will be called by the model after each call to `env.step()`.

            For child callback (of an `EventCallback`), this will be called
            when the event is triggered.

            :return: If the callback returns False, training is aborted early.
            """
            self.reward_this_episode += self.locals['rewards']

            # check if truncated or done
            if self.locals['infos'][0]['TimeLimit.truncated']:
                reward = self.reward_this_episode[0]
                loss = self.model.logger.name_to_value['train/loss']
                print(f"Ep: {self.ep_num}, Reward: {reward}")
                wdb.log({"reward": reward})
                reward_arr.append(reward)
                loss_arr.append(loss)   
                self.reward_this_episode = 0
                self.ep_num += 1
            # print(f"Reward this episode: {self.reward_this_episode}")

            return True
        
    return CustomCallback()


def main(env_to_run, save_path, use_wandb=False):
    config = env_to_configs[env_to_run]

    env = config.env_creator()

    policy_kwargs = dict(activation_fn=nn.LeakyReLU, net_arch=config.hidden_layers)

    n_steps = config.num_steps
    ep_size = config.num_episodes
    batch_size = config.batch_size
    tau = config.tau
    gamma = config.gamma
    buffer_size = config.memory_capacity
    learning_rate = config.learning_rate

    model = SAC("MlpPolicy", 
                env, 
                policy_kwargs=policy_kwargs, 
                gradient_steps=1, 
                learning_starts=0,
                batch_size=batch_size, 
                gamma=gamma,
                tau=tau,
                buffer_size=buffer_size, 
                learning_rate=learning_rate, 
                verbose=0,
                train_freq=(1, "step"),
                ent_coef=f"auto_{config.alpha}",
            )


    machine = os.uname().nodename

    wandb_config = {
        "type": "Stable Baselines 3",
        "buffer_type": config.replay_type,
        "enviroment": env_to_run,
        "num_steps": config.num_steps,
        "num_episodes": config.num_episodes,
        "batch_size": config.batch_size,
        "machine": machine,
    }

    wandb = get_wandb(get_real=use_wandb)

    rewards = []
    losses = []

    cb = logging_callback_creator(rewards, losses, wandb)

    wandb.init(
        project="SAC Comparison",
        config=wandb_config,
    )


    start = time.time()

    model.learn(total_timesteps=ep_size * n_steps, log_interval=1, callback=cb)

    end = time.time()
    duration = end - start
    wandb.log({"execution_time": duration})
    wandb.finish()
    print(f"Finished training in {duration} seconds")

    plot_and_save_average_plots(rewards, save_path)

if __name__=="__main__":
    sac_cli(main, path_to_save="./results/sac/stable_baseline")

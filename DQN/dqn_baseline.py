# hack to import from parent directory
import sys
import os.path
# get the current path
path = os.path.dirname(os.path.abspath(__name__))
# add the directory to the path
sys.path.insert(0, path)


from dqn_env_config import env_to_configs
from maybe_wandb import get_wandb
from dqn_helpers import dqn_cli, plot_and_save_average_plots

import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import os



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
                wdb.log({"loss_mean_ep": loss})
                reward_arr.append(reward)
                loss_arr.append(loss)   
                self.reward_this_episode = 0
                self.ep_num += 1
            # print(f"Reward this episode: {self.reward_this_episode}")

            return True
        
    return CustomCallback()
    

def main(env_to_run, save_path, use_wandb=False):
    chosen_env = env_to_run
    config = env_to_configs[chosen_env]

    env = config.env_creator(config.action_space)

    policy_kwargs = dict(activation_fn=nn.LeakyReLU, net_arch=config.hidden_layers)

    n_steps = config.max_steps
    ep_size = config.num_episodes
    batch_size = config.batch_size
    tau = config.tau
    gamma = config.gamma
    target_update = config.target_update
    gradient_clip = config.gradient_clip
    buffer_size = config.memory_capacity
    epsilon_init = config.epsilon_init
    epsilon_min = config.epsilon_min
    epsilon_frac = config.epsilon_frac
    learning_rate = config.learning_rate

    model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, gradient_steps=-1, learning_starts=0,
                batch_size=batch_size, max_grad_norm=gradient_clip, gamma=gamma,
                target_update_interval=target_update, tau=tau,
                exploration_initial_eps=epsilon_init, exploration_final_eps=epsilon_min, exploration_fraction=epsilon_frac,
                buffer_size=buffer_size, learning_rate=learning_rate, verbose=0)


    machine = os.uname().nodename

    wandb_config = {
        "type": "Stable Baselines 3",
        "buffer_type": "Default" if not config.use_reverb else "Reverb",
        "enviroment": chosen_env,
        "num_steps": config.max_steps,
        "num_episodes": config.num_episodes,
        "batch_size": config.batch_size,
        "action_space": config.action_space,
        "machine": machine,
    }

    wandb = get_wandb(get_real=use_wandb)

    wandb.init(
        project="DQN Comparison",
        config=wandb_config,
    )

    rewards = []
    losses = []

    cb = logging_callback_creator(rewards, losses, wandb)
    model.learn(total_timesteps=ep_size * n_steps, log_interval=1, callback=cb)

    wandb.finish()

    plot_and_save_average_plots(rewards, save_path, window=5)

if __name__ == "__main__":
    dqn_cli(main, path_to_save="./results/dqn/stable_baseline")
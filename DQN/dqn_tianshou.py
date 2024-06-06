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

import tianshou as ts
from tianshou.utils import BaseLogger

import torch

class DQN(torch.nn.Module):
    def __init__(self, layer_sizes) -> None:
        super().__init__()
        self.model = torch.nn.Sequential()
        for i in range(len(layer_sizes)-1):
            self.model.add_module(f'layer{i}', torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                self.model.add_module(f'relu{i}', torch.nn.LeakyReLU())

    def forward(self, obs, state=None, info={})->torch.Tensor:
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


def get_custom_logger(wdb, log_interval, rewards):
    class CustomLogger(BaseLogger):
        def write(self, step_type, step, data):
            # print(step_type, step, data)
            # print("train/reward", data['train/reward'])
            rewards.append(data['train/reward'])
            wdb.log({"reward": data['train/reward']})

        def save_data(self, epoch, env_step, gradient_step, save_checkpoint_fn) -> None:
            pass
            # raise NotImplementedError

        def restore_data(self):
            pass
            # raise NotImplementedError
    inf = float('inf')
    return CustomLogger(train_interval=log_interval, test_interval=inf, update_interval=inf)



def main(env_to_run, save_path, use_wandb=False):

    chosen_env = env_to_run
    config = env_to_configs[chosen_env]

    env = config.env_creator(config.action_space)


    n_steps = config.max_steps
    ep_size = config.num_episodes
    batch_size = config.batch_size
    tau = config.tau
    gamma = config.gamma
    target_update = config.target_update
    gradient_clip = True if config.gradient_clip is not None else False
    buffer_size = config.memory_capacity
    epsilon_init = config.epsilon_init
    epsilon_min = config.epsilon_min
    epsilon_frac = config.epsilon_frac
    epsilon_decay = (epsilon_init - epsilon_min) / (epsilon_frac * ep_size)
    learning_rate = config.learning_rate

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = config.action_space or config.action_space.n
    net = DQN(state_shape + tuple(config.hidden_layers) + (action_shape,))
    optim = torch.optim.Adam(net.parameters(), lr=learning_rate)

    policy = ts.policy.DQNPolicy(
        model=net,
        optim=optim,
        action_space=env.action_space,
        discount_factor=gamma,
        target_update_freq=target_update,
        is_double=False,
        clip_loss_grad=gradient_clip
    )

    collector = ts.data.Collector(policy, env, ts.data.ReplayBuffer(buffer_size))

    machine = os.uname().nodename

    wandb_config = {
        "type": "TianShou",
        "buffer_type": "Default",
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

    logger = get_custom_logger(wandb, config.max_steps, rewards)

    result = ts.trainer.offpolicy_trainer(policy=policy,
                                        train_collector=collector,
                                        test_collector=None,
                                        max_epoch=ep_size,
                                        step_per_epoch=n_steps,
                                        step_per_collect=1,
                                        update_per_step=1,
                                        episode_per_test=ep_size,
                                        batch_size=batch_size,
                                        train_fn=lambda epoch, env_step: policy.set_eps(epsilon_init - epsilon_decay * epoch,),
                                        logger=logger,
    )

    wandb.log({"execution_time": result["duration"]})
    wandb.finish()
    print(f'Finished training in {result["duration"]}')

    plot_and_save_average_plots(rewards, save_path)


if __name__ == "__main__":
    dqn_cli(main, path_to_save="./results/dqn/tianshou")
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

import tianshou as ts
from tianshou.data import Collector
from tianshou.utils import BaseLogger
from tianshou.policy import SACPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic


import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def get_custom_logger(wdb, log_interval, rewards):
    class CustomLogger(BaseLogger):
        def write(self, step_type, step, data):
            # print(step_type, step, data)
            # print("train/reward", data['train/reward'])
            rewards.append(data['train/reward'])
            wdb.log({"reward": data['train/reward']})
            print(f"Step: {step}, Reward: {data['train/reward']}")

        def save_data(self, epoch, env_step, gradient_step, save_checkpoint_fn) -> None:
            pass
            # raise NotImplementedError

        def restore_data(self):
            pass
            # raise NotImplementedError
    inf = float('inf')
    return CustomLogger(train_interval=log_interval, test_interval=inf, update_interval=inf)



def main(chosen_env, path_to_save, use_wandb=False):
    chosen_env = "gym-pendulum"
    config = env_to_configs[chosen_env]

    env = config.env_creator()
    # https://github.com/thu-ml/tianshou/blob/master/examples/box2d/bipedal_hardcore_sac.py

    net_a = Net(
        state_shape=env.observation_space.shape, 
        hidden_sizes=config.hidden_layers, 
        device=DEVICE
    )

    actor = ActorProb(
        preprocess_net=net_a,
        action_shape=env.action_space.shape,
        device=DEVICE,
        unbounded=True,
    ).to(DEVICE)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=config.learning_rate)

    net_c1 = Net(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape, 
        hidden_sizes=config.hidden_layers, 
        concat=True,
        device=DEVICE
    )
    critic1 = Critic(preprocess_net=net_c1, device=DEVICE).to(DEVICE)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=config.learning_rate)

    net_c2 = Net(
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape, 
        hidden_sizes=config.hidden_layers, 
        concat=True,
        device=DEVICE
    )

    critic2 = Critic(preprocess_net=net_c2, device=DEVICE).to(DEVICE)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=config.learning_rate)

    action_dim = env.action_space.shape[0]

    target_entropy = -action_dim
    log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
    alpha_optim = torch.optim.Adam([log_alpha], lr=config.learning_rate)


    policy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=config.tau,
        gamma=config.gamma,
        alpha= (target_entropy, log_alpha, alpha_optim),
        estimation_step=1,
        action_space=env.action_space
    )

    train_coller = Collector(
        policy,
        env,
        ts.data.ReplayBuffer(size=config.memory_capacity),
        exploration_noise=True
    )

    rewards = []

    machine = os.uname().nodename

    wandb_config = {
        "type": "TianShou",
        "buffer_type": config.replay_type,
        "enviroment": chosen_env,
        "num_steps": config.num_steps,
        "num_episodes": config.num_episodes,
        "batch_size": config.batch_size,
        "machine": machine,
    }

    wandb = get_wandb(get_real=use_wandb)

    logger = get_custom_logger(wandb, config.num_steps, rewards)

    wandb.init(
        project="SAC Comparison",
        config=wandb_config,
    )

    result = ts.trainer.offpolicy_trainer(
        policy=policy,
        train_collector=train_coller,
        test_collector=None,
        max_epoch=config.num_episodes * config.num_steps,
        step_per_epoch=1,
        step_per_collect=1,
        update_per_step=1,
        episode_per_test=config.num_episodes,
        batch_size=config.batch_size,
        logger=logger,
        verbose=False,
        show_progress=False,
    )

    duration = result["duration"][:-1] # remove s 
    duration = float(duration)
    wandb.log({"execution_time": duration})
    wandb.finish()

    print(f"Execution time: {duration} seconds")

    plot_and_save_average_plots(rewards, path_to_save)


if __name__ == "__main__":
    sac_cli(main, path_to_save="./results/sac/tianshou")

# hack to import from parent directory
import cProfile
import sys
import os.path
# get the current path
path = os.path.dirname(os.path.abspath(__name__))
# add the directory to the path
sys.path.insert(0, path)


from dqn_helpers import dqn_cli, plot_and_save_average_plots
from dqn_env_config import env_to_configs
from maybe_wandb import get_wandb
from replay_memory import ReplayMemory

import torch
import random
import os
import time


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEVICE)
print(f"Using device: {DEVICE}")


class DQN(torch.nn.Module):
    def __init__(self, layer_sizes) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential()
        for i in range(len(layer_sizes)-1):
            self.layers.add_module(f'layer{i}', torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes)-2:
                self.layers.add_module(f'relu{i}', torch.nn.LeakyReLU())

    def forward(self, x)->torch.Tensor:
        return self.layers(x)


def epsilon_greedy_action(epsilon, action_rewards, device)->torch.Tensor:
    if random.random() < epsilon:
        return torch.randint(0, len(action_rewards), (1,), device=device)
    else:
        return torch.argmax(action_rewards).view((1,)).to(device)


def loss_func(policy_model, target_model, states, actions, rewards, next_states, dones, gamma): 
    actions = actions.view(-1, 1)
    rewards = rewards.view(-1, 1)
    dones = dones.view(-1, 1)

    q_values = policy_model.forward(states).gather(1, actions)
    next_q_values = target_model.forward(next_states).max(dim=1, keepdim=True)[0]
    expected_q_values = rewards + gamma * next_q_values * (1 - dones)
    loss = torch.nn.functional.mse_loss(q_values, expected_q_values)
    return loss


def main(env_to_run, save_path, use_wandb=False):
    chosen_env = env_to_run
    config = env_to_configs[chosen_env]

    env = config.env_creator(config.action_space)

    tau = config.tau
    gamma = config.gamma
    epsilon = config.epsilon_init
    epsilon_min = config.epsilon_min
    epsilon_steps = config.epsilon_frac * config.num_episodes * config.max_steps
    epsilon_delta = (epsilon - epsilon_min) / epsilon_steps
    batch_size = config.batch_size
    memory_size = config.memory_capacity
    num_episodes = config.num_episodes
    target_update_rate = config.target_update


    input_shape = env.observation_space.shape
    output_shape = config.action_space
    hideen_layers = config.hidden_layers

    gradient_clipping_value = config.gradient_clip

    model_shape = [*input_shape, *hideen_layers, output_shape]


    # memory = NumpyMemory(memory_size, input_shape, 1)
    memory = ReplayMemory(memory_size)
    # memory = ReverbMemory(memory_size)
    # memory = ReplayMemorySlow(memory_size)
    policy_model = DQN(model_shape)
    target_model = DQN(model_shape)
    target_model.load_state_dict(policy_model.state_dict())
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=config.learning_rate)

    machine = os.uname().nodename

    wandb_config = {
        "type": "Self-Implemented DQN",
        "buffer_type": config.replay_type,
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

    start_time = time.time()

    episode_steps = []
    rewards_tracker = []
    for i in range(num_episodes):
        state, _ = env.reset()
        steps = 0
        rewards_tracker.append([])
        while True:
            steps+=1
            qs = policy_model.forward(torch.from_numpy(state).to(DEVICE))
            action = epsilon_greedy_action(epsilon, qs, device=DEVICE)
            epsilon = max(epsilon_min, epsilon - epsilon_delta)
            next_observation, reward, done, trunc, info = env.step(action.item())

            next_observation = next_observation.reshape(-1)
            memory.append(state, action.item(), reward, next_observation, int(done))
            state = next_observation
            
            rewards_tracker[-1].append(reward)
            
            # Learning

            batch = memory.sample(batch_size, device=DEVICE)
            states, actions, rewards, next_states, dones = batch
            loss = loss_func(policy_model, target_model, states, actions, rewards, next_states, dones, gamma)
            
            policy_model.zero_grad()
            loss.backward()

            if gradient_clipping_value is not None:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), gradient_clipping_value)
            optimizer.step()
            
            if steps % target_update_rate == 0:
                for target_param, policy_param in zip(target_model.parameters(), policy_model.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - tau) + policy_param.data * tau)
            
            if done or trunc:
                break

        episode_steps.append(steps)
        print(f"Episode {i} finished with reward {sum(rewards_tracker[-1])}")
        wandb.log({'reward': sum(rewards_tracker[-1])})        

    execution_time_end = time.time()
    wandb.log({'execution_time': execution_time_end - start_time})
    wandb.finish()
    print(f"Finished training in {execution_time_end - start_time} seconds")
    rewards_sums = [sum(rewards) for rewards in rewards_tracker]

    plot_and_save_average_plots(rewards_sums, save_path)


if __name__ == "__main__":
    # with cProfile.Profile() as pr:
    dqn_cli(main, path_to_save="./results/dqn/self_implemented")
        # pr.dump_stats('./dqn.prof')
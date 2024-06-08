# hack to import from parent directory
import sys
import os.path
# get the current path
path = os.path.dirname(os.path.abspath(__name__))
# add the directory to the path
sys.path.insert(0, path)

from sac_helper import sac_cli, plot_and_save_average_plots
from sac_env_config import SacEnvConfigs, env_to_configs

from maybe_wandb import get_wandb
from replay_memory import ReplayMemory

import os
import time
import numpy as np
import torch


import torch
import numpy as np
import numpy as np


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class Critic(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers1, self.layers2 = self._build_network(dims)

    def _build_network(self, dims):
        layers1 = []
        layers2 = []
        for i in range(len(dims)-1):
            layers1.append(torch.nn.Linear(dims[i], dims[i+1]))
            layers2.append(torch.nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers1.append(torch.nn.LeakyReLU())
                layers2.append(torch.nn.LeakyReLU())
        return torch.nn.Sequential(*layers1), torch.nn.Sequential(*layers2)
    
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        return self.layers1(x), self.layers2(x)


class Actor(torch.nn.Module):
    def __init__(self, dims, high=None, low=None):
        super().__init__()
        self.layers = torch.nn.Sequential()
        
        for i in range(len(dims)-2):
            self.layers.append(torch.nn.Linear(dims[i], dims[i+1]))
            if i != len(dims)-2:
                self.layers.append(torch.nn.LeakyReLU())

        self.mu_layer = torch.nn.Linear(dims[-2], dims[-1])
        self.sigma_layer = torch.nn.Linear(dims[-2], dims[-1])
        
        if high is None or low is None:
            self.action_scale = torch.tensor(1.0, device=DEVICE)
            self.action_bias = torch.tensor(0.0, device=DEVICE)
        else:
            self.action_scale = torch.tensor((high - low) / 2, device=DEVICE)
            self.action_bias = torch.tensor((high + low) / 2, device=DEVICE)
            
    def sample(self, x):
        mus, sigmas = self.forward(x)
        sigmas = torch.exp(sigmas)
        action = torch.distributions.Normal(mus, sigmas).sample()
        action = torch.tanh(action) * self.action_scale + self.action_bias
        return action

    def sample_with_log_prob(self, x):
        mus, sigmas = self.forward(x)
        sigmas = torch.exp(sigmas)
        probs = torch.distributions.Normal(mus, sigmas)
        action = probs.rsample()

        # Adjusting for the tanh squashing function
        log_prob = probs.log_prob(action)
        log_prob -= (2 * (np.log(2) - action - torch.nn.functional.softplus(-2 * action)))
        log_prob = log_prob.sum(1, keepdim=True)

        action = torch.tanh(action) * self.action_scale + self.action_bias

        return action, log_prob
    
    def deterministic_action(self, x):
        mus, sigmas = self.forward(x)
        action = torch.tanh(mus) * self.action_scale + self.action_bias
        return action
           
    def forward(self, x):
        mid = self.layers(x)
        mus = self.mu_layer(mid)
        sigmas = self.sigma_layer(mid)
        sigmas = torch.clamp(sigmas, min=-20, max=2)
        return mus, sigmas


class SAC:
    def __init__(self, env, config: SacEnvConfigs):
        
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        policy_hidden_dim = config.hidden_layers
        q_hidden_dim = config.hidden_layers
        
        self.policy_net_dims = [input_size, *policy_hidden_dim, output_size]
        self.critic_net_dims = [input_size + output_size, *q_hidden_dim, 1]
                
        self.gamma = config.gamma
        self.batch_size = config.batch_size
        self.steps_to_wait = 0
        self.polyak = config.tau

        self.log_alpha = torch.nn.Parameter(torch.tensor(np.log(0.2), device=DEVICE))
        self.target_entropy = -torch.prod(torch.tensor(env.action_space.shape, device=DEVICE))
        
        self.env = env
        self.replay_buffer = ReplayMemory(config.memory_capacity)
        
        self.policy = Actor(self.policy_net_dims, env.action_space.high, env.action_space.low)
        self.critic = Critic(self.critic_net_dims)
        self.critic_t = Critic(self.critic_net_dims)
        self.policy.cuda()
        self.critic.cuda()
        self.critic_t.cuda()

        self.critic_t.load_state_dict(self.critic.state_dict())
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.learning_rate)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.learning_rate)
        
        
    def train(self, episodes, wandb):
        state, info = self.env.reset()
        r = 0
        scores = []
        losses = []
        for episode in range(episodes):
            state, _ = self.env.reset()
            scores.append([])
            losses
            while True:
                # gather experience
                action = self.policy.sample(torch.as_tensor(state, device=DEVICE))
                cpu_action = action.detach().cpu().numpy()
                next_state, reward, done, trunc, _ = self.env.step(cpu_action)
                scores[-1].append(reward)
                done = done or trunc
                self.replay_buffer.append(state, cpu_action, reward, next_state, int(done))
                
                state = next_state
                
                # sample from buffer
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, device=DEVICE)
                # actions = actions.view(-1, 1)
                rewards = rewards.view(-1, 1)
                dones = dones.view(-1, 1)


                # update critic
                with torch.no_grad():
                    next_actions, next_log_prob = self.policy.sample_with_log_prob(next_states)

                    q1tv, q2tv = self.critic_t(next_states, next_actions)
                    target = rewards + self.gamma * (~dones) * torch.min(q1tv, q2tv) - self.log_alpha.exp() * next_log_prob
                
                q1v, q2v = self.critic(states, actions)
                q1_loss = torch.nn.functional.mse_loss(q1v, target)
                q2_loss = torch.nn.functional.mse_loss(q2v, target)
                
                critic_loss = q1_loss + q2_loss
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                
                # update policy
                now_actions, now_log_prob = self.policy.sample_with_log_prob(next_states)
                with torch.no_grad():
                    q1v, q2v = self.critic(states, now_actions)
                
                policy_loss = torch.mean(self.log_alpha.exp() * now_log_prob - torch.min(q1v, q2v))
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
                
                # update alpha
                alpha_loss = torch.mean(self.log_alpha.exp() * (-self.target_entropy - now_log_prob).detach())
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                

                losses.append([q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item(), self.log_alpha.exp().item()])
                
                # update target networks using polyak averaging
                for target_param, param in zip(self.critic_t.parameters(), self.critic.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.polyak) + param.data * self.polyak)
                        

                if done:
                    break
            episode_reward = sum(scores[-1])
            wandb.log({"reward": episode_reward})
                    
            print(f"Episode: {episode+1}, Reward: {episode_reward}")
        return scores, losses


def main(env_to_run, save_path, use_wandb=False):
    chosen_env = env_to_run
    config = env_to_configs[chosen_env]
    env = config.env_creator()
                
    sac = SAC(env, config)
    machine = os.uname().nodename

    wandb_config = {
        "type": "Self-Implemented DQN",
        "buffer_type": config.replay_type,
        "enviroment": chosen_env,
        "num_steps": config.num_steps,
        "num_episodes": config.num_episodes,
        "batch_size": config.batch_size,
        "machine": machine,
    }

    wandb = get_wandb(get_real=use_wandb)

    wandb.init(
        project="SAC Comparison",
        config=wandb_config,
    )
    start = time.time()
    scores, losses = sac.train(10, wandb=wandb)
    end = time.time()
    time_taken = end - start
    wandb.log({"execution_time": time_taken})

    print(f"Execution time: {time_taken}")
    
    scores = [sum(score) for score in scores]
    plot_and_save_average_plots(scores, save_path)


if __name__=="__main__":
    sac_cli(main, path_to_save="./results/sac/self_implemented")


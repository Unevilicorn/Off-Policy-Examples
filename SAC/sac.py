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
        self.layers = []
        
        for i in range(len(dims)-2):
            self.layers.append(torch.nn.Linear(dims[i], dims[i+1]))
            self.layers.append(torch.nn.LeakyReLU())

        self.layers = torch.nn.Sequential(*self.layers)

        self.mu_layer = torch.nn.Linear(dims[-2], dims[-1])
        self.sigma_layer = torch.nn.Linear(dims[-2], dims[-1])
        
        if high is None or low is None:
            self.action_scale = torch.tensor(1.0, device=DEVICE)
            self.action_bias = torch.tensor(0.0, device=DEVICE)
        else:
            self.action_scale = torch.tensor((high - low) / 2, device=DEVICE)
            self.action_bias = torch.tensor((high + low) / 2, device=DEVICE)
        
        self.eps = 1e-6

    def unnormalize(self, action):
        return action * self.action_scale + self.action_bias
            
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
        # reparametrization trick
        action = probs.rsample()
        action_tanh = torch.tanh(action)

        # Adjusting for the tanh squashing function
        log_prob = probs.log_prob(action)

        # sum over the action dimensions
        log_prob -= torch.log(1 - action_tanh ** 2 + self.eps)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action_tanh, log_prob
    
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

        self.log_alpha = torch.nn.Parameter(torch.tensor(np.log([config.alpha]), dtype=torch.float32, device=DEVICE))
        self.target_entropy = -torch.prod(torch.tensor(env.action_space.shape, dtype=torch.float32, device=DEVICE))
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
            losses.append([])
            while True:
                # gather experience
                action = self.policy.sample(torch.as_tensor(state, device=DEVICE))
                unscaled_action = self.policy.unnormalize(action).cpu().numpy()
                cpu_action = action.cpu().numpy()
                next_state, reward, done, trunc, _ = self.env.step(unscaled_action)

                scores[-1].append(reward)
                done = done or trunc

                self.replay_buffer.append(state, cpu_action, reward, next_state, int(done))
                
                state = next_state
                
                # sample from buffer
                r_states, r_actions, r_rewards, r_next_states, r_dones = self.replay_buffer.sample(self.batch_size, device=DEVICE)
                # actions = actions.view(-1, 1)
                r_rewards = r_rewards.reshape(-1, 1)
                r_dones = r_dones.reshape(-1, 1)

                now_actions, now_log_prob = self.policy.sample_with_log_prob(r_states)

                # update alpha
                alpha_detach = self.log_alpha.detach().exp()
                alpha_loss = -(self.log_alpha * (self.target_entropy + now_log_prob).detach()).mean()
                # print(f"Alpha: {self.log_alpha.exp().item()}")
                # print(f"Average log prob: {now_log_prob.mean().item()}")
                # print(f"Alpha loss: {alpha_loss.item()}")
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                # update critic
                with torch.no_grad():
                    next_actions, next_log_prob = self.policy.sample_with_log_prob(r_next_states)
                    q1tv, q2tv = self.critic_t(r_next_states, next_actions)
                    qtvmins = torch.min(torch.cat((q1tv, q2tv), dim=1), dim=1, keepdim=True)[0]
                    next_qvs = qtvmins - alpha_detach * next_log_prob
                    target = r_rewards + (1 - r_dones) * self.gamma * next_qvs
                
                q1v, q2v = self.critic(r_states, r_actions)
                q1_loss = torch.nn.functional.mse_loss(q1v, target)
                q2_loss = torch.nn.functional.mse_loss(q2v, target)
                
                critic_loss = (q1_loss + q2_loss) / 2
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                # update policy
                q1v, q2v = self.critic(r_states, now_actions)
                qvmins = torch.min(torch.cat((q1v, q2v), dim=1), dim=1, keepdim=True)[0]
                policy_loss = (alpha_detach * now_log_prob - qvmins).mean()
                
                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()
            
                losses[-1].append([q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item(), self.log_alpha.exp().item()])
                
                # update target networks using polyak averaging
                with torch.no_grad():
                    for target_param, param in zip(self.critic_t.parameters(), self.critic.parameters()):
                            target_param.data.mul_(1.0 - self.polyak)
                            torch.add(target_param, param, alpha=self.polyak, out=target_param)                        

                if done:
                    break
            episode_reward = sum(scores[-1])
            wandb.log({"reward": episode_reward})
                    
            print(f"Episode: {episode+1}, Reward: {episode_reward}")
            # print(f"Average critic loss: {np.mean([l[0] for l in losses[-1]])}")
            # print(f"Average policy loss: {np.mean([l[2] for l in losses[-1]])}")
            # print(f"Average alpha loss: {np.mean([l[3] for l in losses[-1]])}")

        return scores, losses


def main(env_to_run, save_path, use_wandb=False):
    chosen_env = env_to_run
    config = env_to_configs[chosen_env]
    env = config.env_creator()
                
    sac = SAC(env, config)
    machine = os.uname().nodename

    wandb_config = {
        "type": "Self-Implemented SAC",
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
        setting={
            "_stats_sample_rate_seconds": 1,
            "_stats_samples_to_average": 5,
        },
    )
    start = time.time()
    scores, losses = sac.train(config.num_episodes, wandb=wandb)
    end = time.time()
    time_taken = end - start
    wandb.log({"execution_time": time_taken})

    print(f"Execution time: {time_taken}")
    
    scores = [sum(score) for score in scores]
    plot_and_save_average_plots(scores, save_path)


if __name__=="__main__":
    sac_cli(main, path_to_save="./results/sac/self_implemented")


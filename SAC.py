# IMPORT LIBs

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import logging
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#%matplotlib inline
import os
import datetime
import math

from collections import deque
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet

from torchviz import make_dot


print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

#------------------------------------------------------
# ENVIRONMENT

def process_actions(action, input_requests):
    action_sum = np.sum(action)
    if action_sum == 0:
        action_sum += 1e-8
    action /= action_sum

    local = int(action[0] * input_requests)
    forwarded = int(action[1] * input_requests)
    rejected = int(action[2] * input_requests)
    local_fraction= (action[0] * input_requests) - local
    forwarded_fraction = (action[1] * input_requests) - forwarded
    rejected_fraction = (action[2] * input_requests) - rejected
    total_actions = local + forwarded + rejected

    if total_actions < input_requests:
        fractions = [local_fraction, forwarded_fraction, rejected_fraction]
        actions = [local, forwarded, rejected]
        max_fraction_index = np.argmax(fractions)
        actions[max_fraction_index] += input_requests - total_actions
        local, forwarded, rejected = actions

    return local, forwarded, rejected

def calculate_reward1(local, forwarded, rejected, QUEUE_factor, FORWARD_factor):
    reward_local = 3 * local * QUEUE_factor
    reward_forwarded = 1 * forwarded * (1 - QUEUE_factor) * FORWARD_factor
    reward_rejected = -5 * rejected * FORWARD_factor * QUEUE_factor
    reward = reward_local + reward_forwarded + reward_rejected

    return reward

class TrafficManagementEnv(gym.Env):
    def __init__(self, CPU_capacity = 50, queue_capacity = 100, queue_length = 0, forward_capacity = 100, average_requests = 100, amplitude_requests = 50, period=50):
        super().__init__()
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low = np.array([50, 0, 0]), high = np.array([150, 100, 100]), dtype = np.float32)

        self.max_CPU_capacity = CPU_capacity
        self.max_queue_capacity = queue_capacity
        self.queue_length = queue_length

        self.average_requests = average_requests
        self.amplitude_requests = amplitude_requests
        self.period = period
        self.t = 0

        self.max_forward_capacity = forward_capacity
        self.forward_capacity_t = self.max_forward_capacity

        self.input_requests = self.calculate_requests()

    def calculate_requests(self):
        return int(self.average_requests + self.amplitude_requests * math.sin(2 * math.pi * self.t / self.period))
    
    def reset(self):
        self.t = 0
        self.CPU_capacity = self.max_CPU_capacity
        self.queue_capacity = self.max_queue_capacity
        self.forward_capacity = self.max_forward_capacity
        self.forward_capacity_t = self.max_forward_capacity
        self.queue_length = 0

        return np.array([self.input_requests, self.queue_capacity, self.forward_capacity], dtype=np.float32)
    
    def step(self, action):
        print(f"INPUT: {self.input_requests}")
        print(f"CPU Capacity: {self.CPU_capacity}")
        print(f"Queue Capacity: {self.queue_capacity}")
        print(f"Forward Capacity: {self.forward_capacity}")
    
        self.local, self.forwarded, self.rejected = process_actions(action, self.input_requests)

        print(f"LOCAL: {self.local}")
        print(f"FORWARDED: {self.forwarded}")
        print(f"REJECTED: {self.rejected}")

        self.QUEUE_factor = self.queue_capacity / self.max_queue_capacity
        self.FORWARD_factor = self.forward_capacity / self.max_forward_capacity

        reward = calculate_reward1(self.local, self.forwarded, self.rejected, self.QUEUE_factor, self.FORWARD_factor)
        print(f"REWARD: {reward}")

        self.queue_length = max(0, self.local - self.CPU_capacity)
        self.CPU_capacity = max(-100, self.max_CPU_capacity - self.queue_length)
        self.queue_capacity = max(0, self.max_queue_capacity - self.queue_length)

        self.forward_capacity = int(25 + 75 * (1 + math.sin(2 * math.pi * self.t / self.period)) / 2)
        self.forward_capacity_t = self.forward_capacity

        
        self.t += 1

        if self.t == 100:
            done = True
        else:
            done = False
            
        self.input_requests = self.calculate_requests()
        state = np.array([self.input_requests, self.queue_capacity, self.forward_capacity], dtype=np.float32)
        
        return state, reward, done

#------------------------------------------------------
# SAC

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.exp(self.fc4(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.tanh(self.fc1(torch.cat([state, action], dim=-1)))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x
    
class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.next_state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done_memory = np.zeros(max_size, dtype=np.bool_)
        self.pointer = 0
        self.size = 0

    def store(self, state, action, reward, next_state, done):
        idx = self.pointer % self.max_size
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.next_state_memory[idx] = next_state
        self.done_memory[idx] = done
        self.pointer += 1
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.state_memory[idxs], self.action_memory[idxs], self.reward_memory[idxs],
                self.next_state_memory[idxs], self.done_memory[idxs])
    
class SAC:
    def __init__(self, state_dim, action_dim, device, lr=3e-4, gamma=0.9, tau=0.005, 
                 target_entropy = None):
        self.device = device
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.target_critic_1 = Critic(state_dim, action_dim).to(device)
        self.target_critic_2 = Critic(state_dim, action_dim).to(device)
        self.soft_update(tau=1)  # Copy the critic parameters initially

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)  
        self.alpha = self.log_alpha.exp()
        
        if target_entropy is None:
            target_entropy = -action_dim
        self.target_entropy = target_entropy

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=1e-3)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=lr, weight_decay=1e-3)

        self.gamma = gamma
        self.tau = tau

    def soft_update(self, tau=None):
        if tau is None:
            tau = self.tau
        for t_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            t_param.data.copy_((1 - tau) * t_param + tau * param)
        for t_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            t_param.data.copy_((1 - tau) * t_param + tau * param)

    def train(self, replay_buffer, batch_size):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.from_numpy(state).float().to(self.device)
        action = torch.from_numpy(action).float().to(self.device)
        reward = torch.from_numpy(reward).float().unsqueeze(1).to(self.device)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        done = torch.from_numpy(done).float().unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_action = self.actor(next_state)
            target_q1 = self.target_critic_1(next_state, next_action)
            target_q2 = self.target_critic_2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_value = reward + (1 - done) * self.gamma * target_q
#-------------------------------------------------------------------------------------
        # Critic losses
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        critic_1_loss = nn.functional.mse_loss(current_q1, target_value)
        critic_2_loss = nn.functional.mse_loss(current_q2, target_value)
        critic_loss = critic_1_loss + critic_2_loss

        # 1. Update critics
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
#-------------------------------------------------------------------------------------
        # Actor loss
        dist = Dirichlet(self.actor(state))
        new_action = dist.rsample()
        log_prob = dist.log_prob(new_action).view(-1,1) # dimensione [B,1] dove B=batch-size
        q1 = self.critic_1(state, new_action)
        q2 = self.critic_2(state, new_action)
        q = torch.min(q1, q2) # dimensione [B,1] dove B=batch-size
        actor_loss = (self.alpha.detach() * log_prob - q).mean()
        # alpha loss
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

        # 2. update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # 3. Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
# ---------------------------------------------------------------------------------------
        # Soft update target critics
        self.soft_update()

        return actor_loss.item(), critic_1_loss.item(), critic_2_loss.item(), alpha_loss.item()
#---------------------------------------------------------------------------------------
    def select_action(self, state):
        #with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dist = Dirichlet(self.actor(state))
        action = dist.sample().squeeze(0).cpu().numpy()
        return action
    
    def save_weights(self, path):
        self.actor.save_weights(path + '_actor.h5')
        self.critic_1.save_weights(path + '_critic1.h5')
        self.critic_2.save_weights(path + '_critic2.h5')

    def load_weights(self, path):
        self.actor.load_weights(path + '_actor.h5')
        self.critic_1.load_weights(path + '_critic1.h5')
        self.critic_2.load_weights(path + '_critic2.h5')


#---------------------------------------------------------------------------------------
# TRAINING 
 
def train_sac_agent(env, agent, buffer_size=1000000, batch_size=256, num_episodes=100, 
                    max_steps_per_episode=100, warm_up=512):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/SAC/' + current_time
    writer = SummaryWriter(train_log_dir)
    
    replay_buffer = ReplayBuffer(max_size=buffer_size, state_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0])
    total_rewards = []
    actor_losses = []
    critic1_losses = []
    critic2_losses = []
    alpha_losses = []
    steps = 0

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_actor_loss = 0
        episode_critic1_loss = 0
        episode_critic2_loss = 0
        episode_alpha_loss = 0

        for step in range(max_steps_per_episode):
            print("---------------------------------")
            print(f"Episode: {episode}, Step: {step}")
            print("---------------------------------")

            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            replay_buffer.store(state, action, reward, next_state, done)

            if replay_buffer.size > warm_up:
                actor_loss, critic_1_loss, critic_2_loss, alpha_loss = agent.train(replay_buffer, batch_size)
                episode_actor_loss += actor_loss
                episode_critic1_loss += critic_1_loss
                episode_critic2_loss += critic_2_loss
                episode_alpha_loss += alpha_loss

                # scriviamo le loss nel SummaryWriter
                writer.add_scalar('Loss/Actor', actor_loss, steps)
                writer.add_scalar('Loss/Critic1',critic_1_loss, steps)
                writer.add_scalar('Loss/Critic2',critic_2_loss, steps)
                writer.add_scalar('Loss/Alpha',alpha_loss, steps)

            state = next_state
            episode_reward += reward
            steps += 1

            if done:
                break
        
        writer.add_scalar('Reward', episode_reward, episode)

        total_rewards.append(episode_reward)
        actor_losses.append(episode_actor_loss / step)
        critic1_losses.append(episode_critic1_loss / step)
        critic2_losses.append(episode_critic2_loss / step)
        alpha_losses.append(episode_alpha_loss / step)

        print(f"Episode: {episode + 1}, Reward: {episode_reward}")

    writer.close() 
    #agent.save_weights("C:/Users/giaco/Desktop/local-git/Pesi reti")

    metrics = pd.DataFrame({'Reward': total_rewards, 'Actor Loss': actor_losses, 'Alpha Loss': alpha_losses,
                            'Critic 1 Loss': critic1_losses, 'Critic 2 Loss': critic2_losses})
    metrics.to_csv('metrics.csv')

    plt.figure(figsize=(12, 8))
    plt.plot(metrics['Critic 1 Loss'].rolling(10).mean(), label='Critic1 Loss')
    plt.plot(metrics['Critic 2 Loss'].rolling(10).mean(), label='Critic2 Loss')
    plt.title('Critic Networks Losses')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(metrics['Actor Loss'].rolling(10).mean(), label='Actor Loss')
    plt.title('Actor Loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(metrics['Alpha Loss'].rolling(10).mean(), label='Alpha Loss')
    plt.title('Alpha Loss')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.plot(metrics['Reward'].rolling(10).mean(), label='Reward')
    plt.title('Rewards')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.show()


if __name__ == "__main__":
    env = TrafficManagementEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    sac_agent = SAC(state_dim, action_dim, device)

    train_sac_agent(env, sac_agent)


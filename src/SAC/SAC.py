import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.dirichlet import Dirichlet

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ACTOR NETWORK
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

# CRITIC NETWORK
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

# SAC AGENT
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

        # Soft update target critics
        self.soft_update()

        return actor_loss.item(), critic_1_loss.item(), critic_2_loss.item(), alpha_loss.item()

    def select_action(self, state):
        #with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        dist = Dirichlet(self.actor(state))
        action = dist.sample().squeeze(0).cpu().numpy()
        return action
    
    def save_weights(self, path):
        torch.save(self.actor.state_dict(), path + '_actor.pth')
        torch.save(self.critic_1.state_dict(), path + '_critic1.pth')
        torch.save(self.critic_2.state_dict(), path + '_critic2.pth')

    def load_weights(self, path):
        self.actor.load_state_dict(torch.load(path + '_actor.pth'))
        self.critic_1.load_state_dict(torch.load(path + '_critic1.pth'))
        self.critic_2.load_state_dict(torch.load(path + '_critic2.pth'))
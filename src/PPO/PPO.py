import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.dirichlet import Dirichlet

print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

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
        #x = torch.softmax(self.fc4(x), dim=-1)
        return x

# CRITIC NETWORK
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        return x
    
class PPO:
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, ent_coef=0.01, max_grad_norm=0.0):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gamma = gamma                   # discount factor, how much to discount future rewards
        self.gae_lambda = gae_lambda         # parameter for Generalized Advantage Estimation
        self.clip_epsilon = clip_epsilon     # parameter for PPO, for clipping surrogate objective
        self.ent_coef = ent_coef             # parameter that controls the amount of exploration
        self.max_grad_norm = max_grad_norm   # parameter for gradient clipping, if 0 then no clipping

        self.actor.to(self.device)
        self.critic.to(self.device)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        action_probs = self.actor(state)
        dist = Dirichlet(action_probs)
        action = dist.rsample().squeeze(0).detach().cpu().numpy()
        return action

    def entropy_loss(self, entropy: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        ent_loss = -entropy
        reduction = reduction.lower()
        if reduction == "none":
            return ent_loss
        elif reduction == "mean":
            return ent_loss.mean()
        elif reduction == "sum":
            return ent_loss.sum()
        else:
            raise ValueError(f"Unrecognized reduction: {reduction}")

    def value_loss(self, values, returns):
        return ((values - returns)**2).mean()
    
    # Estimate returns with GAE (https://arxiv.org/abs/1506.02438)   
    def compute_gae(self, rewards, masks, values):
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            if t < len(rewards) - 1:
                delta = rewards[t] + self.gamma * values[t + 1] * masks[t] - values[t]
            else:
                delta = rewards[t] - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * masks[t] * last_gae
            advantages[t] = last_gae
        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, old_probs, rewards, masks, values, vf_coef=0.5, epochs=10, batch_size=256):
        # vf_coef: how much the critic loss should be weighted in the total loss. 
        #          If 1.0, then the critic loss is weighted the same as the actor loss.

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        old_probs = torch.FloatTensor(np.array(old_probs)).to(self.device)

        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        masks = torch.FloatTensor(np.array(masks)).to(self.device)
        values = torch.FloatTensor(np.array(values)).to(self.device)

        advantages, returns = self.compute_gae(rewards, masks, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        for _ in range(epochs):
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                mini_batch_states = states[start:end]
                mini_batch_actions = actions[start:end]
                mini_batch_old_probs = old_probs[start:end]
                mini_batch_advantages = advantages[start:end]
                mini_batch_returns = returns[start:end]

                action_probs = self.actor(mini_batch_states)
                dist_current = Dirichlet(action_probs)
                new_probs = dist_current.log_prob(mini_batch_actions)
                new_values = self.critic(mini_batch_states).squeeze(1)
                
                log_ratio = new_probs - mini_batch_old_probs
                ratio = log_ratio.exp()

                pg_loss1 = mini_batch_advantages * ratio
                pg_loss2 = mini_batch_advantages * torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                entropy = dist_current.entropy().mean()
                actor_loss = -torch.min(pg_loss1, pg_loss2).mean()

                critic_loss = self.value_loss(new_values, mini_batch_returns)
                ent_loss = self.entropy_loss(entropy, reduction="mean")

                loss = actor_loss + vf_coef * critic_loss + self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()

                if self.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)

                self.optimizer.step()
                
                return loss.item(), actor_loss.item(), critic_loss.item(), ent_loss.item()
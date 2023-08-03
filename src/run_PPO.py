import torch
from env.env import TrafficManagementEnv
from PPO.PPO import PPO
from training.training_PPO import train_ppo_agent

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TrafficManagementEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    ppo_agent = PPO(state_dim, action_dim)

    train_ppo_agent(env, ppo_agent)
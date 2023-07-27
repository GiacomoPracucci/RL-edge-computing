import torch
from env.env import TrafficManagementEnv
from SAC.SAC import SAC
from training.training import train_sac_agent

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TrafficManagementEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    sac_agent = SAC(state_dim, action_dim, device)

    train_sac_agent(env, sac_agent)
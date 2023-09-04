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
    
    # Carica i pesi da un checkpoint se desiderato
    #checkpoint_episode = 500  # Ad esempio, se si desidera caricare il checkpoint dopo 500 episodi
    #checkpoint_path = f"C:/Users/giaco/Desktop/local-git/PPO_weights/checkpoint_{checkpoint_episode}"
    #ppo_agent.load_weights_PPO(checkpoint_path)

    train_ppo_agent(env, ppo_agent)
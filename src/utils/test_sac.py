import torch
import sys
sys.path.append('C:/Users/giaco/Desktop/tesi_git/src')
from env.env import TrafficManagementEnv
from SAC.SAC import SAC
import matplotlib.pyplot as plt


# Creo l'istanza PPO
state_dim = 4  # Lo spazio delle osservazioni ha 4 dimensioni come definito in TrafficManagementEnv
action_dim = 3  # Lo spazio delle azioni ha 3 dimensioni come definito in TrafficManagementEnv
agent = SAC(state_dim, action_dim, device=torch.device("cpu"))

# Carica i pesi salvati
path_to_weights = "C:/Users/giaco/Desktop/local-git/SAC_weights/SAC_weights"  # Sostituisci con il percorso effettivo
agent.load_weights_SAC(path_to_weights)

env = TrafficManagementEnv()  # Puoi anche impostare altri parametri se necessario
num_episodes = 10  # Numero di episodi da eseguire per il test
all_episode_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        episode_reward += reward
        state = next_state
    print(f"Episodio {episode + 1}: Ricompensa Totale = {episode_reward}")
    all_episode_rewards.append(episode_reward)
    
# Plotting
plt.figure(figsize=(10,5))
plt.plot(all_episode_rewards, marker='o', linestyle='-')
plt.title('Ricompensa Totale per Episodio')
plt.xlabel('Episodi')
plt.ylabel('Ricompensa Totale')
plt.grid(True)
plt.show()


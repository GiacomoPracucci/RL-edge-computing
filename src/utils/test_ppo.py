import sys
sys.path.append('C:/Users/giaco/Desktop/tesi_git/src')
from env.env import TrafficManagementEnv
from PPO.PPO import PPO
import matplotlib.pyplot as plt
import csv

state_dim = 5  
action_dim = 3 
agent = PPO(state_dim, action_dim)

path_to_weights = "C:/Users/giaco/Desktop/local-git/PPO_weights/PPO_weights" 
agent.load_weights_PPO(path_to_weights)

env = TrafficManagementEnv()  
num_episodes = 50
all_episode_rewards = []
all_episode_rejections = []
all_managed_requests_per_episode = []
congestione_counts_per_episode = []

prev_total_requests = 0

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
    # Registra le metriche
    all_episode_rewards.append(episode_reward)
    all_episode_rejections.append(env.total_rejected_requests)
    # Calcolo delle richieste gestite per l'episodio corrente
    managed_requests_this_episode = env.total_managed_requests - prev_total_requests
    all_managed_requests_per_episode.append(managed_requests_this_episode)
    # Aggiornamento del valore di prev_total_requests per il prossimo episodio
    prev_total_requests = env.total_managed_requests 
    # Registra il numero di steps in congestione e poi azzeralo per il prossimo episodio
    congestione_counts_per_episode.append(env.congestione_one_count)
    env.congestione_one_count = 0
# Calcola la percentuale di richieste rifiutate per ogni episodio
rejection_percentages = [(rejections/requests) * 100 if requests != 0 else 0 for rejections, requests in zip(all_episode_rejections, all_managed_requests_per_episode)]

# Salvataggio delle informazioni in un file CSV
path_to_save_csv = "C:/Users/giaco/Desktop/Esperimenti/PPO/Scenario 2/seed 4/results.csv"
with open(path_to_save_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    # Scrivi l'intestazione del CSV
    csv_writer.writerow(["Episodio", "Ricompensa Totale", "Richieste Rifiutate", "Richieste Gestite", "Steps in Congestione", "Percentuale di Richieste Rifiutate"])
    # Scrivi i dati per ogni episodio
    for i in range(num_episodes):
        csv_writer.writerow([i + 1, all_episode_rewards[i], all_episode_rejections[i], all_managed_requests_per_episode[i], congestione_counts_per_episode[i], rejection_percentages[i]])

# Plotting
plt.figure(figsize=(30,5))

# subplot per la ricompensa
plt.subplot(1, 4, 1)
plt.plot(all_episode_rewards, marker='o', linestyle='-')
plt.title('Ricompensa Totale per Episodio')
plt.xlabel('Episodi')
plt.ylabel('Ricompensa Totale')
plt.grid(True)

# subplot per la percentuale di rifiuti
plt.subplot(1, 4, 2)
plt.plot(rejection_percentages, marker='o', linestyle='-', color='blue')
plt.title('Percentuale di Richieste Rifiutate per Episodio')
plt.xlabel('Episodi')
plt.ylabel('Percentuale di Richieste Rifiutate')
plt.ylim(0, 100) 
plt.grid(True)

# subplot per il numero assoluto di rifiuti e il numero totale di richieste gestite
plt.subplot(1, 4, 3)
plt.plot(all_episode_rejections, marker='o', linestyle='-', color='red', label='Richieste Rifiutate')
#plt.plot(all_managed_requests_per_episode, marker='o', linestyle='--', color='green', label='Richieste Gestite')
plt.title('Numero di Richieste Rifiutate per Episodio')
plt.xlabel('Episodi')
plt.ylabel('Numero di Richieste')
#plt.legend(loc='upper left')  
plt.grid(True)

# subplot per il numero di steps in congestione per episodio
plt.subplot(1, 4, 4)
plt.plot(congestione_counts_per_episode, marker='o', linestyle='-', color='purple', label='Steps in Congestione')
plt.title('Numero di Steps in Congestione per Episodio')
plt.xlabel('Episodi')
plt.ylabel('Steps in Congestione')
max_congestion = max(congestione_counts_per_episode) 
plt.yticks(range(0, max_congestion + 1))
plt.grid(True)

plt.tight_layout()  
plt.show()
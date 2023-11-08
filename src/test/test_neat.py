import neat
import pickle
import sys
import csv
import matplotlib.pyplot as plt

sys.path.append('C:/Users/giaco/Desktop/tesi_git/src')
from env.env import TrafficManagementEnv

with open('C:/Users/giaco/Desktop/local-git/NEAT/winner_genome.pkl', 'rb') as f:
    winner_genome = pickle.load(f)

config_path = "C:/Users/giaco/Desktop/tesi_git/src/NEAT/config.txt"
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

env = TrafficManagementEnv()
num_episodes = 1
all_episode_rewards = []
all_episode_rejections = []
all_episode_forwarded = []
all_episode_queue_factor = []
all_timesteps_forwarded = []
all_timesteps_local = []
all_timesteps_queue_factor = []
all_managed_requests_per_episode = []
congestione_counts_per_episode = []
prev_total_requests = 0

for episode in range(num_episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action = net.activate(state)
        next_state, reward, done = env.step(action)
            # Salva i valori per ogni timestep
        all_timesteps_forwarded.append(env.forwarded)  # Assumendo che tu abbia un modo per ottenere le richieste inoltrate al timestep corrente
        all_timesteps_local.append(env.local)  # Assumendo che tu abbia un modo per ottenere le richieste elaborate localmente al timestep corrente
        all_timesteps_queue_factor.append(env.QUEUE_factor)
        episode_reward += reward
        state = next_state

    print(f"Episodio {episode + 1}: Ricompensa Totale = {episode_reward}")
    all_episode_rewards.append(episode_reward)
    all_episode_rejections.append(env.total_rejected_requests)
    all_episode_forwarded.append(env.total_forwarded_requests)
    all_episode_queue_factor.append(env.QUEUE_factor)
    managed_requests_this_episode = env.total_managed_requests - prev_total_requests
    all_managed_requests_per_episode.append(managed_requests_this_episode)
    prev_total_requests = env.total_managed_requests
    congestione_counts_per_episode.append(env.congestione_one_count)
    env.congestione_one_count = 0

rejection_percentages = [(rejections/requests) * 100 if requests != 0 else 0 for rejections, requests in zip(all_episode_rejections, all_managed_requests_per_episode)]

path_to_save_csv = "C:/Users/giaco/Desktop/Esperimenti/TUNED/NEAT/Scenario 3/seed 3/results.csv"

with open(path_to_save_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(["Episodio", "Ricompensa Totale", "Richieste Rifiutate", "Richieste Gestite", "Steps in Congestione", "Percentuale di Richieste Rifiutate"])
    for i in range(num_episodes):
        csv_writer.writerow([i + 1, all_episode_rewards[i], all_episode_rejections[i], all_managed_requests_per_episode[i], congestione_counts_per_episode[i], rejection_percentages[i]])
        
# Plotting
plt.figure(figsize=(30,5))
plt.subplot(1, 4, 1)
plt.plot(all_episode_rewards, marker='o', linestyle='-')
plt.title('Ricompensa Totale per Episodio')
plt.xlabel('Episodi')
plt.ylabel('Ricompensa Totale')
plt.grid(True)

plt.subplot(1, 4, 2)
plt.plot(rejection_percentages, marker='o', linestyle='-', color='blue')
plt.title('Percentuale di Richieste Rifiutate per Episodio')
plt.xlabel('Episodi')
plt.ylabel('Percentuale di Richieste Rifiutate')
plt.ylim(0, 100)
plt.grid(True)

plt.subplot(1, 4, 3)
plt.plot(all_episode_rejections, marker='o', linestyle='-', color='red', label='Richieste Rifiutate')
plt.title('Numero di Richieste Rifiutate per Episodio')
plt.xlabel('Episodi')
plt.ylabel('Numero di Richieste')
plt.grid(True)

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

# Crea una nuova figura
plt.figure(figsize=(12, 7))

# Crea l'asse principale
ax1 = plt.gca()
ax1.plot(all_timesteps_forwarded, marker='o', linestyle='-', color='red', label='Richieste Inoltrate')
ax1.plot(all_timesteps_local, marker='o', linestyle='-', color='blue', label='Richieste Locali')
ax1.set_xlabel('Timesteps', color='black', fontsize = 20)
ax1.set_ylabel('Richieste', color='black', fontsize = 20)
ax1.tick_params(axis='y', labelcolor='black', labelsize=14)
ax1.tick_params(axis='x', labelcolor='black', labelsize=14)
ax1.legend(loc='upper right', fontsize = 18)
ax1.grid(True)

# Crea un secondo asse y condividendo l'asse x
ax2 = ax1.twinx()  
ax2.bar(range(len(all_timesteps_queue_factor)), all_timesteps_queue_factor, color='lightblue', alpha=0.5, label='Fattore di Coda')
ax2.set_ylabel('Fattore di Coda', color='black', fontsize = 20)
ax2.tick_params(axis='y', labelcolor='black', labelsize=14)

# Titolo e layout
#plt.title('Richieste e Fattore di Coda per Timestep')
plt.tight_layout()
plt.show()

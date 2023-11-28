import neat
import pickle
import sys
import csv
import matplotlib.pyplot as plt
sys_path = 'C:/Users/giaco/Desktop/repos/RL-edge-computing/src' 
sys.path.append(sys_path)
from env.env import TrafficManagementEnv

with open('C:/Users/giaco/Desktop/repos/RL-edge-computing/logs/NEAT/winner_genome.pkl', 'rb') as f:
    winner_genome = pickle.load(f)

config_path = "C:/Users/giaco/Desktop/repos/RL-edge-computing/src/NEAT/config_optimized.txt"
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

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
        action = net.activate(state)
        next_state, reward, done = env.step(action)
        episode_reward += reward
        state = next_state

    print(f"Episodio {episode + 1}: Ricompensa Totale = {episode_reward}")
    all_episode_rewards.append(episode_reward)
    all_episode_rejections.append(env.total_rejected_requests)
    managed_requests_this_episode = env.total_managed_requests - prev_total_requests
    all_managed_requests_per_episode.append(managed_requests_this_episode)
    prev_total_requests = env.total_managed_requests
    congestione_counts_per_episode.append(env.congestione_one_count)
    env.congestione_one_count = 0

rejection_percentages = [(rejections/requests) * 100 if requests != 0 else 0 for rejections, requests in zip(all_episode_rejections, all_managed_requests_per_episode)]

path_to_save_csv = "C:/Users/giaco/Desktop/repos/RL-edge-computing/logs/NEAT/results.csv"

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
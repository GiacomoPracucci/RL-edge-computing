import neat
import pickle
from env.env import TrafficManagementEnv

with open('C:/Users/giaco/Desktop/local-git/NEAT/winner_genome.pkl', 'rb') as f:
    winner_genome = pickle.load(f)
    
config_path = "C:/Users/giaco/Desktop/tesi-git/src/NEAT/config.txt"
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

env = TrafficManagementEnv()  # Adattalo al tuo ambiente specifico
ob = env.reset()
done = False

while not done:
    action = net.activate(ob)  # Utilizza la rete per determinare l'azione
    ob, reward, done = env.step(action)
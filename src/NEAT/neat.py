import neat
import matplotlib.pyplot as plt
import os
from env.env import TrafficManagementEnv

class ExtendedStatisticsReporter(neat.StatisticsReporter):
    def __init__(self):
        super().__init__()  # chiama il costruttore della classe madre
        self.generation = 0  # inizializza l'attributo generation
    
    def post_evaluate(self, config, population, species, best_genome):
        super().post_evaluate(config, population, species, best_genome)
        
        # Stampa le statistiche ogni N generazioni
        if self.generation % 5 == 0:
            print("\n----- Statistics till Generation {} -----".format(self.generation))
            print("Mean fitness: {:.2f}".format(self.get_fitness_mean()[-1]))
            print("Max fitness: {:.2f}".format(self.get_fitness_stdev()[-1]))
            # ... aggiungi altre statistiche se necessario

    def end_generation(self, config, population, species):
        super().end_generation(config, population, species)

        # Salva grafici ogni N generazioni
        if self.generation % 49 == 0:
            self.plot_statistics()
        self.generation += 1  # Increment the generation counter.

    def plot_statistics(self):
        fig, ax = plt.subplots()
    
        # Retrieve the max fitness values for each generation
        max_fitness_values = [genome.fitness for genome in self.most_fit_genomes]
        ax.plot(max_fitness_values, label="Max Fitness")
    
    # Retrieve the mean fitness values for each generation and plot
        mean_fitness_values = self.get_fitness_mean()
        ax.plot(mean_fitness_values, label="Mean Fitness", linestyle='--')  # Using a dashed line for mean fitness
    
        ax.set(xlabel='Generation', ylabel='Fitness', title='Fitness over Generations')
        ax.grid()
        ax.legend()  # This will display the legend, differentiating Max and Mean fitness
        plt.savefig("C:/Users/giaco/Desktop/local-git/NEAT/neat_fitness_plot_gen{}.png".format(self.generation))
        plt.close()

def eval_genomes(genomes, config):
    env = TrafficManagementEnv()
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        ob = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = net.activate(ob)
            ob, reward, done = env.step(action)
            total_reward += reward
        genome.fitness = total_reward
        if genome.fitness is None:
            print("Warning: Fitness is None for genome_id", genome_id)

def run():
    config_path = "C:/Users/giaco/Desktop/tesi-git/src/NEAT/config.txt"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    checkpointer = neat.Checkpointer(50, None)  # salva ogni 50 generazioni
    pop.add_reporter(checkpointer)
    
    # Utilizza l'ExtendedStatisticsReporter al posto del neat.StatisticsReporter
    stats = ExtendedStatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    winner = pop.run(eval_genomes, 100)

    return winner, stats

"""
def find_latest_checkpoint(directory):
    #Trova l'ultimo checkpoint nella directory specificata.
    checkpoints = [f for f in os.listdir(directory) if "neat-checkpoint-" in f]
    if not checkpoints:
        return None
    return os.path.join(directory, max(checkpoints, key=lambda x: int(x.split('-')[-1])))

def run():
    config_path = "C:/Users/giaco/Desktop/tesi-git/src/NEAT/config.txt"
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    checkpoint_path = "C:/Users/giaco/Desktop/tesi-git"  # Aggiorna con il percorso corretto dei tuoi checkpoint
    latest_checkpoint = find_latest_checkpoint(checkpoint_path)

    if latest_checkpoint:
        # Carica la popolazione dal checkpoint
        pop = neat.Checkpointer.restore_checkpoint(latest_checkpoint)
    else:
        # Crea una nuova popolazione
        pop = neat.Population(config)

    checkpointer = neat.Checkpointer(50, None)  # salva ogni 50 generazioni
    pop.add_reporter(checkpointer)
    
    # Utilizza l'ExtendedStatisticsReporter al posto del neat.StatisticsReporter
    stats = ExtendedStatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    winner = pop.run(eval_genomes, 100)

    return winner, stats
"""
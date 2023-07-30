import neat
import matplotlib.pyplot as plt
from env.env import TrafficManagementEnv
import lightning as L
from torch.utils.data import DataLoader, Dataset
import torch

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
        if self.generation % 9 == 0:
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

# this class generate a fake tensor for the DataLoader of lightning
class DummyDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return torch.tensor([0])

class NeatLightningModule(L.LightningModule):
    def __init__(self, config_path):
        super(NeatLightningModule, self).__init__()
        
        # Carica la configurazione NEAT
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  config_path)
        
        # Crea una popolazione NEAT
        self.pop = neat.Population(self.config)
        
        # Aggiungi reporter per salvare e stampare le statistiche
        self.checkpointer = neat.Checkpointer(50, None)  # salva ogni 50 generazioni
        self.pop.add_reporter(self.checkpointer)
        
        self.stats = ExtendedStatisticsReporter()
        self.pop.add_reporter(self.stats)
        self.pop.add_reporter(neat.StdOutReporter(True))

    def training_step(self, batch, batch_idx):
        # Qui viene eseguito il tuo training loop principale
        winner = self.pop.run(eval_genomes, 10)
        loss = torch.tensor([-winner.fitness], requires_grad=True)
        return {'loss': loss} 
    
    def configure_optimizers(self):
        # NEAT non è basato su gradienti, quindi non c'è un vero e proprio ottimizzatore
        return None
    
    def train_dataloader(self):
        return DataLoader(DummyDataset(), batch_size=1)
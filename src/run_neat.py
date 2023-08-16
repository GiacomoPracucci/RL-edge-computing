import pickle
import sys
sys.path.append('C:/Users/giaco/Desktop/tesi_git/src')
from NEAT.neat import NeatLightningModule
import lightning as L

#fabric = L.Fabric
#fabric.launch()
if __name__ == "__main__":
    winner_genome_path = "C:/Users/giaco/Desktop/local-git/NEAT/winner_genome_ppo.pkl"
    neat_module = NeatLightningModule("C:/Users/giaco/Desktop/tesi_git/src/NEAT/config.txt", winner_genome_path=winner_genome_path)
    
    trainer = L.Trainer(max_epochs=1, accelerator="auto")  # Puoi configurare ulteriori opzioni qui
    trainer.fit(neat_module)
    
    # Salva il genoma vincente
    winner = neat_module.pop.best_genome
    with open("winner_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
import pickle
import sys
sys.path.append('C:/Users/giaco/Desktop/tesi-git/src')
from NEAT.neat import NeatLightningModule
import lightning as L

if __name__ == "__main__":
    neat_module = NeatLightningModule("C:/Users/giaco/Desktop/tesi-git/src/NEAT/config.txt")
    
    trainer = L.Trainer(max_epochs=1, accelerator="auto")  # Puoi configurare ulteriori opzioni qui
    trainer.fit(neat_module)
    
    # Salva il genoma vincente
    winner = neat_module.pop.best_genome
    with open("winner_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
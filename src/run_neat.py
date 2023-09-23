import sys
import pickle
from NEAT.neat import run
sys.path.append('C:/Users/giaco/Desktop/tesi_git/src')

if __name__ == "__main__":
    winner, stats = run()
    print(winner)
    
    # Salva il genoma vincente
    with open("winner_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
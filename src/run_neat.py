import sys
import pickle
from NEAT.neat import run
sys_path = 'C:/Users/giaco/Desktop/repos/RL-edge-computing/src' 
sys.path.append(sys_path)

if __name__ == "__main__":
    winner, stats = run()
    print(winner)
    
    # Salva il genoma vincente
    with open("winner_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
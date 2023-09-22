import numpy as np
import torch
import optuna
import sys
sys.path.append('C:/Users/giaco/Desktop/tesi_git/src')
from ppo.PPO import PPO
from training.training_PPO import train_ppo_agent
from env.env import TrafficManagementEnv

def objective(trial):
    # 1. Definizione degli iperparametri
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    num_units = trial.suggest_int('num_units', 32, 256)
    gamma = trial.suggest_float('gamma', 0.89, 0.999)

    # 2. Inizializzazione dell'agente SAC con i parametri ottimizzati
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TrafficManagementEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPO(state_dim, action_dim, lr=lr, gamma=gamma, num_units=num_units)

    
    # 3. Addestramento e valutazione dell'agente
    rewards = train_ppo_agent(env, agent, num_episodes = 300)
    average_reward = np.mean(rewards)

    return average_reward

def print_best_trial(study, trial):
    if trial.value is not None and trial.value == study.best_value:
        print("\nMiglior trial al termine dell'iterazione", trial.number + 1)
        print("Value: ", trial.value)
        print("Params: ")
        for key, value in trial.params.items():
            print(f"{key}: {value}")
        print("\n")


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100, callbacks=[print_best_trial])

print("Numero di trial: ", len(study.trials))
print("Miglior trial alla fine di tutti i trials:")
trial = study.best_trial

print("Value: ", trial.value)
print("Params: ")
for key, value in trial.params.items():
    print(f"{key}: {value}")
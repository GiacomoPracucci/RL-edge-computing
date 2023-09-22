import numpy as np
import torch
import torch.nn.functional as F
import optuna
import sys
sys.path.append('C:/Users/giaco/Desktop/tesi_git/src')
from sac.SAC import SAC
from training.training_SAC import train_sac_agent
from env.env import TrafficManagementEnv

def objective(trial):

    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.9999)
    tau = trial.suggest_uniform('tau', 0.001, 0.1)
    num_units = trial.suggest_int('num_units', 64, 256)

    # 2. Inizializzazione dell'agente SAC con i parametri ottimizzati
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TrafficManagementEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SAC(state_dim, action_dim, device, lr=lr, gamma=gamma, tau=tau, num_units=num_units)

    # 3. Addestramento e valutazione dell'agente
    rewards = train_sac_agent(env, agent)
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
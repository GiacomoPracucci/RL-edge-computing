import optuna
import sys
sys_path = 'C:/Users/giaco/Desktop/repos/RL-edge-computing/src' 
sys.path.append(sys_path)
from NEAT.neat import run

def modify_config(config_file, config_output_file, params):
    with open(config_file, 'r') as file:
        config_content = file.readlines()
    
    for i in range(len(config_content)):
        for param, value in params.items():
            if param in config_content[i]:
                config_content[i] = f"{param} = {value}\n"
    
    with open(config_output_file, 'w') as file:
        file.writelines(config_content)

def objective(trial):
    conn_add_prob = trial.suggest_float('conn_add_prob', 0.2, 0.7)
    node_add_prob = trial.suggest_float('node_add_prob', 0.2, 0.5)
    weight_mutate_rate = trial.suggest_float('weight_mutate_rate', 0.7, 0.95)
    survival_threshold = trial.suggest_float('survival_threshold', 0.2, 0.5)

    modify_config(
        'C:/Users/giaco/Desktop/repos/RL-edge-computing/src/NEAT/config.txt', # path to existing config file
        'C:/Users/giaco/Desktop/repos/RL-edge-computing/src/NEAT/config_optimized.txt', # path to new optimized config file
        {
            'conn_add_prob': conn_add_prob,
            'node_add_prob': node_add_prob,
            'weight_mutate_rate': weight_mutate_rate,
            'survival_threshold': survival_threshold
        }
    )

    winner, stats = run()  
    return winner.fitness  

def print_best_trial(study, trial):
    if trial.value is not None and trial.value == study.best_value:
        print("\nMiglior trial al termine dell'iterazione", trial.number + 1)
        print("Value: ", trial.value)
        print("Params: ")
        for key, value in trial.params.items():
            print(f"{key}: {value}")
        print("\n")

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100, callbacks=[print_best_trial])

    # Salva i migliori parametri
    trial = study.best_trial
    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

    # Potresti anche voler salvare i migliori parametri in un file
    with open('best_params.txt', 'w') as f:
        f.write(str(trial.params))
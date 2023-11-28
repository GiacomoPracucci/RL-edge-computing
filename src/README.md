# How to navigate through code scripts

Code scripts are organized in the following way:
1. The `NEAT` folder contains scripts related to the implementation of the NEAT algorithm.
2. The `PPO` folder contains scripts related to the implementation of the Proximal Policy Optimization algorithm.
      - PPO.py contains the definition of the structure of the networks and the implementation of the PPO class, where the parameters of the algorithm and the necessary steps for its operation (advantage, loss functions, etc...etc...) are defined.
3. The `SAC` folder contains scripts related to the implementation of the Soft-Actor Critic algorithm.
      - SAC.py contains the definition of the structure of the networks and the implementation of the SAC class, where the parameters of the algorithm and the necessary steps for its operation (calculation of q-values, loss functions, etc...etc...) are defined.
      - replay_buffer.py contains the implementation of the replay_buffer class necessary for the operation of SAC
4. The `env` folder contains scripts related to the implementation of the environment
      - env.py contains the class that defines the core part of the environment (the observation space, actions, what happens at reset and the dynamics of each step).
      - workload_management.py contains the class in which the dynamics related to CPU request allocation and buffer queue management are defined
      - env_functions.py contains functions that will be used in the main environment class (reward function, process actions)
5. The `training folder` contains:
      - the `training_SAC.py` and `training_PPO` files for SAC and PPO training, which specifies how agent training takes place (episodes, results graphs, etc.).
6. The `utils directory` contains useful scripts:
   - `check gradients.py` is the script that checks the gradient values ​​of the networks
7. The `test directory` contains the scripts needed to run simulation of the 3 algorithms in the implemented scenarios
8. The `optimization directory` contains the scripts needed to run bayesian optimization of the 3 algorithms in the implemented scenarios
9.
   - `run_PPO.py` is the file that needs to be run to launch the training of PPO
   - `run_SAC.py` is the file that needs to be run to launch the training of SAC
   - `run_NEAT.py` is the file that needs to be run to launch the training of NEAT

# How to navigate through code scripts

Code scripts are organized in the following way:
1. The `NEAT` folder contains scripts related to the implementation of the NEAT algorithm.
2. The `SAC` folder contains scripts related to the implementation of the Soft-Actor Critic algorithm.
      - SAC.py contains the definition of the structure of the networks and the implementation of the SAC class, where the parameters of the algorithm and the necessary steps for its operation (calculation of q-values, loss functions, etc...etc...) are defined.
      - replay_buffer.py contains the implementation of the replay_buffer class necessary for the operation of SAC
3. The `env` folder contains scripts related to the implementation of the environment
      - env.py contains the class that defines the core part of the environment (the observation space, actions, what happens at reset and the dynamics of each step).
      - workload_management.py contains the class in which the dynamics related to CPU request allocation and buffer queue management are defined
      - env_functions.py contains functions that will be used in the main environment class (reward function, process actions)
4. The `training folder` contains the training.py file, which specifies how agent training takes place (episodes, results graphs, etc.).
5.
   - `run_SAC.py` is the file that needs to be run to launch the training of SAC
   - `run_NEAT.py` is the file that needs to be run to launch the training of NEAT
   - `genome_test.py` is the file that loads the winning genome weights and runs an episode of the environment

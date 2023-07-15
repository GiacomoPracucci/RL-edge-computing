# How to move between code scripts

Code scripts are organized in the following way:
1. The `SAC` folder contains scripts related to the implementation of the Soft-Actor Critic algorithm.
      - SAC.py contains the definition of the structure of the networks and the implementation of the SAC class, where the parameters of the algorithm and the necessary steps for its operation (calculation of q-values, loss functions, etc...etc...) are defined.
      - replay_buffer.py contains the implementation of the replay_buffer class necessary for the operation of SAC
2. The 'env' folder contains scripts related to the implementation of the environment
      - env.py contains the class that defines the core part of the environment (the observation space, actions, what happens at reset and the dynamics of each step).
      - env_functions.py contains functions that will be used in the main environment class (reward function, action processing and function that implements the dynamics between CPU_capacity and the buffer queue)
3. The `training folder` contains the training.py file, which specifies how agent training takes place (episodes, result graphs, etc.).
4. `run_experiment.py` is the file that needs to be run to launch the training

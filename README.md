![s2s3_reward](https://github.com/GiacomoPracucci/RL-edge-computing/assets/94844087/0e768c82-e1a0-4045-a260-8601523aed19)# Reinforcement Learning for Traffic Management in an Edge Computing System 

## Description
The project proposes the implementation of  SAC (Soft actor-critic) and PPO (Proximal Policy Optimization) deep reinforcement learning algorithms and of the evolutionary algorithm NEAT (Neuro Evolution of Augmenting Topologies) to optimize workload management in an Edge Computing system. The goal is to find the optimal policy for local processing, forwarding of requests to edge nodes, and rejection of requests based on system conditions.
The current implementation still has simplifying assumptions compared to the real scenario.

In the simulated environment, the agent receives a sequence of incoming requests over time. It must decide, at each step, to process these requests locally, forward them to another edge node, or reject them. The number of incoming requests can change over time, following a sinusoidal function.

The `action space` is a three-dimensional continuous box where each dimension corresponds to the proportions of requests that are processed locally, forwarded, or rejected.

The `observation space` consists of four components:
- The number of incoming requests
- The remaining queue capacity
- The remaining forward capacity
- A congestion flag, indicating whether the queue is congested

The `reward function` in this environment depends on the actions taken by the agent and the system state. The reward function provides more points for processing requests locally and fewer points for forwarding requests. It penalizes the system heavily for rejecting requests and for causing congestion in the queue.

## Best experiment results
The highest reward scores and best generalization abilities in scenarios other than training were obtained by PPO with standard hyperparameters, trained in scenario 2.

- Results achieved by testing ppo (trained in scenario 2) in scenario 3
![s2s3_reward](https://github.com/GiacomoPracucci/RL-edge-computing/assets/94844087/e8c54160-2083-4040-9aef-1ce708b8822c)
![s2s3_rejected](https://github.com/GiacomoPracucci/RL-edge-computing/assets/94844087/2d19ae0f-1dd8-4515-92c5-a4ad58a1cfc4)


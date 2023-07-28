# Reinforcement Learning for Traffic Management in an Edge Computing System 

## Description
This project proposes the implementation of the reinforcement learning algorithm `SAC` (Soft actor-critc) and `NEAT` (Neuro Evolution of Augmenting Topologies) to optimize `workload management in an Edge Computing system`. The goal is to find the optimal policy for local processing, forwarding of requests to edge nodes, and rejection of requests based on system conditions.  
The current implementation still has simplifying assumptions compared to the real scenario.

In the simulated environment, the agent receives a sequence of incoming requests over time. It must decide, at each step, to process these requests locally, forward them to another edge node, or reject them. The number of incoming requests can change over time, following a sinusoidal function.

The `action space` is a three-dimensional continuous box where each dimension corresponds to the proportions of requests that are processed locally, forwarded, or rejected.

The `observation space` consists of four components:
- The number of incoming requests
- The remaining queue capacity
- The remaining forward capacity
- A congestion flag, indicating whether the queue is congested

The `reward function` in this environment depends on the actions taken by the agent and the system state. The reward function provides more points for processing requests locally and fewer points for forwarding requests. It penalizes the system heavily for rejecting requests and for causing congestion in the queue.

## Results
- SAC  
![download](https://github.com/GiacomoPracucci/RL-edge-computing/assets/94844087/4ae669dc-18b7-4205-b06c-4c9c2fe4acdd)

- NEAT  
![neat_fitness_plot_gen98](https://github.com/GiacomoPracucci/RL-edge-computing/assets/94844087/43d20003-c541-4f29-b6e0-4ef494f40eb8)

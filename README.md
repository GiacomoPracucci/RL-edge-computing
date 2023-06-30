# Reinforcement Learning for Traffic Management in an Edge Computing System 

## Description
This project proposes the implementation of a reinforcement learning algorithm, Deep Deterministic Policy Gradient (DDPG), to optimize workload management in an Edge Computing system. The goal is to find the optimal policy for local processing, forwarding of requests to edge nodes, and rejection of requests based on system conditions.
The current implementation still has simplifying assumptions compared to the real scenario.

![Immagine 2023-06-30 143629](https://github.com/GiacomoPracucci/Tesi-RL/assets/94844087/cc469b30-55a2-4374-81b7-a58b71c60e7b)

## Environment
The environment simulates a distributed processing system with maximum local processing capacity and a queue to handle incoming requests. At each new episode, the environment is reset with the following conditions:  

- Maximum CPU capacity (50 units)  
- Maximum queue capacity (100 units)  
- Requests are generated according to a sinusoidal function with a minimum of 50, a maximum of 150, and a period of 99. All requests are assumed to require the same amount of CPU.  

A "step" ends when the queue fills up, indicating system congestion. At the end of each step, system status information, including CPU capacity and queue capacity, is updated.  

The goal is to prioritize local processing unless the queue is nearly full. In that case, to avoid congestion, the agent must forward requests.

## DDPG
The DDPG algorithm is implemented in TensorFlow. The parameters of the algorithm were not optimized through a specific technique, but through various training attempts and certainly need more precise tuning.    

To avoid endless episodes in the case of optimal policies, a maximum of 50 steps per episode is set.    

To aid exploration in a deterministic context, Ornstein-Uhlenbeck (OU) noise is introduced to modify the output of the actor network. The sigma value for OU noise starts from a relatively high level (0.30) and decays over the course of episodes.

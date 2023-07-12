# Reinforcement Learning for Traffic Management in an Edge Computing System 

## Description
This project proposes the implementation of a reinforcement learning algorithms (SAC), to optimize workload management in an Edge Computing system. The goal is to find the optimal policy for local processing, forwarding of requests to edge nodes, and rejection of requests based on system conditions.  
The current implementation still has simplifying assumptions compared to the real scenario.

## Environment
The environment simulates a distributed processing system with maximum local processing capacity and a queue to handle incoming requests. At each new episode, the environment is reset with the following conditions:  

- Maximum CPU capacity (50 units)  
- Maximum queue capacity (100 units)  
- Requests are generated according to a sinusoidal function with a minimum of 50, a maximum of 150, and a period of 99. All requests are assumed to require the same amount of CPU.   

The goal is to prioritize local processing unless the queue is nearly full. In that case, to avoid congestion, the agent must forward requests.

# Reinforcement Learning for load distribution in decentralized Edge environment

Paper: https://dl.acm.org/doi/10.1145/3660319.3660331

## Description
The project proposes the implementation of  SAC (Soft actor-critic) and PPO (Proximal Policy Optimization) deep reinforcement learning algorithms and of the evolutionary algorithm NEAT (Neuro Evolution of Augmenting Topologies) to optimize workload management in an Edge Computing system (DFaaS). The goal is to find the optimal policy for local processing, forwarding of requests to edge nodes, and rejection of requests based on system conditions.
The current implementation still has simplifying assumptions compared to the real scenario.

In the simulated environment, the agent receives a sequence of incoming requests over time. At each step, it must decide how many of these requests to process locally, how many to forward to another edge node, and/or how many to reject. The number of incoming requests varies over time.

The `action space` is a three-dimensional continuous box where each dimension corresponds to the proportions of requests that are processed locally, forwarded, or rejected.

The `observation space` consists of four components:
- The number of incoming requests
- The remaining queue capacity
- The remaining forward capacity
- A congestion flag, indicating whether the queue is congested

The `reward function` in this environment depends on the actions taken by the agent and the system state. The reward function provides more points for processing requests locally and fewer points for forwarding requests. It penalizes the system heavily for rejecting requests and for causing congestion in the queue.

## Training and test settings
Three different training scenarios were defined, distinguished by the different way of generating requests to be processed and the different way of updating the available forwarding capacity to other nodes.
- Scenario 1
![scenario_1](https://github.com/GiacomoPracucci/RL-edge-computing/assets/94844087/bd9a1f56-d54a-4353-93f4-6f4ec56050b5)
- Scenario 2
![scneario_2](https://github.com/GiacomoPracucci/RL-edge-computing/assets/94844087/785cb86f-30d4-49df-b208-c797498834c3)
- Scenario 3
![scenario_3](https://github.com/GiacomoPracucci/RL-edge-computing/assets/94844087/b513e072-40de-4f35-8a35-e051416a1599)

The idea is to evaluate the results obtained according to different work contexts. Different scenarios allow us to assess the generalization capabilities of the algorithms by evaluating the performance obtained in work scenarios other than the training scenario (overfitting evaluation).

## Best experiment results
The highest reward scores and best generalization abilities have been achieved by PPO with standard hyperparameters, trained in scenario 2.

- Results achieved by testing ppo (trained in scenario 2) in scenario 3
![s2s3_reward](https://github.com/GiacomoPracucci/RL-edge-computing/assets/94844087/e8c54160-2083-4040-9aef-1ce708b8822c)
![s2s3_rejected](https://github.com/GiacomoPracucci/RL-edge-computing/assets/94844087/2d19ae0f-1dd8-4515-92c5-a4ad58a1cfc4)

- Results achieved by testing ppo (trained in scenario 2) in scenario 1
![s2s1_reward](https://github.com/GiacomoPracucci/RL-edge-computing/assets/94844087/c362afe5-6b91-452e-ad54-c31751a3951a)
![s2s1_rejected](https://github.com/GiacomoPracucci/RL-edge-computing/assets/94844087/1e3487d9-7011-4add-a20c-dace4c962e64)

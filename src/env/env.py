import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from env.env_functions import process_actions, sample_workload, calculate_reward1

# ENV CLASS
class TrafficManagementEnv(gym.Env):
    def __init__(self, CPU_capacity = 1000, queue_capacity = 100, forward_capacity = 100, 
                 average_requests = 100, amplitude_requests = 50, period=50, congestione =0, DFAAS_capacity = 8000):
        super().__init__()
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low = np.array([50, 0, 0, 0]), high = np.array([150, 100, 100, 1]), dtype = np.float32)

        self.max_CPU_capacity = CPU_capacity
        self.max_queue_capacity = queue_capacity
        self.max_DFAAS_capacity = DFAAS_capacity

        self.average_requests = average_requests
        self.amplitude_requests = amplitude_requests
        self.period = period
        self.t = 0
        
        self.congestione = congestione

        self.max_forward_capacity = forward_capacity
        self.forward_capacity_t = self.max_forward_capacity

        self.input_requests = self.calculate_requests()

    def calculate_requests(self):
        return int(self.average_requests + self.amplitude_requests * math.sin(2 * math.pi * self.t / self.period))
    
    def reset(self):
        self.t = 0
        self.CPU_capacity = self.max_CPU_capacity
        self.queue_capacity = self.max_queue_capacity
        self.forward_capacity = self.max_forward_capacity
        self.forward_capacity_t = self.max_forward_capacity
        self.DFAAS_capacity = self.max_DFAAS_capacity
        self.congestione = 0

        return np.array([self.input_requests, self.queue_capacity, self.forward_capacity, self.congestione], dtype=np.float32)
    
    def step(self, action):
        print(f"Stato sistema: {self.congestione}")
        print(f"INPUT: {self.input_requests}")
        print(f"CPU Capacity: {self.CPU_capacity}")
        print(f"DFAAS Capacity: {self.DFAAS_capacity}")
        print(f"Queue Capacity: {self.queue_capacity}")
        print(f"Forward Capacity: {self.forward_capacity}")
    
        self.local, self.forwarded, self.rejected = process_actions(action, self.input_requests)

        print(f"LOCAL: {self.local}")
        print(f"FORWARDED: {self.forwarded}")
        print(f"REJECTED: {self.rejected}")

        self.QUEUE_factor = self.queue_capacity / self.max_queue_capacity
        self.FORWARD_factor = self.forward_capacity / self.max_forward_capacity

        reward = calculate_reward1(self.local, self.forwarded, self.rejected, self.QUEUE_factor, self.FORWARD_factor, self.congestione)
        print(f"REWARD: {reward}")
        
        local_workload = sample_workload(self.local)
        self.CPU_workload = []
        self.queue_workload = []
        for request in local_workload:
            if self.CPU_capacity >= request['shares'] and self.DFAAS_capacity >= request['dfaas_mb']:
                self.CPU_capacity -= request['shares']
                self.DFAAS_capacity -= request['dfaas_mb']
                self.CPU_workload.append(request)
            else:
                self.queue_workload.append(request)
        queue_length_share = sum(request['shares'] for request in self.queue_workload)
        dfaas_queue_length_mb = sum(request['dfaas_mb'] for request in self.queue_workload)
        self.CPU_capacity = max(-1000, self.max_CPU_capacity - queue_length_share)
        self.DFAAS_capacity = max(-1000, self.max_DFAAS_capacity - dfaas_queue_length_mb)
        queue_length_requests = len(self.queue_workload)
        self.queue_capacity = max(0, self.max_queue_capacity - queue_length_requests)

        self.forward_capacity = int(25 + 75 * (1 + math.sin(2 * math.pi * self.t / self.period)) / 2)
        self.forward_capacity_t = self.forward_capacity
        self.congestione = 1 if self.queue_capacity == 0 else 0
        
        self.t += 1
        if self.t == 100:
            done = True
        else:
            done = False
        self.input_requests = self.calculate_requests()
        state = np.array([self.input_requests, self.queue_capacity, self.forward_capacity, self.congestione], dtype=np.float32)
        
        return state, reward, done

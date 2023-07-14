import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math

# PROCESS THE ACTION COMING FROM THE AGENT
# NORMALIZE AND CONVERT TO INTEGER
# ASSIGN THE REMAINING REQUEST (AFTER NORM) TO THE ACTION WITH THE HIGHEST FRACTION
def process_actions(action, input_requests):
    action_sum = np.sum(action)
    if action_sum == 0:
        action_sum += 1e-8
    action /= action_sum

    local = int(action[0] * input_requests)
    forwarded = int(action[1] * input_requests)
    rejected = int(action[2] * input_requests)
    local_fraction= (action[0] * input_requests) - local
    forwarded_fraction = (action[1] * input_requests) - forwarded
    rejected_fraction = (action[2] * input_requests) - rejected
    total_actions = local + forwarded + rejected

    if total_actions < input_requests:
        fractions = [local_fraction, forwarded_fraction, rejected_fraction]
        actions = [local, forwarded, rejected]
        max_fraction_index = np.argmax(fractions)
        actions[max_fraction_index] += input_requests - total_actions
        local, forwarded, rejected = actions

    return local, forwarded, rejected

def sample_workload(local):
    workload = []
    for i in range(local):
        sample = np.random.uniform()
        if sample < 0.33:
            request_class = 'A'
            shares = np.random.randint(1, 11)  
        elif sample < 0.67:
            request_class = 'B'
            shares = np.random.randint(11, 21)  
        else:
            request_class = 'C'
            shares = np.random.randint(21, 31)  
        workload.append({'class': request_class, 'shares': shares, 'position': i})
    return workload

# CALCULATE THE REWARD
def calculate_reward1(local, forwarded, rejected, QUEUE_factor, FORWARD_factor):
    reward_local = 3 * local * QUEUE_factor
    reward_forwarded = 1 * forwarded * (1 - QUEUE_factor) * FORWARD_factor
    reward_rejected = -5 * rejected * FORWARD_factor * QUEUE_factor
    reward = reward_local + reward_forwarded + reward_rejected

    return reward

# ENV CLASS
class TrafficManagementEnv(gym.Env):
    def __init__(self, CPU_capacity = 50, queue_capacity = 100, forward_capacity = 100, average_requests = 100, amplitude_requests = 50, period=50):
        super().__init__()
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low = np.array([50, 0, 0]), high = np.array([150, 100, 100]), dtype = np.float32)

        self.max_CPU_capacity = CPU_capacity
        self.max_queue_capacity = queue_capacity

        self.average_requests = average_requests
        self.amplitude_requests = amplitude_requests
        self.period = period
        self.t = 0

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

        return np.array([self.input_requests, self.queue_capacity, self.forward_capacity], dtype=np.float32)
    
    def step(self, action):
        print(f"INPUT: {self.input_requests}")
        print(f"CPU Capacity: {self.CPU_capacity}")
        print(f"Queue Capacity: {self.queue_capacity}")
        print(f"Forward Capacity: {self.forward_capacity}")
    
        self.local, self.forwarded, self.rejected = process_actions(action, self.input_requests)

        print(f"LOCAL: {self.local}")
        print(f"FORWARDED: {self.forwarded}")
        print(f"REJECTED: {self.rejected}")

        self.QUEUE_factor = self.queue_capacity / self.max_queue_capacity
        self.FORWARD_factor = self.forward_capacity / self.max_forward_capacity

        reward = calculate_reward1(self.local, self.forwarded, self.rejected, self.QUEUE_factor, self.FORWARD_factor)
        print(f"REWARD: {reward}")
        
        local_workload = sample_workload(self.local)
        self.CPU_workload = []
        self.queue_workload = []
        for request in local_workload:
            if self.CPU_capacity >= request['shares']:
                self.CPU_capacity -= request['shares']
                self.CPU_workload.append(request)
            else:
                self.queue_workload.append(request)
        queue_length_share = sum(request['shares'] for request in self.queue_workload)
        self.CPU_capacity = max(-1000, self.max_CPU_capacity - queue_length_share)
        queue_length_requests = len(self.queue_workload)
        self.queue_capacity = max(0, self.max_queue_capacity - queue_length_requests)

        self.forward_capacity = int(25 + 75 * (1 + math.sin(2 * math.pi * self.t / self.period)) / 2)
        self.forward_capacity_t = self.forward_capacity
        
        self.t += 1
        if self.t == 100:
            done = True
        else:
            done = False
            
        self.input_requests = self.calculate_requests()
        state = np.array([self.input_requests, self.queue_capacity, self.forward_capacity], dtype=np.float32)
        
        return state, reward, done

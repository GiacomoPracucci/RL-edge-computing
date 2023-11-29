import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from env.env_functions import process_actions, calculate_reward1, update_obs_space
from env.workload_management import workload

'''This script contains a special version of the environment that meets the specifications necessary 
to be used as a custom environment in the RL sheeprl framework.
It does not work in the algorithms in the repo because the state is saved as a dictionary,
a format not supported in this code.
sheeeprl: https://github.com/Eclectic-Sheep/sheeprl
How to add an env in sheeprl: https://github.com/Eclectic-Sheep/sheeprl/blob/main/howto/add_environment.md
'''

# ENV CLASS
class TrafficManagementEnv(gym.Env):
    def __init__(self, CPU_capacity = 1000, queue_capacity = 100, DFAAS_capacity = 8000, forward_capacity = 100,
                average_requests = 100, amplitude_requests = 50, period=50, cong1 = 0, cong2 =0, forward_exceed = 0):
        super().__init__()
        
        self.action_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'input_requests': spaces.Box(low=50, high=150, shape=(1,), dtype=np.float32),
            'queue_capacity': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            'forward_capacity': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            'cong1': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'cong2': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })

        self.max_CPU_capacity = CPU_capacity
        self.max_queue_capacity = queue_capacity
        self.max_DFAAS_capacity = DFAAS_capacity
        self.max_forward_capacity = forward_capacity
        self.forward_capacity_t = self.max_forward_capacity
        self.forward_exceed = forward_exceed

        self.cong1 = cong1
        self.cong2 = cong2
        self.congestione_zero_count = 0
        self.congestione_one_count = 0
        self.total_managed_requests = 0
        self.total_rejected_requests = 0
        self.total_forwarded_requests = 0
        self.total_local_requests = 0
        
        self.average_requests = average_requests
        self.amplitude_requests = amplitude_requests
        self.period = period
        self.t = 0
        
        self.queue_workload = []
        self.input_requests = self.calculate_requests()
        
        self.reward_range = (-np.inf, np.inf)
    
    def calculate_requests(self):
        return int(self.average_requests + self.amplitude_requests * math.sin(2 * math.pi * self.t / self.period))
    
    def reset(self, seed=None, options=None):
        self.t = 0
        self.CPU_capacity = self.max_CPU_capacity
        self.queue_capacity = self.max_queue_capacity
        self.DFAAS_capacity = self.max_DFAAS_capacity
        self.forward_capacity = self.max_forward_capacity
        self.forward_capacity_t = self.max_forward_capacity
        self.forward_exceed = 0
        self.queue_shares = 0
        self.queue_workload = []
        self.total_rejected_requests = 0
        self.total_forwarded_requests = 0
        self.total_local_requests = 0
        self.cong1 = 0
        self.cong2 = 0

        initial_observation = {
            'input_requests': np.array([self.input_requests], dtype=np.float32),
            'queue_capacity': np.array([self.queue_capacity], dtype=np.float32),
            'forward_capacity': np.array([self.forward_capacity], dtype=np.float32),
            'cong1': np.array([self.cong1], dtype=np.float32),
            'cong2': np.array([self.cong2], dtype=np.float32)
        }
        return initial_observation, {}
    
    def step(self, action):
        '''
        #1. VISUALIZZO LO STATO ATTUALE DEL SISTEMA
        print(f"Stato del Sistema 1: {self.cong1}")
        print(f"Stato del Sistema 2: {self.cong2}")
        print(f"Queue Capacity: {self.queue_capacity}")
        print(f"Shares in Coda: {self.queue_shares}")
        print(f"Forward Capacity: {self.forward_capacity}")
        print(f"INPUT: {self.input_requests}")
        '''
        
        #2. ESTRAGGO, SALVO E VISUALIZZO IL NUMERO DI RICHIESTE ELABORATE LOCALMENTE, INOLTRATE E RIFIUTATE
        self.local, self.forwarded, self.rejected = process_actions(action, self.input_requests)
        self.total_managed_requests += self.local + self.forwarded + self.rejected
        self.total_forwarded_requests += self.forwarded
        self.total_local_requests += self.local
        #print(f"LOCAL: {self.local}")
        #print(f"FORWARDED: {self.forwarded}")
        #print(f"REJECTED: {self.rejected}")

        #3. CALCOLO I PESI PER IL SISTEMA DI RICOMPENSA E LA REWARD
        self.QUEUE_factor = self.queue_capacity / self.max_queue_capacity
        self.FORWARD_factor = self.forward_capacity / self.max_forward_capacity
        self.forward_exceed = max(0, self.forwarded - self.forward_capacity) # limito il valore a 0 come minimo, perchè se inoltro meno richieste di quelle che gli altri nodi possono accogliere, vuol dire che non ho ecceduto
        reward = calculate_reward1(self.local, self.forwarded, self.rejected, 
                                   self.QUEUE_factor, self.FORWARD_factor, self.cong1, self.cong2, self.forward_exceed)
        #print(f"REWARD: {reward}")
        
        '''
        4. COSTRUISCO LE LISTE DI CPU_workload E queue_workload
        Viene fatto il campionamento delle richieste elaborate in CPU e quelle messe in coda (Classe, shares, dfaas_mb, position)
        Il campionamento per la classe avviene da una distrib uniforme, per gli shares e i dfaas_mb da una distrib normale
        Costruisco le liste che descrivono quanto ho elaborato in CPU in questo step e quanto ho messo in coda
        Viene data precedenza all'elaborazione delle requests in coda dallo step precedente
        '''
        self.CPU_workload, self.queue_workload, new_rejections = workload.manage_workload(self.local, self.CPU_capacity, 
                                                                    self.DFAAS_capacity, self.queue_workload, self.max_queue_capacity,
                                                                    self.max_CPU_capacity, self.max_DFAAS_capacity)
        self.total_rejected_requests += self.rejected + new_rejections
        #5. AGGIORNO LO SPAZIO DELLE OSSERVAZIONI
        # Aggiorno la capacità disponibile in base al n di requests in queue_workload
        # Verifico la condizione per il done
        scenario = "scenario1"
        self.queue_capacity, self.queue_shares, self.t, done, self.forward_capacity, self.forward_capacity_t, self.cong1, self.cong2, self.congestione_zero_count, self.congestione_one_count, self.input_requests = update_obs_space(scenario, self.average_requests, self.amplitude_requests, self.queue_workload, self.queue_capacity, self.max_queue_capacity, self.t,
                                                                                                                                                                                                                                        self.forward_capacity, self.forward_capacity_t, self.period, self.cong1, self.cong2,
                                                                                                                                                                                                                                        self.forward_exceed, self.congestione_zero_count, self.congestione_one_count)   
        
        truncated = False
        terminated = done
        info = {}
        
        #print(f"Steps non in congestione: {self.congestione_zero_count}")
        #print(f"Steps in congestione: {self.congestione_one_count}")
        state = {
            'input_requests': np.array([self.input_requests], dtype=np.float32),
            'queue_capacity': np.array([self.queue_capacity], dtype=np.float32),
            'forward_capacity': np.array([self.forward_capacity], dtype=np.float32),
            'cong1': np.array([self.cong1], dtype=np.float32),
            'cong2': np.array([self.cong2], dtype=np.float32)
        }
        return state, reward, truncated, terminated, info
    
    def render(self, mode="human", close=False):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        pass
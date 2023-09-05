import numpy as np
import math
import random
seed = 0
np.random.seed(seed)

class workload:
    
    @staticmethod
    def sample_workload(local):
        workload = []
        for i in range(local):
            sample = np.random.uniform()
            if sample < 0.33:
                request_class = 'A'
                mean, std_dev = 5.5, 2.5
                shares = np.clip(np.random.normal(mean, std_dev), 1, 10)  
                dfaas_mb = np.clip(np.random.normal(13, 2.5), 1, 25) 
            elif sample < 0.67:
                request_class = 'B'
                mean, std_dev = 15.5, 2.5 
                shares = np.clip(np.random.normal(mean, std_dev), 11, 20)
                dfaas_mb = np.clip(np.random.normal(38, 2.5), 26, 50) 
            else:
                request_class = 'C'
                mean, std_dev = 25.5, 2.5
                shares = np.clip(np.random.normal(mean, std_dev), 21, 30)
                dfaas_mb = np.clip(np.random.normal(63, 2.5), 51, 75) 
            workload.append({'class': request_class, 'shares': shares, 'dfaas_mb': dfaas_mb, 'position': i})
        return workload

    @staticmethod
    def calculate_requests(average_requests, amplitude_requests, t, period):
        return int(average_requests + amplitude_requests * math.sin(2 * math.pi * t / period))
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------   
    # SCENARIO RANDOM-RANDOM
    @staticmethod
    def scenario1(average_requests=50, average_capacity=50, stddev=10):  
        input_requests = int(random.gauss(average_requests, stddev))
        forward_capacity = int(random.gauss(average_capacity, stddev))
        return input_requests, forward_capacity

    # SCENARIO SINUSOIDE NOISY - SINUSOIDE NOISY
    @staticmethod
    def scenario2(t, period, average_requests=50, amplitude_requests=50, average_capacity=50, amplitude_capacity=50, noise_ratio=0.1):  
        base_input = average_requests + amplitude_requests * math.sin(2 * math.pi * t / period)
        noisy_input = base_input + noise_ratio * random.gauss(0, amplitude_requests)
        input_requests = int(noisy_input)

        base_capacity = average_capacity + amplitude_capacity * math.sin(2 * math.pi * t / period)
        noisy_capacity = base_capacity + noise_ratio * random.gauss(0, amplitude_capacity)
        forward_capacity = int(noisy_capacity)

        return input_requests, forward_capacity
    
    # SCENARIO SINUSOIDE - SINUSOIDE
    def scenario3(average_requests, amplitude_requests, t, period):  
        input_requests = int(average_requests + amplitude_requests * math.sin(2 * math.pi * t / period))
        forward_capacity = int(25 + 75 * (1 + math.sin(2 * math.pi * t / period)) / 2)
        return input_requests, forward_capacity
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------      
    @staticmethod
    def manage_workload(local ,CPU_capacity, DFAAS_capacity, queue_workload,
                        max_queue_capacity, max_CPU_capacity, max_DFAAS_capacity):
        
        local_workload = workload.sample_workload(local) # sample del workload dal numero di richieste da elaborare in locale
        CPU_workload = []                                # lista di richieste da elaborare in CPU
        CPU_capacity = max_CPU_capacity 
        DFAAS_capacity = max_DFAAS_capacity
        requests_rejected = 0                           
        
        # 1. Diamo precedenza alle richieste in coda dallo step precedente
        # Se la CPU_capacity e la DFAAS_capacity sono sufficienti per soddisfare le richieste in coda,
        # le richieste in coda vengono aggiunte alla CPU_workload e rimosse dalla queue_workload
        for request in queue_workload.copy(): 
            if CPU_capacity >= request['shares'] and DFAAS_capacity >= request['dfaas_mb']:
                CPU_capacity -= request['shares']
                DFAAS_capacity -= request['dfaas_mb']
                CPU_workload.append(request)
                queue_workload.remove(request)
            else:
                break
        print(f"CPU disponibile per le nuove requests: {CPU_capacity}")
        print(f"DFAAS disponibile per le nuove requests: {DFAAS_capacity}") 
        
        # 2. Processiamo le requests local_workload
        # Viene processata una richiesta alla volta, se la CPU_capacity e la DFAAS_capacity sono sufficienti
        # la richiesta viene aggiunta alla CPU_workload, altrimenti viene messa in queue_workload
        # (solo se la queue_workload non ha raggiunto la sua capacità massima)
        # se la coda ha già raggiunto la sua capacità massima, la richiesta viene rigettata
        for request in local_workload:
            if CPU_capacity >= request['shares'] and DFAAS_capacity >= request['dfaas_mb']:
                CPU_capacity -= request['shares']
                DFAAS_capacity -= request['dfaas_mb']
                CPU_workload.append(request)
            else:
                if len(queue_workload) < max_queue_capacity:
                    queue_workload.append(request)
                else:
                    requests_rejected += 1
        
        print(f"Num requests in queue: {len(queue_workload)}")
        print(f"Shares in QUEUE: {sum(request['shares'] for request in queue_workload)}")
        print(f"MB in QUEUE: {sum(request['dfaas_mb'] for request in queue_workload)}")

        return CPU_workload, queue_workload, requests_rejected

    @staticmethod
    def update_obs_space(scenario, average_requests, amplitude_requests, queue_workload, queue_capacity, max_queue_capacity, t,
                     forward_capacity, forward_capacity_t, period, cong1, cong2, congestione,
                     forward_exceed, congestione_zero_count, congestione_one_count):

        print(f"Num requests in queue: {len(queue_workload)}")
        queue_length_requests = len(queue_workload)
        queue_capacity = max(0, max_queue_capacity - queue_length_requests)
        queue_shares = sum(request['shares'] for request in queue_workload)
        
        if scenario == "scenario1":
            input_requests, forward_capacity = workload.scenario1(average_requests=average_requests)
        elif scenario == "scenario2":
            input_requests, forward_capacity = workload.scenario2(t, period, average_requests=average_requests, amplitude_requests=amplitude_requests)
        elif scenario == "scenario3":
            input_requests, forward_capacity = workload.scenario3(average_requests, amplitude_requests, t, period)

        forward_capacity_t = forward_capacity
        cong1 = 1 if queue_capacity == 0 else 0
        cong2 = 1 if forward_exceed > 0 else 0
        congestione = 1 if cong1 == 1 or cong2 == 1 else 0
        
        if congestione == 0:
            congestione_zero_count += 1
        elif congestione == 1:
            congestione_one_count += 1
        
        t += 1
        if t == 500:
            done = True
        else:
            done = False

        return queue_capacity, queue_shares, t, done, forward_capacity, forward_capacity_t, cong1, cong2, congestione, congestione_zero_count, congestione_one_count, input_requests
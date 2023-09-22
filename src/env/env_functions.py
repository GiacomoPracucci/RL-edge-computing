import numpy as np
from env.workload_management import workload
seed = 0

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

# REWARD FUNCTION
def calculate_reward1(local, forwarded, rejected, QUEUE_factor, FORWARD_factor, cong1, cong2, forward_exceed):
    
    if cong1 == 0 and cong2 == 0: 
        reward_local = 3 * local * QUEUE_factor
        reward_forwarded = 1 * forwarded * (1 - QUEUE_factor) * FORWARD_factor
        reward_rejected = -10 * rejected * FORWARD_factor * QUEUE_factor
        reward = reward_local + reward_forwarded + reward_rejected - 2 * forward_exceed
    else:
        reward_local = -10 * local
        reward_forwarded = -2 * forwarded * FORWARD_factor
        reward_rejected = 2 * rejected * (1 - FORWARD_factor)
        reward = reward_local + reward_forwarded + reward_rejected - 500 - 2 * forward_exceed

    return reward


#@staticmethod
def update_obs_space(scenario, average_requests, amplitude_requests, queue_workload, queue_capacity, max_queue_capacity, t,
                     forward_capacity, forward_capacity_t, period, cong1, cong2,
                     forward_exceed, congestione_zero_count, congestione_one_count):

        #print(f"Num requests in queue: {len(queue_workload)}")
        queue_length_requests = len(queue_workload)
        queue_capacity = max(0, max_queue_capacity - queue_length_requests)
        queue_shares = sum(request['shares'] for request in queue_workload)
        queue_mb = sum(request['dfaas_mb'] for request in queue_workload)
        
        if scenario == "scenario1":
            input_requests, forward_capacity = workload.scenario1(average_requests=average_requests)
        elif scenario == "scenario2":
            input_requests, forward_capacity = workload.scenario2(t, period, average_requests=average_requests, amplitude_requests=amplitude_requests)
        elif scenario == "scenario3":
            input_requests, forward_capacity = workload.scenario3(average_requests, amplitude_requests, t, period)

        forward_capacity_t = forward_capacity
        cong1 = 1 if queue_capacity == 0 else 0
        cong2 = 1 if forward_exceed > 0 else 0
        
        if cong1 == 0 and cong2 == 0:
            congestione_zero_count += 1
        else:
            congestione_one_count += 1
        
        t += 1
        if t == 100:
            done = True
        else:
            done = False

        return queue_capacity, queue_shares, t, done, forward_capacity, forward_capacity_t, cong1, cong2, congestione_zero_count, congestione_one_count, input_requests



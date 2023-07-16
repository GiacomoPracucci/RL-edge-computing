import numpy as np

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

# FUNCTION TO MANAGE THE CPU CAPACITY AND BUFFER QUEUE
def sample_workload(local):
    workload = []
    for i in range(local):
        sample = np.random.uniform()
        if sample < 0.33:
            request_class = 'A'
            mean, std_dev = 5.5, 2.5
            shares = np.clip(np.random.normal(mean, std_dev), 1, 10)  
        elif sample < 0.67:
            request_class = 'B'
            mean, std_dev = 15.5, 2.5 
            shares = np.clip(np.random.normal(mean, std_dev), 11, 20)
        else:
            request_class = 'C'
            mean, std_dev = 25.5, 2.5
            shares = np.clip(np.random.normal(mean, std_dev), 21, 30)
        workload.append({'class': request_class, 'shares': shares, 'position': i})
    return workload

# REWARD FUNCTION
def calculate_reward1(local, forwarded, rejected, QUEUE_factor, FORWARD_factor, congestione):
    
    if congestione == 0:
        reward_local = 3 * local * QUEUE_factor
        reward_forwarded = 1 * forwarded * (1 - QUEUE_factor) * FORWARD_factor
        reward_rejected = -10 * rejected * FORWARD_factor * QUEUE_factor
        reward = reward_local + reward_forwarded + reward_rejected
    else:
        reward_local = -10 * local
        reward_forwarded = 2 * forwarded * FORWARD_factor
        reward_rejected = 2 * rejected * (1 - FORWARD_factor)
        reward = reward_local + reward_forwarded + reward_rejected - 200

    return reward
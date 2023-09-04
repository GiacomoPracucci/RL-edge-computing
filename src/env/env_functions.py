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

# REWARD FUNCTION
def calculate_reward2(local, forwarded, rejected, QUEUE_factor, FORWARD_factor, cong1, cong2, forward_exceed):
    
    reward_local = 3 * local * QUEUE_factor
    reward_forwarded = 1 * forwarded * (1 - QUEUE_factor) * FORWARD_factor
    reward_rejected = -10 * rejected * FORWARD_factor * QUEUE_factor
    reward = reward_local + reward_forwarded + reward_rejected - 2 * forward_exceed - 500 * (cong1) - 500 * (cong2)

    return reward

# REWARD FUNCTION
def calculate_reward3(local, forwarded, rejected, QUEUE_factor, FORWARD_factor, cong1, cong2, forward_exceed):
    
    reward_local = 3 * local * QUEUE_factor
    reward_forwarded = 1 * forwarded * (1 - QUEUE_factor) * FORWARD_factor
    reward_rejected = -10 * rejected * FORWARD_factor * QUEUE_factor
    reward = reward_local + reward_forwarded + reward_rejected

    return reward




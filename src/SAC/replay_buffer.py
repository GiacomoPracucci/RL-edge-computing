import numpy as np

# REPLAY BUFFER
class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        self.max_size = max_size
        self.state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action_memory = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.next_state_memory = np.zeros((max_size, state_dim), dtype=np.float32)
        self.done_memory = np.zeros(max_size, dtype=np.bool_)
        self.pointer = 0
        self.size = 0

    def store(self, state, action, reward, next_state, done):
        idx = self.pointer % self.max_size
        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.next_state_memory[idx] = next_state
        self.done_memory[idx] = done
        self.pointer += 1
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (self.state_memory[idxs], self.action_memory[idxs], self.reward_memory[idxs],
                self.next_state_memory[idxs], self.done_memory[idxs])
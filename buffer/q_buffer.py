import numpy as np


class iql_buffer:
    def __init__(self, obs_dim, max_size=10000000, batch_size=256):
        self.mem_size = max_size
        self.batch_size = batch_size
        self.mem_cnt = 0

        self.state_memory = np.zeros((self.mem_size, obs_dim))
        self.action_memory = np.zeros((self.mem_size,))
        self.reward_memory = np.zeros((self.mem_size,))
        self.next_state_memory = np.zeros((self.mem_size, obs_dim))
        self.dones_memory = np.zeros((self.mem_size,), dtype=np.bool_)

    def insert(self, data1, data2):
        mem_idx = self.mem_cnt % self.mem_size
        s1, a_1, next_s1, done1, r1 = data1
        s2, a_2, next_s2, done2, r2 = data2
        self.state_memory[mem_idx] = s1
        self.action_memory[mem_idx] = a_1
        # TODO：合作
        self.reward_memory[mem_idx] = r1 + r2
        self.next_state_memory[mem_idx] = next_s1
        self.dones_memory[mem_idx] = done1
        self.mem_cnt += 1
        mem_idx = self.mem_cnt % self.mem_size
        self.state_memory[mem_idx] = s2
        self.action_memory[mem_idx] = a_2
        self.reward_memory[mem_idx] = r1 + r2
        self.next_state_memory[mem_idx] = next_s2
        self.dones_memory[mem_idx] = done2
        self.mem_cnt += 1

    def sample_buffer(self, use_importance_sampling=False):
        mem_len = min(self.mem_size, self.mem_cnt)
        if use_importance_sampling:
            batch = np.random.choice(mem_len, self.batch_size, replace=False,
                                     p=(np.exp(np.abs(self.reward_memory[:mem_len])) / np.exp(
                                         np.abs(self.reward_memory[:mem_len])).sum()).tolist())
        else:
            batch = np.random.choice(mem_len, self.batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.next_state_memory[batch]
        dones = self.dones_memory[batch]

        return states, actions, rewards, states_, dones

    def ready(self):
        return self.mem_cnt > self.batch_size


class vdn_buffer:
    def __init__(self, obs_dim, max_size=10000000, batch_size=256):
        self.mem_size = max_size
        self.batch_size = batch_size
        self.mem_cnt1 = 0
        self.mem_cnt2 = 0
        self.state_memory1 = np.zeros((self.mem_size, obs_dim))
        self.action_memory1 = np.zeros((self.mem_size,))
        self.reward_memory1 = np.zeros((self.mem_size,))
        self.next_state_memory1 = np.zeros((self.mem_size, obs_dim))
        self.dones_memory1 = np.zeros((self.mem_size,), dtype=np.bool_)

        self.state_memory2 = np.zeros_like(self.state_memory1)
        self.action_memory2 = np.zeros_like(self.action_memory1)
        self.reward_memory2 = np.zeros_like(self.reward_memory1)
        self.next_state_memory2 = np.zeros_like(self.next_state_memory1)
        self.dones_memory2 = np.zeros_like(self.dones_memory1)

    def insert(self, data1, data2):
        mem_idx = self.mem_cnt1 % self.mem_size
        s1, a1, next_s1, done1, r1 = data1
        self.state_memory1[mem_idx] = s1
        self.action_memory1[mem_idx] = a1
        self.reward_memory1[mem_idx] = r1
        self.next_state_memory1[mem_idx] = next_s1
        self.dones_memory1[mem_idx] = done1
        self.mem_cnt1 += 1
        mem_idx = self.mem_cnt2 % self.mem_size
        s2, a2, next_s2, done2, r2 = data2
        self.state_memory2[mem_idx] = s2
        self.action_memory2[mem_idx] = a2
        self.reward_memory2[mem_idx] = r2
        self.next_state_memory2[mem_idx] = next_s2
        self.dones_memory2[mem_idx] = done2
        self.mem_cnt2 += 1

    def sample_buffer(self, use_importance_sampling=False):
        mem_len = min(self.mem_size, self.mem_cnt1)

        if use_importance_sampling:
            batch = np.random.choice(mem_len, self.batch_size, replace=False,
                                     p=(np.exp(np.abs(self.reward_memory1[:mem_len])+np.abs(self.reward_memory2[:mem_len])) / np.exp(
                                         np.abs(np.abs(self.reward_memory1[:mem_len])+np.abs(self.reward_memory2[:mem_len]))).sum()).tolist())
        else:
            batch = np.random.choice(mem_len, self.batch_size, replace=False)

        states1 = self.state_memory1[batch]
        actions1 = self.action_memory1[batch]
        rewards1 = self.reward_memory1[batch]
        states_1 = self.next_state_memory1[batch]
        dones1 = self.dones_memory1[batch]

        states2 = self.state_memory2[batch]
        actions2 = self.action_memory2[batch]
        rewards2 = self.reward_memory2[batch]
        states_2 = self.next_state_memory2[batch]
        dones2 = self.dones_memory2[batch]

        return states1, actions1, rewards1, states_1, dones1, states2, actions2, rewards2, states_2, dones2

    def ready(self):
        return self.mem_cnt1 > self.batch_size

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer.q_buffer import iql_buffer
from algorithm.base_model import QNet


class IQL:
    def __init__(self, n_states, n_actions, device, lr=3e-4):
        self.gamma = 0.99
        self.device = device
        self.lr = lr
        self.q_eval = QNet(state_dim=n_states, action_dim=n_actions).to(device)
        self.q_target = QNet(state_dim=n_states, action_dim=n_actions).to(device)
        self.q_optimizer = torch.optim.Adam(self.q_eval.parameters(), lr=self.lr)
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=0.05):
        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

    def take_action(self, observation, isTrain=True):
        state = torch.tensor([observation], dtype=torch.float).to(self.device)
        q = self.q_eval.forward(state)
        if isTrain:
            probs = torch.nn.functional.softmax(q, dim=-1)
            action_dist = torch.distributions.Categorical(probs)  # 构造概率分布
            action = action_dist.sample().item()  # 从概率分布中随机取样 int
        else:
            action = torch.argmax(q).item()
        return action

    def update(self, buffer: iql_buffer):
        if not buffer.ready():
            return None

        states, actions, rewards, next_states, dones = buffer.sample_buffer()
        batch_idx = np.arange(buffer.batch_size)

        states_tensor = torch.tensor(states, dtype=torch.float).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float).to(self.device)

        with torch.no_grad():
            q_ = self.q_eval.forward(next_states_tensor)
            next_actions = torch.argmax(q_, dim=-1)
            q_ = self.q_target.forward(next_states_tensor)
            # q_[dones_tensor] = 0.0
            target = rewards_tensor + self.gamma * q_[batch_idx, next_actions]
        q = self.q_eval.forward(states_tensor)[batch_idx, actions]

        loss = F.mse_loss(q, target.detach())
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        train_info = {
            'q_loss': loss.detach().cpu().item()
        }

        self.update_network_parameters()
        return train_info

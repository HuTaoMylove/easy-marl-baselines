import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer.q_buffer import vdn_buffer
from algorithm.base_model import QNet


class VDN:
    def __init__(self, n_states, n_actions, device, use_importance_sampling=True, lr=3e-4, epsilon=1.0, eps_end=0.01,
                 eps_dec=5e-4):
        self.gamma = 0.99
        self.device = device
        self.lr = lr
        self.q_eval = QNet(state_dim=n_states, action_dim=n_actions).to(device)
        self.q_target = QNet(state_dim=n_states, action_dim=n_actions).to(device)
        self.q_optimizer = torch.optim.Adam(self.q_eval.parameters(), lr=self.lr)
        self.update_network_parameters(tau=1)
        self.use_importance_sampling = use_importance_sampling
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.n_actions=n_actions

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def update_network_parameters(self, tau=0.9):
        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

    def take_action(self, observation, isTrain=True):
        state = torch.tensor([observation], dtype=torch.float).to(self.device)
        q = self.q_eval.forward(state)
        action = torch.argmax(q).item()

        if (np.random.random() < self.epsilon) and isTrain:
            action = np.random.choice(self.n_actions)

        return action

    def update(self, buffer: vdn_buffer):
        if not buffer.ready():
            return None

        states1, actions1, rewards1, states_1, dones1, states2, actions2, rewards2, states_2, dones2 = buffer.sample_buffer(self.use_importance_sampling)
        batch_idx = np.arange(buffer.batch_size)

        states_tensor1 = torch.tensor(states1, dtype=torch.float).to(self.device)
        rewards_tensor1 = torch.tensor(rewards1, dtype=torch.float).to(self.device)
        next_states_tensor1 = torch.tensor(states_1, dtype=torch.float).to(self.device)
        states_tensor2 = torch.tensor(states2, dtype=torch.float).to(self.device)
        rewards_tensor2 = torch.tensor(rewards2, dtype=torch.float).to(self.device)
        next_states_tensor2 = torch.tensor(states_2, dtype=torch.float).to(self.device)

        q_evals1 = self.q_eval.forward(states_tensor1)[batch_idx, actions1]
        q_evals2 = self.q_eval.forward(states_tensor2)[batch_idx, actions2]
        q_targets1 = self.q_eval.forward(next_states_tensor1).max(dim=-1)[0]
        q_targets2 = self.q_eval.forward(next_states_tensor2).max(dim=-1)[0]

        loss = F.mse_loss(q_evals1 + q_evals2,
                          rewards_tensor1 + rewards_tensor2 + self.gamma * (q_targets1.detach() + q_targets2.detach()))
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        train_info = {
            'q_loss': loss.detach().cpu().item()
        }

        self.update_network_parameters()
        self.decrement_epsilon()
        return train_info

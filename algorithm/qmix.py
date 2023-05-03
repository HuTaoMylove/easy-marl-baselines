import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from buffer.q_buffer import vdn_buffer
from algorithm.base_model import QNet
import itertools
from utils import init


class QMixNet(nn.Module):

    def __init__(self, state_dim, n_agents=2, hyper_hidden_dim=32, qmix_hidden_dim=32):
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.hyper_hidden_dim = hyper_hidden_dim
        self.qmix_hidden_dim = qmix_hidden_dim
        self.hyper_w1 = nn.Sequential(self.init_(nn.Linear(n_agents * state_dim, hyper_hidden_dim)),
                                      nn.Tanh(),
                                      self.init_(nn.Linear(hyper_hidden_dim, n_agents * qmix_hidden_dim)))

        # hyper_w2 生成推理网络需要的从隐层到输出 Q 值的所有 weights，共 qmix_hidden 个
        self.hyper_w2 = nn.Sequential(self.init_(nn.Linear(n_agents * state_dim, hyper_hidden_dim)),
                                      nn.Tanh(),
                                      self.init_(nn.Linear(hyper_hidden_dim, qmix_hidden_dim)))

        # hyper_b1 生成第一层网络对应维度的偏差 bias
        self.hyper_b1 = self.init_(nn.Linear(n_agents * state_dim, qmix_hidden_dim))
        # hyper_b2 生成对应从隐层到输出 Q 值层的 bias
        self.hyper_b2 = nn.Sequential(self.init_(nn.Linear(n_agents * state_dim, qmix_hidden_dim)),
                                      nn.Tanh(),
                                      self.init_(nn.Linear(qmix_hidden_dim, 1))
                                      )

    def init_(self, m):
        return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), 0.1)

    def forward(self, q_values, states):  # states的shape为(episode_num, max_episode_len， state_shape)

        w1 = torch.abs(self.hyper_w1(states))
        b1 = self.hyper_b1(states)

        w1 = w1.view(-1, self.n_agents, self.qmix_hidden_dim)
        b1 = b1.view(-1, 1, self.qmix_hidden_dim)

        hidden = F.elu(torch.bmm(q_values.unsqueeze(1), w1) + b1)  # torch.bmm(a, b) 计算矩阵 a 和矩阵 b 相乘

        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(-1, self.qmix_hidden_dim, 1)
        b2 = b2.view(-1, 1, 1)

        q_total = torch.bmm(hidden, w2) + b2
        q_total = q_total.squeeze(1).squeeze(1)
        return q_total


class QMIX:
    def __init__(self, n_states, n_actions, device, lr=3e-4,epsilon=1.0, eps_end=0.01,
                 eps_dec=5e-4):
        self.gamma = 0.99
        self.device = device
        self.lr = lr
        self.q_eval = QNet(state_dim=n_states, action_dim=n_actions).to(device)
        self.q_target = QNet(state_dim=n_states, action_dim=n_actions).to(device)
        self.eval_Qmix = QMixNet(state_dim=n_states)
        self.target_Qmix = QMixNet(state_dim=n_states)
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.n_actions=n_actions
        self.q_optimizer = torch.optim.Adam(itertools.chain(self.q_eval.parameters(), self.eval_Qmix.parameters()),
                                            lr=self.lr)
        self.update_network_parameters(tau=1)
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def update_network_parameters(self, tau=0.9):
        for q_target_params, q_eval_params in zip(self.q_target.parameters(), self.q_eval.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)
        for q_target_params, q_eval_params in zip(self.target_Qmix.parameters(), self.eval_Qmix.parameters()):
            q_target_params.data.copy_(tau * q_eval_params + (1 - tau) * q_target_params)

    def take_action(self, observation, isTrain=True):
        state = torch.tensor([observation], dtype=torch.float).to(self.device)
        q = self.q_eval.forward(state)
        if (np.random.random() < self.epsilon) and isTrain:
            action = np.random.choice(self.n_actions)
        else:
            action = torch.argmax(q).item()
        return action

    def update(self, buffer: vdn_buffer):
        if not buffer.ready():
            return None

        states1, actions1, rewards1, states_1, dones1, states2, actions2, rewards2, states_2, dones2 = buffer.sample_buffer()
        batch_idx = np.arange(buffer.batch_size)

        states_tensor1 = torch.tensor(states1, dtype=torch.float).to(self.device)
        rewards_tensor1 = torch.tensor(rewards1, dtype=torch.float).to(self.device)
        next_states_tensor1 = torch.tensor(states_1, dtype=torch.float).to(self.device)
        states_tensor2 = torch.tensor(states2, dtype=torch.float).to(self.device)
        rewards_tensor2 = torch.tensor(rewards2, dtype=torch.float).to(self.device)
        next_states_tensor2 = torch.tensor(states_2, dtype=torch.float).to(self.device)

        q_evals1 = self.q_eval.forward(states_tensor1)[batch_idx, actions1].reshape(-1, 1)
        q_evals2 = self.q_eval.forward(states_tensor2)[batch_idx, actions2].reshape(-1, 1)
        q_targets1 = self.q_eval.forward(next_states_tensor1).max(dim=-1)[0].reshape(-1, 1).detach()
        q_targets2 = self.q_eval.forward(next_states_tensor2).max(dim=-1)[0].reshape(-1, 1).detach()
        q_total_eval = self.eval_Qmix(torch.concat([q_evals1, q_evals2], dim=1),
                                      torch.concat([states_tensor1, states_tensor2], dim=1))
        q_total_target = self.target_Qmix(torch.concat([q_targets1, q_targets2], dim=1),
                                          torch.concat([next_states_tensor1, next_states_tensor2], dim=1)).detach()

        targets = rewards_tensor1 + rewards_tensor2 + self.gamma * q_total_target
        loss = F.mse_loss(q_total_eval, targets.detach())

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        train_info = {
            'q_loss': loss.detach().cpu().item()
        }

        self.update_network_parameters()
        self.decrement_epsilon()
        return train_info

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from utils import init

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as f


class PolicyNet(nn.Module):  # 输入当前状态，输出动作的概率分布
    def init_(self, m):
        return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), 0.1)

    def __init__(self, n_states, n_hiddens, n_actions, use_rnn=False):
        super(PolicyNet, self).__init__()
        self.fc1 = self.init_(nn.Linear(n_states, n_hiddens))
        self.fc2 = self.init_(nn.Linear(n_hiddens, n_hiddens))
        self.fc3 = self.init_(nn.Linear(n_hiddens, n_actions))

    def forward(self, x):  # [b,n_states]
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.tanh(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_hiddens]
        x = F.tanh(x)
        x = self.fc3(x)  # [b,n_hiddens]-->[b,n_actions]
        x = F.softmax(x, dim=1)  # 每种动作选择的概率
        return x


class ValueNet(nn.Module):  # 评价当前状态的价值
    def init_(self, m):
        return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), 0.1)

    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.fc1 = self.init_(nn.Linear(n_states, n_hiddens))
        self.fc2 = self.init_(nn.Linear(n_hiddens, n_hiddens))
        self.fc3 = self.init_(nn.Linear(n_hiddens, 1))

    def forward(self, x):  # [b,n_states]
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc3(x)  # [b,n_hiddens]-->[b,1]
        return x


class RNN(nn.Module):

    def __init__(self, n_states, n_outputs, n_hiddens=128, use_softmax=False):
        super(RNN, self).__init__()
        self.n_hiddens = n_hiddens
        self.use_softmax = use_softmax
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.rnn = nn.GRUCell(n_hiddens, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_outputs)

    def init_(self, m):
        return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), 0.1)

    def forward(self, obs, hidden_state, done=None):
        if done is None:
            x = f.tanh(self.fc1(obs))
            assert hidden_state.shape[0] == x.shape[0]
            hidden_state = self.rnn(x, hidden_state)
            output = self.fc2(hidden_state)
            if self.use_softmax:
                output = torch.nn.functional.softmax(output, dim=-1)
        else:
            output = []
            for obs, mask in zip(obs, done):
                obs=obs.unsqueeze(0)
                x = f.tanh(self.fc1(obs))
                assert hidden_state.shape[0] == x.shape[0]
                hidden_state = self.rnn(x, hidden_state)
                q = self.fc2(hidden_state)
                hidden_state = hidden_state*(1 - mask)
                if self.use_softmax:
                    q = torch.nn.functional.softmax(q, dim=-1)
                output.append(q)
            output = torch.cat(output, dim=0)
        return output, hidden_state


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNet, self).__init__()

        self.fc1 = self.init_(nn.Linear(state_dim, hidden_dim))
        self.fc2 = self.init_(nn.Linear(hidden_dim, hidden_dim))
        self.q = self.init_(nn.Linear(hidden_dim, action_dim))

    def init_(self, m):
        return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), 0.1)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        q = self.q(x)

        return q

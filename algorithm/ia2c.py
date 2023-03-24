import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from utils import init


# ----------------------------------------- #
# 策略网络--actor
# ----------------------------------------- #

class PolicyNet(nn.Module):  # 输入当前状态，输出动作的概率分布
    def init_(self, m):
        return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), 0.1)

    def __init__(self, n_states, n_hiddens, n_actions):
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


class A2C:
    def __init__(self, n_states, n_actions, device, n_hiddens=128, actor_lr=3e-4, critic_lr=1e-3):
        # 属性分配
        self.n_hiddens = n_hiddens
        self.actor_lr = actor_lr  # 策略网络的学习率
        self.critic_lr = critic_lr  # 价值网络的学习率
        self.lmbda = 0.97  # 优势函数的缩放因子
        self.eps = 0.2  # ppo截断范围缩放因子
        self.gamma = 0.9  # 折扣因子
        self.device = device
        # 网络实例化
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)  # 策略网络
        self.critic = ValueNet(n_states, n_hiddens).to(device)  # 价值网络
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    # 动作选择
    def take_action(self, state, max_act=False):  # [n_states]
        state = torch.tensor([state], dtype=torch.float).to(self.device)  # [1,n_states]
        probs = self.actor(state)  # 当前状态的动作概率 [b,n_actions]
        action_dist = torch.distributions.Categorical(probs)  # 构造概率分布
        action = action_dist.sample().item()  # 从概率分布中随机取样 int
        return action

    # 训练
    def update(self, transition_dict, transition_dict1):

        # 取出数据集
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)  # [b,n_states]
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)  # [b,1]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)  # [b,n_states]
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)  # [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)  # [b,1]

        if transition_dict1 is not None:
            states1 = torch.tensor(transition_dict1['states'], dtype=torch.float).to(self.device)  # [b,n_states]
            actions1 = torch.tensor(transition_dict1['actions']).view(-1, 1).to(self.device)  # [b,1]
            next_states1 = torch.tensor(transition_dict1['next_states'], dtype=torch.float).to(
                self.device)  # [b,n_states]
            dones1 = torch.tensor(transition_dict1['dones'], dtype=torch.float).view(-1, 1).to(self.device)  # [b,1]
            rewards1 = torch.tensor(transition_dict1['rewards'], dtype=torch.float).view(-1, 1).to(self.device)  # [b,1]
            states = torch.concat([states, states1], dim=0)
            actions = torch.concat([actions, actions1], dim=0)
            next_states = torch.concat([next_states, next_states1], dim=0)
            dones = torch.concat([dones, dones1], dim=0)
            rewards = torch.concat([rewards, rewards1], dim=0)

        with torch.no_grad():
            next_state_value = self.critic(next_states)  # 下一时刻的state_value  [b,1]
            td_target = rewards + self.gamma * next_state_value * (1 - dones)  # 目标--当前时刻的state_value  [b,1]
            td_value = self.critic(states)  # 预测--当前时刻的state_value  [b,1]
            td_delta = td_target - td_value  # 时序差分  # [b,1]

            # 计算GAE优势函数，当前状态下某动作相对于平均的优势
            advantage = 0  # 累计一个序列上的优势函数
            advantage_list = []  # 存放每个时序的优势函数值
            td_delta = td_delta.cpu().detach().numpy()  # gpu-->numpy
            for delta in td_delta[::-1]:  # 逆序取出时序差分值
                advantage = self.gamma * self.lmbda * advantage + delta
                advantage_list.append(advantage)  # 保存每个时刻的优势函数
            advantage_list.reverse()  # 正序
            advantage = torch.tensor(np.array(advantage_list), dtype=torch.float).to(self.device)
            old_log_probs = torch.log(self.actor(states).gather(1, actions))  # [b,1]

        probs = self.actor(states)
        log_prob = torch.log(probs.gather(1, actions))
        entropy = torch.distributions.Categorical(probs).entropy().mean()
        ratio = torch.exp(log_prob - old_log_probs)

        # clip截断
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
        td_value = self.critic(states)
        # 损失计算
        actor_loss = torch.mean(-torch.min(surr1, surr2)) - entropy * 0.1
        critic_loss = torch.mean(F.mse_loss(td_value, td_target))
        # 梯度更新
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        train_info = {
            "ratio": ratio.detach().mean().cpu().item(),
            "actor_loss": actor_loss.detach().cpu().item(),
            "critic_loss": critic_loss.detach().cpu().item(),
            'entropy': entropy.detach().cpu().item()
        }

        return train_info

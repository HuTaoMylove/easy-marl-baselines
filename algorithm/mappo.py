import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from utils import init

from torch.utils.data import Dataset, DataLoader
from algorithm.base_model import PolicyNet, ValueNet, RNN


class MappoDataset(Dataset):
    def __init__(self, states_a, states_v, actions, old_log_probs, advantage, td_target):
        self.states_a = states_a
        self.states_v = states_v
        self.actions = actions
        self.old_log_probs = old_log_probs
        self.advantage = advantage
        self.td_target = td_target

    def __getitem__(self, index):
        """ 必须实现，作用是:获取索引对应位置的一条数据 :param index: :return: """
        return self.states_a[index], self.states_v[index], self.actions[index], self.old_log_probs[index], \
               self.advantage[index], \
               self.td_target[index],

    def __len__(self):
        """ 必须实现，作用是得到数据集的大小 :return: """
        return len(self.states_v)


class r_MappoDataset(Dataset):
    def __init__(self, obs_a, obs_v, actions, action_log_prob, advantage, td_target, done, chunklength=8):
        self.states_a = obs_a
        self.states_v = obs_v
        self.actions = actions
        self.old_log_probs = action_log_prob
        self.advantage = advantage
        self.td_target = td_target
        self.done = done
        self.chunklength = chunklength

    def __getitem__(self, index):
        return self.states_a[index:index + self.chunklength], self.states_v[index:index + self.chunklength], \
               self.actions[index:index + self.chunklength], self.old_log_probs[
                                                             index:index + self.chunklength], self.advantage[
                                                                                              index:index + self.chunklength], \
               self.td_target[index:index + self.chunklength], self.done[index:index + self.chunklength]

    def __len__(self):
        return len(self.states_a) - self.chunklength


class MAPPO:
    def __init__(self, n_states, n_actions, device, n_agent=2, n_hiddens=128, actor_lr=3e-4, critic_lr=1e-3):
        # 属性分配
        self.n_hiddens = n_hiddens
        self.actor_lr = actor_lr  # 策略网络的学习率
        self.critic_lr = critic_lr  # 价值网络的学习率
        self.lmbda = 0.97  # 优势函数的缩放因子
        self.eps = 0.2  # ppo截断范围缩放因子
        self.gamma = 0.9  # 折扣因子
        self.n_agent = n_agent
        self.device = device
        # 网络实例化
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)  # 策略网络
        self.critic = ValueNet(self.n_agent * n_states, n_hiddens).to(device)  # 价值网络
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    # 动作选择
    def take_action(self, obs, isTrain=True):  # [n_states]
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)  # [n_agents,n_states]
            probs = self.actor(obs)  # 当前状态的动作概率 [b,n_actions]
            action_dist = torch.distributions.Categorical(probs)  # 构造概率分布
            if isTrain:
                action = action_dist.sample()  # 从概率分布中随机取样 int
            else:
                action = torch.argmax(probs, dim=-1)
            action_log_prob = action_dist.log_prob(action)
            obs_value = self.critic(torch.cat(torch.split(obs, 1, dim=0), dim=1))

            return action.reshape(self.n_agent, -1), action_log_prob.reshape(self.n_agent, -1), \
                   torch.cat([obs_value] * self.n_agent, dim=0).reshape(self.n_agent, -1)

    def reset(self):
        return

    # 训练
    def update(self, data):

        # 取出数据集
        obs, next_obs, obs_value, action, action_log_prob, reward, done, eposide_idx = data

        obs_a = obs.permute(1, 0, 2).reshape(-1, obs.shape[-1]).to(self.device)
        obs_v = torch.cat(torch.split(obs.permute(1, 0, 2), 1, dim=0), dim=-1).squeeze(0)
        obs_v = torch.cat([obs_v] * self.n_agent, dim=0).to(self.device)

        next_obs_v = torch.cat(torch.split(next_obs.permute(1, 0, 2), 1, dim=0), dim=-1).squeeze(0)
        next_obs_v = torch.cat([next_obs_v] * self.n_agent, dim=0).to(self.device)

        obs_value = obs_value.permute(1, 0, 2).reshape(-1, obs_value.shape[-1]).to(self.device)
        action = action.permute(1, 0, 2).reshape(-1, action.shape[-1]).to(self.device)
        action_log_prob = action_log_prob.permute(1, 0, 2).reshape(-1, action_log_prob.shape[-1]).to(self.device)

        # TODO:合作
        reward = reward.sum(dim=1).reshape(-1, 1)
        reward = torch.concat([reward] * self.n_agent).to(self.device)

        done = done.permute(1, 0, 2).reshape(-1, done.shape[-1]).to(self.device)

        with torch.no_grad():
            next_state_value = self.critic(next_obs_v)  # 下一时刻的state_value  [b,1]
            td_target = reward + self.gamma * next_state_value * (1 - done)  # 目标--当前时刻的state_value  [b,1]
            td_delta = td_target - obs_value  # 时序差分  # [b,1]

            # 计算GAE优势函数，当前状态下某动作相对于平均的优势
            advantage = 0  # 累计一个序列上的优势函数
            advantage_list = []  # 存放每个时序的优势函数值
            td_delta = td_delta.cpu().detach().numpy()  # gpu-->numpy
            for delta in td_delta[::-1]:  # 逆序取出时序差分值
                advantage = self.gamma * self.lmbda * advantage + delta
                advantage_list.append(advantage)  # 保存每个时刻的优势函数
            advantage_list.reverse()  # 正序
            advantage = torch.tensor(np.array(advantage_list), dtype=torch.float).to(self.device)

        dataset = MappoDataset(obs_a, obs_v, action, action_log_prob, advantage, td_target)
        dataloader = DataLoader(dataset=dataset, batch_size=100, shuffle=True, drop_last=False)
        for i in range(3):
            for step, (states_a, states_v, actions, old_log_probs, advantage, td_target) in enumerate(dataloader):
                probs = self.actor(states_a)
                log_prob = torch.log(probs.gather(1, actions))
                entropy = torch.distributions.Categorical(probs).entropy().mean()
                ratio = torch.exp(log_prob - old_log_probs)

                # clip截断
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
                td_value = self.critic(states_v)
                # 损失计算
                actor_loss = torch.mean(-torch.min(surr1, surr2)) - entropy * 0.01
                critic_loss = torch.mean(F.mse_loss(td_value, td_target))
                # 梯度更新
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        train_info = {
            "ratio": ratio.detach().mean().cpu().item(),
            "actor_loss": actor_loss.detach().cpu().item(),
            "critic_loss": critic_loss.detach().cpu().item(),
            'entropy': entropy.detach().cpu().item()
        }

        return train_info


class r_MAPPO:
    def __init__(self, n_states, n_actions, device, n_agent=2, n_hiddens=128, actor_lr=3e-4, critic_lr=1e-3):
        # 属性分配
        self.n_hiddens = n_hiddens
        self.actor_lr = actor_lr  # 策略网络的学习率
        self.critic_lr = critic_lr  # 价值网络的学习率
        self.lmbda = 0.97  # 优势函数的缩放因子
        self.eps = 0.2  # ppo截断范围缩放因子
        self.gamma = 0.9  # 折扣因子
        self.n_agent = n_agent
        self.device = device
        self.batchsize = 50
        # 网络实例化
        self.actor = RNN(n_states, n_actions, n_hiddens, use_softmax=True).to(device)  # 策略网络
        self.critic = RNN(n_states * self.n_agent, 1, n_hiddens).to(device)  # 价值网络
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    # 动作选择
    def take_action(self, obs, isTrain=True):  # [n_states]
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float).to(self.device)  # [n_agents,n_states]
            assert self.n_agent == obs.shape[0]
            probs, self.actor_r_hidden = self.actor(obs, self.actor_r_hidden)  # 当前状态的动作概率 [b,n_actions]
            action_dist = torch.distributions.Categorical(probs)  # 构造概率分布
            if isTrain:
                action = action_dist.sample()  # 从概率分布中随机取样 int
            else:
                action = torch.argmax(probs, dim=-1)
            action_log_prob = action_dist.log_prob(action)
            obs_value, self.critic_r_hidden = self.critic(torch.cat([obs.reshape(1, -1)] * self.n_agent, dim=0),
                                                          self.critic_r_hidden)
            return action.reshape(self.n_agent, -1), action_log_prob.reshape(self.n_agent, -1), obs_value.reshape(
                self.n_agent, -1)

    def reset(self):
        self.actor_r_hidden = torch.zeros([self.n_agent, self.n_hiddens]).to(self.device)
        self.critic_r_hidden = torch.zeros([self.n_agent, self.n_hiddens]).to(self.device)

    # 训练
    def update(self, data):

        # 取出数据集
        obs, next_obs, obs_value, action, action_log_prob, reward, done, eposide_idx = data

        obs_a = obs.permute(1, 0, 2).reshape(-1, obs.shape[-1]).to(self.device)
        obs_v = torch.cat(torch.split(obs.permute(1, 0, 2), 1, dim=0), dim=-1).squeeze(0)
        obs_v = torch.cat([obs_v] * self.n_agent, dim=0).to(self.device)

        next_obs_v = torch.cat(torch.split(next_obs.permute(1, 0, 2), 1, dim=0), dim=-1).squeeze(0)
        next_obs_v = torch.cat([next_obs_v] * self.n_agent, dim=0).to(self.device)

        obs_value = obs_value.permute(1, 0, 2).reshape(-1, obs_value.shape[-1]).to(self.device)
        action = action.permute(1, 0, 2).reshape(-1, action.shape[-1]).to(self.device)
        action_log_prob = action_log_prob.permute(1, 0, 2).reshape(-1, action_log_prob.shape[-1]).to(self.device)

        # TODO:合作
        reward = reward.sum(dim=1).reshape(-1, 1)
        reward = torch.concat([reward] * self.n_agent).to(self.device)

        done = done.permute(1, 0, 2).reshape(-1, done.shape[-1]).to(self.device)

        with torch.no_grad():
            next_obs_value, _ = self.critic(next_obs_v, torch.zeros([1, self.n_hiddens]).to(self.device), done)
            td_target = reward + self.gamma * next_obs_value * (1 - done)  # 目标--当前时刻的state_value  [b,1]
            td_delta = td_target - obs_value  # 时序差分  # [b,1]

            # 计算GAE优势函数，当前状态下某动作相对于平均的优势
            advantage = 0  # 累计一个序列上的优势函数
            advantage_list = []  # 存放每个时序的优势函数值
            td_delta = td_delta.cpu().detach().numpy()  # gpu-->numpy
            for delta in td_delta[::-1]:  # 逆序取出时序差分值
                advantage = self.gamma * self.lmbda * advantage + delta
                advantage_list.append(advantage)  # 保存每个时刻的优势函数
            advantage_list.reverse()  # 正序
            advantage = torch.tensor(np.array(advantage_list), dtype=torch.float).to(self.device)

        dataset = r_MappoDataset(obs_a, obs_v, action, action_log_prob, advantage, td_target, done, chunklength=8)
        dataloader = DataLoader(dataset=dataset, batch_size=50, shuffle=True, drop_last=False)
        for i in range(3):
            for step, (states_a, states_v, actions, old_log_probs, advantage, td_target, done) in enumerate(dataloader):
                states_a = states_a.reshape(-1, states_a.shape[-1])
                states_v = states_v.reshape(-1, states_v.shape[-1])
                actions = actions.reshape(-1, actions.shape[-1])
                old_log_probs = old_log_probs.reshape(-1, old_log_probs.shape[-1])
                advantage = advantage.reshape(-1, advantage.shape[-1])
                td_target = td_target.reshape(-1, td_target.shape[-1])
                done = done.squeeze(dim=-1)
                done[:, -1] = 1
                done = done.unsqueeze(dim=-1).reshape(-1, 1)

                probs, _ = self.actor(states_a, torch.zeros([1, self.n_hiddens]), done)
                log_prob = torch.log(probs.gather(1, actions))
                entropy = torch.distributions.Categorical(probs).entropy().mean()
                ratio = torch.exp(log_prob - old_log_probs)
                # clip截断
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
                td_value, _ = self.critic(states_v, torch.zeros([1, self.n_hiddens]), done)
                # 损失计算
                actor_loss = torch.mean(-torch.min(surr1, surr2)) - entropy * 0.01
                critic_loss = torch.mean(F.mse_loss(td_value, td_target))
                # 梯度更新
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        train_info = {
            "ratio": ratio.detach().mean().cpu().item(),
            "actor_loss": actor_loss.detach().cpu().item(),
            "critic_loss": critic_loss.detach().cpu().item(),
            'entropy': entropy.detach().cpu().item()
        }

        return train_info

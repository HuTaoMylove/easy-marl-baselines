import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from utils import init
from algorithm.base_model import PolicyNet, ValueNet


# ----------------------------------------- #
# 策略网络--actor
# ----------------------------------------- #
class A2C:
    def __init__(self, n_states, n_actions, device, n_agent=2, n_hiddens=128, actor_lr=3e-4, critic_lr=1e-3):
        # 属性分配
        self.n_hiddens = n_hiddens
        self.actor_lr = actor_lr  # 策略网络的学习率
        self.critic_lr = critic_lr  # 价值网络的学习率
        self.lmbda = 0.97  # 优势函数的缩放因子
        self.eps = 0.2  # ppo截断范围缩放因子
        self.gamma = 0.9  # 折扣因子
        self.device = device
        self.n_agent = n_agent
        # 网络实例化
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)  # 策略网络
        self.critic = ValueNet(n_states, n_hiddens).to(device)  # 价值网络
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
            obs_value = self.critic(obs)
            return action.reshape(self.n_agent, -1), action_log_prob.reshape(self.n_agent, -1), obs_value.reshape(self.n_agent, -1)

    # 训练
    def update(self, data):
        # shape [buffersize,n_agent,dim]
        obs, next_obs, obs_value, action, action_log_prob, reward, done, eposide_idx = data

        obs = obs.permute(1, 0, 2).reshape(-1, obs.shape[-1]).to(self.device)
        next_obs = next_obs.permute(1, 0, 2).reshape(-1, next_obs.shape[-1]).to(self.device)
        obs_value = obs_value.permute(1, 0, 2).reshape(-1, obs_value.shape[-1]).to(self.device)
        action = action.permute(1, 0, 2).reshape(-1, action.shape[-1]).to(self.device)
        action_log_prob = action_log_prob.permute(1, 0, 2).reshape(-1, action_log_prob.shape[-1]).to(self.device)

        # TODO:合作
        reward = reward.sum(dim=1).reshape(-1, 1)
        reward = torch.concat([reward]*self.n_agent).to(self.device)

        done = done.permute(1, 0, 2).reshape(-1, done.shape[-1]).to(self.device)

        with torch.no_grad():
            next_obs_value = self.critic(next_obs)  # 下一时刻的state_value  [b,1]
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

        probs = self.actor(obs)
        log_prob = torch.log(probs.gather(1, action))
        entropy = torch.distributions.Categorical(probs).entropy().mean()
        ratio = torch.exp(log_prob - action_log_prob)
        # ratio==1 没啥用

        # clip截断
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
        td_value = self.critic(obs)
        # 损失计算
        actor_loss = torch.mean(-torch.min(surr1, surr2)) - entropy * 0.1
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

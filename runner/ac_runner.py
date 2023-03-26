import os.path

import torch
from algorithm.ippo import PPO, r_PPO
from algorithm.ia2c import A2C
from algorithm.mappo import MAPPO
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class ac_runner:
    def __init__(self, env, agent, writer, num_update=10000, num_episodes=50, algo='ippo'):
        self.env = env
        self.agent = agent
        self.writer = writer
        self.num_update = num_update
        self.num_episodes = num_episodes
        self.n_agents = getattr(self.env, 'n_agents', 2)
        self.max_buffer_size = 5000
        self.obs_dim = env.observation_space[0].shape[0]  # 状态数
        self.act_dim = env.action_space[0].n  # 动作数
        self.algo = algo

    def reset(self):
        self.obs = torch.zeros([self.max_buffer_size, self.n_agents, self.obs_dim], dtype=torch.float)
        self.next_obs = torch.zeros([self.max_buffer_size, self.n_agents, self.obs_dim], dtype=torch.float)
        self.obs_value = torch.zeros([self.max_buffer_size, self.n_agents, 1], dtype=torch.float)
        self.action = torch.zeros([self.max_buffer_size, self.n_agents, 1], dtype=torch.int64)
        self.action_log_prob = torch.zeros([self.max_buffer_size, self.n_agents, 1], dtype=torch.float)
        self.reward = torch.zeros([self.max_buffer_size, self.n_agents, 1], dtype=torch.float)
        self.done = torch.zeros([self.max_buffer_size, self.n_agents, 1], dtype=torch.float)
        self.eposide_idx = [0]
        self.idx = 0

    def insert(self, obs, next_obs, obs_value, action, action_log_prob, reward, done):
        action = torch.tensor(action, dtype=torch.float)
        action_log_prob = torch.tensor(action_log_prob, dtype=torch.float)
        obs_value = torch.tensor(obs_value, dtype=torch.float)
        obs = torch.tensor(obs, dtype=torch.float)
        next_obs = torch.tensor(next_obs, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float).reshape(-1, 1)
        done = torch.tensor(done, dtype=torch.bool).reshape(-1, 1)

        self.obs[self.idx] = obs
        self.next_obs[self.idx] = next_obs
        self.obs_value[self.idx] = obs_value
        self.action[self.idx] = action
        self.action_log_prob[self.idx] = action_log_prob
        self.reward[self.idx] = reward
        self.done[self.idx] = done

        self.idx += 1
        if done.all():
            self.eposide_idx.append(self.idx)

    def get_data(self):
        data = self.obs[:self.idx], self.next_obs[:self.idx], self.obs_value[:self.idx], \
               self.action[:self.idx], self.action_log_prob[:self.idx], self.reward[:self.idx], \
               self.done[:self.idx], self.eposide_idx
        return data

    def run(self):
        if self.algo == 'ia2c':
            for i in range(self.num_update):
                self.reset()
                s = self.env.reset()  # 状态初始化
                eposide_reward1, eposide_reward2 = 0, 0
                now_episodes = 0
                while now_episodes < self.num_episodes:
                    # 动作选择
                    a, a_lp, v = self.agent.take_action(s)
                    # 环境更新
                    next_s, r, done, info = self.env.step(a.squeeze(dim=1).numpy())
                    # 构造数据集
                    self.insert(s, next_s, v, a, a_lp, r, done)
                    s = next_s  # 状态更新
                    if all(done):  # 判断当前回合是否都为True，是返回True，不是返回False
                        s = self.env.reset()
                        now_episodes += 1
                        train_info = self.agent.update(self.get_data())
                        self.reset()
                    eposide_reward1 += r[0]
                    eposide_reward2 += r[1]
                    # time.sleep(0.1)
                print('epoch:', i)
                # 回合训练

                for k, v in train_info.items():
                    self.writer.add_scalar('train/' + k, v, i)
                self.writer.add_scalar("train/mean_epo_reward1", eposide_reward1 / self.num_episodes, i)
                self.writer.add_scalar("train/mean_epo_reward2", eposide_reward2 / self.num_episodes, i)
                self.writer.add_scalar("train/mean_epo_reward_sum",
                                       eposide_reward1 / self.num_episodes + eposide_reward2 / self.num_episodes, i)
        else:
            for i in range(self.num_update):
                self.reset()
                self.agent.reset()
                s = self.env.reset()  # 状态初始化
                eposide_reward1, eposide_reward2 = 0, 0
                now_episodes = 0
                while now_episodes < self.num_episodes:
                    # 动作选择
                    a, a_lp, v = self.agent.take_action(s)
                    # 环境更新
                    next_s, r, done, info = self.env.step(a.squeeze(dim=1).numpy())
                    # 构造数据集
                    self.insert(s, next_s, v, a, a_lp, r, done)
                    s = next_s  # 状态更新
                    if all(done):  # 判断当前回合是否都为True，是返回True，不是返回False
                        s = self.env.reset()
                        now_episodes += 1
                        self.agent.reset()
                    eposide_reward1 += r[0]
                    eposide_reward2 += r[1]
                    # time.sleep(0.1)
                print('epoch:', i)
                # 回合训练

                train_info = self.agent.update(self.get_data())
                for k, v in train_info.items():
                    self.writer.add_scalar('train/' + k, v, i)
                self.writer.add_scalar("train/mean_epo_reward1", eposide_reward1 / self.num_episodes, i)
                self.writer.add_scalar("train/mean_epo_reward2", eposide_reward2 / self.num_episodes, i)
                self.writer.add_scalar("train/mean_epo_reward_sum",
                                       eposide_reward1 / self.num_episodes + eposide_reward2 / self.num_episodes, i)
            self.writer.close()

    def eval(self):
        self.agent.reset()
        s = self.env.reset()  # 状态初始化
        terminal = False  # 结束标记
        while not terminal:
            self.env.render()
            time.sleep(0.1)
            a = self.agent.take_action(s)
            next_s, r, done, info = self.env.step(a.squeeze(dim=1).numpy())
            terminal = all(done)

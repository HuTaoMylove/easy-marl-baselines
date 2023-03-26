import os.path

import torch
from algorithm.ippo import PPO
from algorithm.ia2c import A2C
from algorithm.mappo import MAPPO
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from buffer.q_buffer import iql_buffer, vdn_buffer


class q_runner:
    def __init__(self, env, agent, writer, num_update=10000, num_episodes=50, buffer_max_size=1000000, batch_size=256,
                 algo='iql'):
        self.env = env
        self.agent = agent
        self.writer = writer
        self.num_update = num_update
        self.num_episodes = num_episodes
        obs_dim = env.observation_space[0].shape[0]  # 状态数 = env.observation_space[0].shape[0]
        if algo == 'iql':
            self.buffer = iql_buffer(obs_dim=obs_dim, max_size=buffer_max_size, batch_size=batch_size)
        elif algo in ['vdn', 'qmix']:
            self.buffer = vdn_buffer(obs_dim=obs_dim, max_size=buffer_max_size, batch_size=batch_size)

    def run(self):
        for i in range(self.num_update):
            s = self.env.reset()  # 状态初始化
            eposide_reward1, eposide_reward2 = 0, 0
            now_episodes = 0
            while now_episodes < self.num_episodes:
                # 动作选择
                a_1 = self.agent.take_action(s[0])
                a_2 = self.agent.take_action(s[1])
                # 环境更新
                next_s, r, done, info = self.env.step([a_1, a_2])
                # 构造数据集

                data1 = s[0], a_1, next_s[0], done[0], r[0]
                data2 = s[1], a_2, next_s[1], done[1], r[1]

                self.buffer.insert(data1, data2)

                if all(done):  # 判断当前回合是否都为True，是返回True，不是返回False
                    s = self.env.reset()
                    now_episodes += 1
                else:
                    s = next_s  # 状态更新
                if np.random.rand(1) < 0.5:
                    train_info = self.agent.update(self.buffer)

                eposide_reward1 += r[0]
                eposide_reward2 += r[1]

                # time.sleep(0.1)
            print('epoch:', i)
            if train_info is not None:
                for k, v in train_info.items():
                    self.writer.add_scalar('train/' + k, v, i)
            self.writer.add_scalar("train/mean_epo_reward1", eposide_reward1 / self.num_episodes, i)
            self.writer.add_scalar("train/mean_epo_reward2", eposide_reward2 / self.num_episodes, i)
            self.writer.add_scalar("train/mean_epo_reward_sum",
                                   eposide_reward1 / self.num_episodes + eposide_reward2 / self.num_episodes, i)
        self.writer.close()

    def eval(self):
        s = self.env.reset()  # 状态初始化
        terminal = False  # 结束标记
        while not terminal:
            self.env.render()
            time.sleep(0.3)
            a_1 = self.agent.take_action(s[0])
            a_2 = self.agent.take_action(s[1])

            next_s, r, done, info = self.env.step([a_1, a_2])
            terminal = all(done)



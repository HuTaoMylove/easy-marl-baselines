import os.path

import torch
from algorithm.ippo import PPO
from algorithm.ia2c import A2C
from algorithm.mappo import MAPPO
import time
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class ac_runner:
    def __init__(self, env, agent, writer, num_update=10000, num_episodes=50):
        self.env = env
        self.agent = agent
        self.writer = writer
        self.num_update = num_update
        self.num_episodes = num_episodes

    def run(self):
        for i in range(self.num_update):
            transition_dict_1 = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': [],
            }
            transition_dict_2 = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': [],
            }
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

                # TODO:不是合作咋办
                r = r[0] + r[1]

                transition_dict_1['states'].append(s[0])
                transition_dict_1['actions'].append(a_1)
                transition_dict_1['next_states'].append(next_s[0])
                transition_dict_1['dones'].append(done[0])
                transition_dict_1['rewards'].append(r)

                transition_dict_2['states'].append(s[1])
                transition_dict_2['actions'].append(a_2)
                transition_dict_2['next_states'].append(next_s[1])
                transition_dict_2['dones'].append(done[1])
                transition_dict_2['rewards'].append(r)

                s = next_s  # 状态更新
                if all(done):  # 判断当前回合是否都为True，是返回True，不是返回False
                    s = self.env.reset()
                    now_episodes += 1
                eposide_reward1 += r[0]
                eposide_reward2 += r[1]

                # time.sleep(0.1)
            print('epoch:', i)
            # 回合训练
            train_info = self.agent.update(transition_dict_1, transition_dict_2)
            for k, v in train_info.items():
                self.writer.add_scalar('train/' + k, v, i)
            self.writer.add_scalar("train/mean_epo_reward1", eposide_reward1 / self.num_episodes, i)
            self.writer.add_scalar("train/mean_epo_reward2", eposide_reward2 / self.num_episodes, i)
            self.writer.add_scalar("train/mean_epo_reward_sum",eposide_reward1 / self.num_episodes + eposide_reward2 / self.num_episodes, i)
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



class r_ac_runner:
    def __init__(self, env, agent, writer, num_update=10000, num_episodes=50):
        self.env = env
        self.agent = agent
        self.writer = writer
        self.num_update = num_update
        self.num_episodes = num_episodes

    def run(self):
        for i in range(self.num_update):
            transition_dict_1 = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': [],
            }
            transition_dict_2 = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': [],
            }
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

                # TODO:不是合作咋办
                r = r[0] + r[1]

                transition_dict_1['states'].append(s[0])
                transition_dict_1['actions'].append(a_1)
                transition_dict_1['next_states'].append(next_s[0])
                transition_dict_1['dones'].append(done[0])
                transition_dict_1['rewards'].append(r)

                transition_dict_2['states'].append(s[1])
                transition_dict_2['actions'].append(a_2)
                transition_dict_2['next_states'].append(next_s[1])
                transition_dict_2['dones'].append(done[1])
                transition_dict_2['rewards'].append(r)

                s = next_s  # 状态更新
                if all(done):  # 判断当前回合是否都为True，是返回True，不是返回False
                    s = self.env.reset()
                    now_episodes += 1
                eposide_reward1 += r[0]
                eposide_reward2 += r[1]

                # time.sleep(0.1)
            print('epoch:', i)
            # 回合训练
            train_info = self.agent.update(transition_dict_1, transition_dict_2)
            for k, v in train_info.items():
                self.writer.add_scalar('train/' + k, v, i)
            self.writer.add_scalar("train/mean_epo_reward1", eposide_reward1 / self.num_episodes, i)
            self.writer.add_scalar("train/mean_epo_reward2", eposide_reward2 / self.num_episodes, i)
            self.writer.add_scalar("train/mean_epo_reward_sum",eposide_reward1 / self.num_episodes + eposide_reward2 / self.num_episodes, i)
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
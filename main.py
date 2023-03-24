import os.path

import torch
from algorithm.ippo import PPO
from algorithm.ia2c import A2C
import time
from pathlib import Path
from utils import load_yaml, generate_env
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# ----------------------------------------- #
# 参数设置
# ----------------------------------------- #

env_name = "Combat"
num_episodes = 2500  # 回合数
device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')
algo = 'ippo'
use_render = False

log_dir = Path('./log')
log_dir = log_dir / env_name / algo
if not os.path.exists(log_dir):
    os.makedirs(str(log_dir))

writer = SummaryWriter(log_dir=str(log_dir))

# 创建Combat环境，格子世界的大小为15x15，己方智能体和敌方智能体数量都为2
env = generate_env(env_name)
obs_dim = env.observation_space[0].shape[0]  # 状态数
act_dim = env.action_space[0].n  # 动作数

if algo == 'ippo':
    agent = PPO(n_states=obs_dim,
                n_actions=act_dim,
                device=device
                )
elif algo == 'ia2c':
    agent = A2C(n_states=obs_dim,
                n_actions=act_dim,
                device=device
                )

# ----------------------------------------- #
# 模型训练
# ----------------------------------------- #

for i in range(num_episodes):
    # 每回合开始前初始化两支队伍的数据集
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

    s = env.reset()  # 状态初始化

    sample_time = 0
    eposide_reward1, eposide_reward2 = 0, 0
    terminal = False  # 结束标记
    while sample_time < 20:
        if use_render:
            env.render()

        # 动作选择
        a_1 = agent.take_action(s[0])
        a_2 = agent.take_action(s[1])

        # 环境更新
        next_s, r, done, info = env.step([a_1, a_2])

        # 构造数据集
        transition_dict_1['states'].append(s[0])
        transition_dict_1['actions'].append(a_1)
        transition_dict_1['next_states'].append(next_s[0])
        transition_dict_1['dones'].append(done[0])
        transition_dict_1['rewards'].append(r[0])

        transition_dict_2['states'].append(s[1])
        transition_dict_2['actions'].append(a_2)
        transition_dict_2['next_states'].append(next_s[1])
        transition_dict_2['dones'].append(done[1])
        transition_dict_2['rewards'].append(r[1])

        s = next_s  # 状态更新
        if all(done):  # 判断当前回合是否都为True，是返回True，不是返回False
            s = env.reset()
            sample_time += 1
        eposide_reward1 += r[0]
        eposide_reward2 += r[1]

        # time.sleep(0.1)
    print('epoch:', i)
    # 回合训练
    train_info = agent.update(transition_dict_1, transition_dict_2)
    for k, v in train_info.items():
        writer.add_scalar('train/' + k, v, i)
    writer.add_scalar("train/mean_epo_reward1", eposide_reward1 / 10, i)
    writer.add_scalar("train/mean_epo_reward2", eposide_reward2 / 10, i)

# eval
s = env.reset()  # 状态初始化
terminal = False  # 结束标记
while not terminal:
    env.render()
    time.sleep(0.3)
    a_1 = agent.take_action(s[0])
    a_2 = agent.take_action(s[1])

    next_s, r, done, info = env.step([a_1, a_2])
    terminal = all(done)

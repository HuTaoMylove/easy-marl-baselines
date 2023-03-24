import os.path

import torch
from algorithm.ippo import PPO
from algorithm.ia2c import A2C
from algorithm.mappo import MAPPO
import time
from pathlib import Path
from utils import load_yaml, generate_env
from torch.utils.tensorboard import SummaryWriter
from buffer.ac_buffer import AC_buffer
import numpy as np

# ----------------------------------------- #
# 参数设置
# ----------------------------------------- #

env_name = "Combat"
num_update = 10000  # 回合数
num_episodes = 50
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


def load_agent(algo: str):
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
    elif algo == 'mappo':
        agent = MAPPO(n_states=obs_dim,
                      n_actions=act_dim,
                      device=device
                      )


agent = load_agent(algo)
buffer = AC_buffer()
data0 = {}
data1 = {}

for i in range(num_update):
    s = env.reset()  # 状态初始化
    eposide_reward1, eposide_reward2 = 0, 0
    for j in range(num_episodes):
        if use_render:
            env.render()
        # 动作选择
        a_1 = agent.take_action(s[0])
        a_2 = agent.take_action(s[1])
        # 环境更新
        next_s, r, done, info = env.step([a_1, a_2])
        # 构造数据集
        data0['states'] = s[0]
        data0['actions'] = a_1
        data0['next_states'] = next_s[0]
        data0['dones'] = done[0]
        data0['rewards'] = r[0]
        data1['states'] = s[1]
        data1['actions'] = a_2
        data1['next_states'] = next_s[1]
        data1['dones'] = done[1]
        data1['rewards'] = r[1]
        if buffer.need_nextaction():
            data0['next_actions'] = agent.take_action(next_s[0])
            data1['next_actions'] = agent.take_action(next_s[1])

        buffer.insert(data0, data1)

        s = next_s  # 状态更新
        if all(done):  # 判断当前回合是否都为True，是返回True，不是返回False
            s = env.reset()
        eposide_reward1 += r[0]
        eposide_reward2 += r[1]

        # time.sleep(0.1)
    print('epoch:', i)
    # 回合训练
    buffer.compute()
    train_info = agent.update(*buffer.get_train_data())
    for k, v in train_info.items():
        writer.add_scalar('train/' + k, v, i)
    writer.add_scalar("train/mean_epo_reward1", eposide_reward1 / num_episodes, i)
    writer.add_scalar("train/mean_epo_reward2", eposide_reward2 / num_episodes, i)
writer.close()

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

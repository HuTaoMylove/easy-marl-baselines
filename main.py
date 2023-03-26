import os.path
import torch
from algorithm.ippo import PPO
from algorithm.ia2c import A2C
from algorithm.mappo import MAPPO
from algorithm.iql import IQL
from algorithm.qmix import QMIX
from algorithm.vdn import VDN
import time
from pathlib import Path
from utils import load_yaml, generate_env
from torch.utils.tensorboard import SummaryWriter
from runner.q_runner import q_runner
import numpy as np
from runner.ac_runner import ac_runner

# ----------------------------------------- #
# 参数设置
# ----------------------------------------- #

env_name = "Combat"
num_update = 10000  # 回合数
num_episodes = 50
device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')
device = torch.device('cpu')
algo = 'ia2c'
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
    elif algo == 'iql':
        agent = IQL(n_states=obs_dim,
                    n_actions=act_dim,
                    device=device
                    )
    elif algo == 'vdn':
        agent = VDN(n_states=obs_dim,
                    n_actions=act_dim,
                    device=device
                    )
    elif algo == 'qmix':
        agent = QMIX(n_states=obs_dim,
                     n_actions=act_dim,
                     device=device
                     )
    return agent


agent = load_agent(algo)
if algo in ['iql', 'qmix', 'vdn']:
    runner = q_runner(env, agent, writer, num_update, num_episodes, algo=algo)
else:
    runner = ac_runner(env, agent, writer, num_update, num_episodes)
runner.run()

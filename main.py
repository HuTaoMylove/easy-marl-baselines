import os.path
import torch
from algorithm.ippo import PPO, r_PPO
from algorithm.ia2c import A2C
from algorithm.mappo import MAPPO, r_MAPPO
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

env_name = "PongDuel"
num_update = 5000  # 回合数
num_episodes = 3
device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')
use_render = False
algo = 'ia2c'

log_dir = Path('./log')
log_dir = log_dir / env_name / algo
if not os.path.exists(log_dir):
    os.makedirs(str(log_dir))

writer = SummaryWriter(log_dir=str(log_dir))

# 创建Combat环境，格子世界的大小为15x15，己方智能体和敌方智能体数量都为2
env = generate_env(env_name)

env.seed(2022)
np.random.seed(2022)
torch.cuda.manual_seed(2023)
torch.cuda.manual_seed_all(2023)
torch.manual_seed(2023)

obs_dim = env.observation_space[0].shape[0]  # 状态数
act_dim = env.action_space[0].n  # 动作数


def load_agent(algo: str):
    if algo == 'ippo':
        agent = PPO(n_states=obs_dim,
                    n_actions=act_dim,
                    device=device
                    )
    elif algo == 'r_ippo':
        agent = r_PPO(n_states=obs_dim,
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
    elif algo == 'r_mappo':
        agent = r_MAPPO(n_states=obs_dim,
                        n_actions=act_dim,
                        device=device
                        )
    elif algo == 'iql':
        agent = IQL(n_states=obs_dim,
                    n_actions=act_dim,
                    device=device
                    )
    elif algo == 'iql_is':
        agent = IQL(n_states=obs_dim,
                    n_actions=act_dim,
                    use_importance_sampling=True,
                    device=device
                    )
    elif algo == 'vdn':
        agent = VDN(n_states=obs_dim,
                    n_actions=act_dim,
                    device=device
                    )
    elif algo == 'vdn_is':
        agent = VDN(n_states=obs_dim,
                    n_actions=act_dim,
                    use_importance_sampling=True,
                    device=device
                    )
    elif algo == 'qmix':
        agent = QMIX(n_states=obs_dim,
                     n_actions=act_dim,
                     device=device
                     )
    return agent


agent = load_agent(algo)
if algo in ['iql', 'iql_is', 'qmix', 'vdn', 'vdn_is']:
    runner = q_runner(env, agent, writer, num_update, num_episodes, algo=algo)
else:
    runner = ac_runner(env, agent, writer, num_update, num_episodes, algo=algo)
runner.run()

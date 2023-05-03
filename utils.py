import yaml
from pathlib import Path
from magym.ma_gym.envs.combat.combat import Combat
from magym.ma_gym.envs.pong_duel.pong_duel import PongDuel
from magym.ma_gym.envs.switch import Switch
from magym.ma_gym.envs.lumberjacks.lumberjacks import Lumberjacks
from magym.ma_gym.envs.checkers.checkers import Checkers
from magym.ma_gym.envs.predator_prey.predator_prey import PredatorPrey
import torch.nn as nn


def load_yaml(path: Path):
    with open(str(path), "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    # 将dict转化为类
    return type('EnvConfig', (object,), data)


def generate_env(env_name: str):
    env_fig_path = Path('cofigs') / f'{env_name}.yaml'
    env_fig = load_yaml(env_fig_path)

    assert env_name in ['Combat', 'PongDuel', 'Switch', 'Lumberjacks', 'Checkers', 'PredatorPrey']

    if env_name == 'Combat':
        team_size = getattr(env_fig, 'team_size', 2)
        grid_size = getattr(env_fig, 'grid_size', 15)
        grid_size = (grid_size, grid_size)
        full_observable = getattr(env_fig, 'full_observable', False)
        env = Combat(grid_shape=grid_size, n_agents=team_size, n_opponents=team_size, full_observable=full_observable)
    elif env_name == 'PongDuel':
        max_rounds = getattr(env_fig, 'max_rounds', 40)
        env = PongDuel(max_rounds=max_rounds)
    elif env_name == 'Switch':
        team_size = getattr(env_fig, 'team_size', 2)
        full_observable = getattr(env_fig, 'full_observable', False)
        env = Switch(n_agents=team_size, full_observable=full_observable)
    elif env_name == 'Lumberjacks':
        team_size = getattr(env_fig, 'team_size', 2)
        full_observable = getattr(env_fig, 'full_observable', False)
        grid_size = getattr(env_fig, 'grid_size', 15)
        grid_size = (grid_size, grid_size)
        n_tree = getattr(env_fig, 'n_tree', 24)
        env = Lumberjacks(grid_shape=grid_size, n_agents=team_size, n_trees=n_tree, full_observable=full_observable)
    elif env_name == 'Checkers':
        full_observable = getattr(env_fig, 'full_observable', False)
        env = Checkers(full_observable=full_observable)
    elif env_name == 'PredatorPrey':
        team_size = getattr(env_fig, 'team_size', 2)
        grid_size = getattr(env_fig, 'grid_size', 15)
        grid_size = (grid_size, grid_size)
        n_preys = 2 * team_size
        full_observable = getattr(env_fig, 'full_observable', False)
        env = PredatorPrey(grid_shape=grid_size, n_agents=team_size, n_preys=n_preys, full_observable=full_observable)

    return env


def init(module: nn.Module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module




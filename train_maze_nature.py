import gym
from utils.preprocess import greyscale
from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from schedule import LinearExploration, LinearSchedule
from nature import NatureQN
from utils.maze_env import EnvMaze

from configs.train_maze_nature import Config

import sys


if __name__ == '__main__':
    output_path = sys.argv[1]
    config = Config()
    config.set_paths(output_path)

    # make env
    env = EnvMaze(n=8)

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
            config.eps_end, config.eps_nsteps, config.env_name)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps, config.env_name)

    # train model
    model = NatureQN(env, config)
    model.bfs_len = env.get_bfs_length()
    model.run(exp_schedule, lr_schedule)

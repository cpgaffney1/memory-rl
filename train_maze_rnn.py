#from utils.preprocess import greyscale
#from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from schedule import LinearExploration, LinearSchedule
from nature import NatureQN
from utils.maze_env import EnvMaze

import configs
from configs.train_maze_rnn import Config
import sys

def train_maze(output_path):
    config = Config()
    config.set_paths(output_path)

    env = EnvMaze(n=config.maze_size, hard=config.hard)

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin,
                                     config.eps_end, config.eps_nsteps, config.env_name)

    # learning rate schedule
    lr_schedule = LinearSchedule(config.lr_begin, config.lr_end,
                                 config.lr_nsteps, config.env_name)

    # train model
    print(config.output_path)
    model = NatureQN(env, config)
    model.bfs_len = env.get_bfs_length()
    evaluation_result_list, oos_evalution_result_list = model.run(exp_schedule, lr_schedule)
    return evaluation_result_list, oos_evalution_result_list



if __name__ == '__main__':
    # make env
    output_path = sys.argv[1]
    train_maze(output_path)
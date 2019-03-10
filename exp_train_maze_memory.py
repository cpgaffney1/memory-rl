#from utils.preprocess import greyscale
#from utils.wrappers import PreproWrapper, MaxAndSkipEnv

from schedule import LinearExploration, LinearSchedule
from nature import NatureQN
from utils.maze_env import EnvMaze

import configs
from configs.train_maze_memory import Config
import sys
from train_maze_memory import train_maze

import matplotlib.pyplot as plt


# DO NOT USE

if __name__ == '__main__':
    # make env
    output_path = output_path = sys.argv[1]

    evaluation_result_list, oos_evalution_result_list = [], []
    for _ in range(10):
        el, oos_el = train_maze(output_path)
        evaluation_result_list += el
        oos_evalution_result_list += oos_el

    avg_reward, percent_completed, avg_length = zip(*evaluation_result_list)

    print(len(evaluation_result_list))

    plt.hist(avg_reward)
    plt.savefig(output_path + 'reward_hist.png')
    plt.clf()

    plt.hist(percent_completed)
    plt.savefig(output_path + 'completed_hist.png')
    plt.clf()

    plt.hist(avg_length)
    plt.savefig(output_path + 'length_hist.png')
    plt.clf()

    avg_reward, percent_completed, avg_length = zip(*oos_evalution_result_list)

    plt.hist(avg_reward)
    plt.savefig(output_path + 'oos_reward_hist.png')
    plt.clf()

    plt.hist(percent_completed)
    plt.savefig(output_path + 'oos_completed_hist.png')
    plt.clf()

    plt.hist(avg_length)
    plt.savefig(output_path + 'oos_length_hist.png')
    plt.clf()


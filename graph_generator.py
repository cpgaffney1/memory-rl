import numpy as np
import matplotlib.pyplot as plt

def load_time_logs():
    dirs = ['memory_update1k_timing', 'memory_update5k_time', 'memory_update10k_time', 'memory_update25k_time',
            'memory_update50k_timing', 'memory_update100k_timing', 'memory_update250k_time', 'memory_update500k_time']

    all_times = []
    std = []
    for n in dirs:
        with open('C:/Users/cpgaf/PycharmProjects/memory-rl/results/' + n + '/time_log.txt') as f:
            lines = f.readlines()
            times = []
            for l in lines:
                if l == '\n':
                    continue
                times += [float(l)]
            times = np.array(times)
            times = times[times >= np.quantile(times, 0.05)]
            times = times[times <= np.quantile(times, 0.95)]
            print(np.median(times))
            all_times += [np.median(times)]
            std += [np.sqrt(np.var(times) / len(times))]
    return all_times, std


if __name__ == '__main__':
    all_times, std = load_time_logs()


    ### Varying memory update frequency. Numbers taken from logs
    x =         np.array([1,    5,    10,   25,   50,    100,   250,  500])
    rewards =   np.array([0.95, 0.85, 0.86, 0.95, 0.63,  0.52,  0.15, 0.09])
    conf =      np.array([0.01, 0.01, 0.01, 0.01, 0.02,  0.02,  0.06, 0.06])
    completed = np.array([0.97, 0.91, 0.9,  0.97, 0.74,  0.59,  0.24, 0.20])

    _, ax = plt.subplots()

    lns1 = ax.plot(x, rewards, color='#539caf', label='Reward')
    ax.set_xscale('log')
    ax.fill_between(x, rewards - 2*conf, rewards + 2*conf, color='#539caf', alpha=0.4)
    ax.set_xlabel('Memory update frequency (thousands)')
    ax.set_ylabel('Reward')

    ax2 = ax.twinx()
    lns2 = ax2.plot(x, all_times, color='#7663b0', label='Time')
    ax2.set_ylabel('Median time per 1000 iterations')
    # Show right frame line
    ax2.spines['right'].set_visible(True)

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    plt.show()
    plt.clf()

    _, ax = plt.subplots()

    lns1 = ax.plot(x, completed, color='#539caf', label='Completion Rate')
    ax.set_xscale('log')
    ax.fill_between(x, completed - 2 * conf, completed + 2 * conf, color='#539caf', alpha=0.4)
    plt.xlabel('Memory update frequency (thousands)')
    plt.ylabel('Completion rate')

    ax2 = ax.twinx()
    lns2 = ax2.plot(x, all_times, color='#7663b0', label='Time')
    ax2.set_ylabel('Median time per 1000 iterations')
    # Show right frame line
    ax2.spines['right'].set_visible(True)

    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    plt.show()
    plt.clf()



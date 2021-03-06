import os
#import gym
import numpy as np
import logging
import time
import sys
#from gym import wrappers
from collections import deque

from utils.general import get_logger, Progbar, export_plot
from utils.replay_buffer import ReplayBuffer
from utils.preprocess import greyscale
from utils.maze_env import EnvMaze
#from utils.wrappers import PreproWrapper, MaxAndSkipEnv


class QN(object):
    """
    Abstract Class for implementing a Q Network
    """
    def __init__(self, env, config, logger=None):
        """
        Initialize Q Network and env

        Args:
            config: class with hyperparameters
            logger: logger instance from logging module
        """
        # directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)
            
        # store hyper params
        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        self.env = env

        # build model
        if self.config.use_memory:
            self.build_with_memory()
        else:
            self.build()


    def build(self):
        """
        Build model
        """
        pass

    def build_with_memory(self):
        """
        Build model
        """
        pass

    @property
    def policy(self):
        """
        model.policy(state) = action
        """
        return lambda state: self.get_action(state)


    def save(self):
        """
        Save model parameters

        Args:
            model_path: (string) directory
        """
        pass


    def initialize(self):
        """
        Initialize variables if necessary
        """
        pass


    def get_best_action(self, state):
        """
        Returns best action according to the network
    
        Args:
            state: observation from gym
        Returns:
            tuple: action, q values
        """
        raise NotImplementedError

    def get_best_action_with_memory(self, state, memory):
        raise NotImplementedError

    def get_action(self, state):
        """
        Returns action with some epsilon strategy

        Args:
            state: observation from gym
        """
        if np.random.random() < self.config.soft_epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_best_action(state)[0]

    def get_action_with_memory(self, state, memory):
        best_action, bottom_q, top_q, memory = self.get_best_action_with_memory(state, memory)
        if np.random.random() < self.config.soft_epsilon:
            return self.env.action_space.sample(), bottom_q, top_q, memory
        else:
            return best_action, bottom_q, top_q, memory


    def update_target_params(self):
        """
        Update params of Q' with params of Q
        """
        pass


    def init_averages(self):
        """
        Defines extra attributes for tensorboard
        """
        self.avg_reward = -21.
        self.max_reward = -21.
        self.std_reward = 0

        self.avg_q = 0
        self.max_q = 0
        self.std_q = 0
        
        self.eval_reward = -21.


    def update_averages(self, rewards, max_q_values, q_values, scores_eval):
        """
        Update the averages

        Args:
            rewards: deque
            max_q_values: deque
            q_values: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        self.max_q      = np.mean(max_q_values)
        self.avg_q      = np.mean(q_values)
        self.std_q      = np.sqrt(np.var(q_values) / len(q_values))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]


    def train(self, exp_schedule, lr_schedule):
        """
        Performs training of Q

        Args:
            exp_schedule: Exploration instance s.t.
                exp_schedule.get_action(best_action) returns an action
            lr_schedule: Schedule for learning rate
        """

        # initialize replay buffer and variables

        if self.config.use_memory:
            replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history, memory_size=self.config.memory_unit_size)
        else:
            replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        rewards = deque(maxlen=self.config.num_episodes_test)
        max_q_values = deque(maxlen=1000)
        q_values = deque(maxlen=1000)
        self.init_averages()

        t = last_eval = last_record = 0 # time control of nb of steps
        scores_eval = [] # list of scores computed at iteration time
        scores_eval += [self.evaluate()[0]]
        
        prog = Progbar(target=self.config.nsteps_train)

        evaluation_result_list = []
        oos_evalution_result_list = []

        # interact with environment
        prev_time = time.time()
        while t < self.config.nsteps_train:
            total_reward = 0
            state = self.env.reset()
            while True:
                t += 1
                last_eval += 1
                last_record += 1
                if self.config.render_train: self.env.render()
                # replay memory stuff
                idx = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                if self.config.use_memory:
                    prev_memory = replay_buffer.encode_recent_memory()
                    best_action, q_values, _, next_memory = self.get_best_action_with_memory(q_input, prev_memory)
                    next_memory = np.squeeze(next_memory)
                else:
                    best_action, q_values = self.get_best_action(q_input)
                # chose action according to current Q and exploration
                action = exp_schedule.get_action(best_action)

                # store q values
                max_q_values.append(max(q_values))
                q_values += list(q_values)

                # perform action in env
                new_state, reward, done, info = self.env.step(action)

                # store the transition
                replay_buffer.store_effect(idx, action, reward, done)
                if self.config.use_memory:
                    replay_buffer.store_memory(idx, next_memory)
                state = new_state

                # perform a training step
                loss_eval, grad_eval = self.train_step(t, replay_buffer, lr_schedule.epsilon)

                # logging stuff
                time_log_freq = 1000
                if t % time_log_freq == 0:
                    with open(self.config.output_path + 'time_log.txt', 'a') as of:
                        of.write('{}\n'.format(time.time() - prev_time))
                        of.write('\n')
                    prev_time = time.time()

                if ((t > self.config.learning_start) and (t % self.config.log_freq == 0) and
                   (t % self.config.learning_freq == 0)):
                    self.update_averages(rewards, max_q_values, q_values, scores_eval)
                    exp_schedule.update(t)
                    lr_schedule.update(t)
                    if len(rewards) > 0:
                        prog.update(t + 1, exact=[("Loss", loss_eval), ("Avg_R", self.avg_reward), 
                                        ("Max_R", np.max(rewards)), ("eps", exp_schedule.epsilon), 
                                        ("Grads", grad_eval), ("Max_Q", self.max_q), 
                                        ("lr", lr_schedule.epsilon)])

                elif (t < self.config.learning_start) and (t % self.config.log_freq == 0):
                    sys.stdout.write("\rPopulating the memory {}/{}...".format(t, 
                                                        self.config.learning_start))
                    sys.stdout.flush()

                # count reward
                total_reward += reward
                if done or t >= self.config.nsteps_train:
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)          

            if (t > self.config.learning_start) and (last_eval > self.config.eval_freq):
                # evaluate our policy
                last_eval = 0
                print("")
                score, complete, length = self.evaluate()
                if complete > 0:
                    evaluation_result_list += [(score, complete, length)]
                if score > self.config.extended_eval_threshold:
                    self.logger.info('Extended in-sample evaluation...')
                    self.evaluate(num_episodes=1000)
                    for _ in range(10):
                        self.logger.info('Extended out-of-sample evaluation...')
                        oos_result = self.evaluate(EnvMaze(n=self.config.maze_size), num_episodes=100)
                        oos_evalution_result_list += [oos_result]
                scores_eval += [score]

            if (t > self.config.learning_start) and self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record = 0
                self.record()

        # last words
        self.logger.info("- Training done.")
        self.save()
        scores_eval += [self.evaluate()[0]]
        export_plot(scores_eval, "Scores", self.config.plot_output)

        return evaluation_result_list, oos_evalution_result_list


    def train_step(self, t, replay_buffer, lr):
        """
        Perform training step

        Args:
            t: (int) nths step
            replay_buffer: buffer for sampling
            lr: (float) learning rate
        """
        loss_eval, grad_eval = 0, 0

        # perform training step
        if (t > self.config.learning_start and t % self.config.learning_freq == 0):
            loss_eval, grad_eval = self.update_step(t, replay_buffer, lr)

        # occasionaly update target network with q network
        if t % self.config.target_update_freq == 0:
            self.update_target_params()
            
        # occasionaly save the weights
        if (t % self.config.saving_freq == 0):
            self.save()

        return loss_eval, grad_eval


    def evaluate(self, env=None, num_episodes=None):
        """
        Evaluation with same procedure as the training
        """
        # log our activity only if default call
        save_paths = False
        if num_episodes is None:
            self.logger.info("Evaluating...")
        else:
            save_paths = True

        # arguments defaults
        if num_episodes is None:
            num_episodes = self.config.num_episodes_test

        if env is None:
            env = self.env
            bfs_len = self.bfs_len
        else:
            bfs_len = env.get_bfs_length()

        # replay memory to play
        if self.config.use_memory:
            replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history, memory_size=self.config.memory_unit_size)
        else:
            replay_buffer = ReplayBuffer(self.config.buffer_size, self.config.state_history)
        rewards = []
        steps = []

        for i in range(num_episodes):
            total_reward = 0
            state = env.reset()
            count = 0
            while True:
                if self.config.render_test: env.render()

                # store last state in buffer
                idx = replay_buffer.store_frame(state)
                q_input = replay_buffer.encode_recent_observation()

                if self.config.use_memory:
                    prev_memory = replay_buffer.encode_recent_memory()
                    action, bottom_q, top_q, next_memory = self.get_action_with_memory(q_input, prev_memory)
                    next_memory = np.squeeze(next_memory)
                else:
                    action = self.get_action(q_input)


                if i == 0 and self.config.use_memory:
                    with open(self.config.output_path + 'eval_example_log.txt', 'a') as of:
                        of.write('State = {}\n'.format(env.cur_state))
                        of.write('Taking action = {}\n'.format(action))
                        of.write('prev_memory = {}\n'.format(prev_memory[0, :6]))
                        of.write('next_memory = {}\n'.format(next_memory[:6]))
                        of.write('bottom_q_values = {}\n'.format(bottom_q))
                        of.write('top_q_values = {}\n'.format(top_q))
                        of.write('\n')

                if save_paths:
                    with open(self.config.output_path + 'path_log.txt', 'a') as of:
                        of.write("(s, a) = ({}, {})\n".format(env.cur_state, action))
                        of.write('\n')


                # perform action in env
                new_state, reward, done, info = env.step(action)

                # store in replay memory
                replay_buffer.store_effect(idx, action, reward, done)

                if self.config.use_memory:
                    replay_buffer.store_memory(idx, next_memory)

                state = new_state

                count += 1

                # count reward
                total_reward += reward
                if done:
                    if save_paths:
                        with open(self.config.output_path + 'path_log.txt', 'a') as of:
                            of.write('\n')
                    break

            # updates to perform at the end of an episode
            rewards.append(total_reward)
            if total_reward <= 0:
                steps.append(np.nan)
            else:
                steps.append(count)

        steps = np.array(steps) - bfs_len # adjust for shortest possible path
        avg_reward = np.mean(rewards)

        avg_length = np.nanmean(steps)
        sigma_length = np.sqrt(np.nanvar(steps) / len(steps))
        percent_completed = np.count_nonzero(~np.isnan(steps)) / float(len(steps))
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))


        if num_episodes > 1:
            msg = "Average reward: {:04.2f} +/- {:04.2f}, Percent completed: {:04.2f}, Average length: {:04.2f} +/- {:04.2f}, n = {}".format(
                avg_reward, sigma_reward, percent_completed, avg_length, sigma_length, len(rewards))
            self.logger.info(msg)

        return avg_reward, percent_completed, avg_length


    def record(self):
        """
        Re create an env and record a video for one episode
        """
        env = gym.make(self.config.env_name)
        env = gym.wrappers.Monitor(env, self.config.record_path, video_callable=lambda x: True, resume=True)
        env = MaxAndSkipEnv(env, skip=self.config.skip_frame)
        env = PreproWrapper(env, prepro=greyscale, shape=(80, 80, 1), 
                        overwrite_render=self.config.overwrite_render)
        self.evaluate(env, 1)


    def run(self, exp_schedule, lr_schedule):
        """
        Apply procedures of training for a QN

        Args:
            exp_schedule: exploration strategy for epsilon
            lr_schedule: schedule for learning rate
        """
        # initialize
        self.initialize()

        # record one game at the beginning
        if self.config.record:
            self.record()

        # model
        evaluation_result_list, oos_evalution_result_list = self.train(exp_schedule, lr_schedule)

        # record one game at the end
        if self.config.record:
            self.record()

        return evaluation_result_list, oos_evalution_result_list

    def resume_and_eval(self, path, n_evals=1):
        self.initialize()

        self.saver.restore(self.sess, path)

        self.evaluate()

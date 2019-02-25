import os
import numpy as np
import tensorflow as tf
import time

from core.q_learning import QN



class DQN(QN):
    """
    Abstract class for Deep Q Learning
    """
    def add_placeholders_op(self):
        raise NotImplementedError


    def get_q_values_op(self, state, scope, reuse=False):
        """
        set Q values, of shape = (batch_size, num_actions)
        """
        raise NotImplementedError

    def get_q_values_and_memory_op(self, state1, state2, mem1, scope, reuse=False):
        raise NotImplementedError


    def add_update_target_op(self, q_scope, target_q_scope):
        """
        Update_target_op will be called periodically 
        to copy Q network to target Q network
    
        Args:
            q_scope: name of the scope of variables for q
            target_q_scope: name of the scope of variables for the target
                network
        """
        raise NotImplementedError


    def add_loss_op(self, q, target_q):
        """
        Set (Q_target - Q)^2
        """
        raise NotImplementedError

    def add_memory_loss_op(self, q_bottom, q_top, target_q_bottom, target_q_top):
        """
        Set (Q_target - Q)^2
        """
        raise NotImplementedError


    def add_optimizer_op(self, scope):
        """
        Set training op wrt to loss for variable in scope
        """
        raise NotImplementedError


    def process_state(self, state):
        """
        Processing of state

        State placeholders are tf.uint8 for fast transfer to GPU
        Need to cast it to float32 for the rest of the tf graph.

        Args:
            state: node of tf graph of shape = (batch_size, height, width, nchannels)
                    of type tf.uint8.
                    if , values are between 0 and 255 -> 0 and 1
        """
        state = tf.cast(state, tf.float32)
        state /= self.config.high

        return state


    def build(self):
        """
        Build model by adding all necessary variables
        """
        # add placeholders
        self.add_placeholders_op()

        # compute Q values of state
        s = self.process_state(self.s)
        self.q = self.get_q_values_op(s, scope="q", reuse=False)

        # compute Q values of next state
        sp = self.process_state(self.sp)
        self.target_q = self.get_q_values_op(sp, scope="target_q", reuse=False)

        # add update operator for target network
        self.add_update_target_op("q", "target_q")

        # add square loss
        self.add_loss_op(self.q, self.target_q)

        # add optmizer for the main networks
        self.add_optimizer_op("q")

    def build_with_memory(self):

        """
        Build model by adding all necessary variables
        """
        # add placeholders
        self.add_placeholders_op()

        # compute Q values of state
        s1 = self.process_state(self.s1)
        s2 = self.process_state(self.s2)
        self.q_bottom, self.q_top, self.next_memory = self.get_q_values_and_memory_op(s1, s2, self.memory1, scope="q", reuse=False)

        # compute Q values of next state
        sp1 = self.process_state(self.sp1)
        sp2 = self.process_state(self.sp2)
        self.target_q_bottom, self.target_q_top, _ = self.get_q_values_and_memory_op(sp1, sp2, self.target_memory, scope="target_q", reuse=False)

        # add update operator for target network
        self.add_update_target_op("q", "target_q")

        # add square loss
        self.add_memory_loss_op(self.q_bottom, self.q_top, self.target_q_bottom, self.target_q_top)

        # add optmizer for the main networks
        self.add_optimizer_op("q")


    def initialize(self):
        """
        Assumes the graph has been constructed
        Creates a tf Session and run initializer of variables
        """
        # create tf session
        self.sess = tf.Session()

        # tensorboard stuff
        self.add_summary()

        # initiliaze all variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # synchronise q and target_q networks
        self.sess.run(self.update_target_op)

        # for saving networks weights
        self.saver = tf.train.Saver()

       
    def add_summary(self):
        """
        Tensorboard stuff
        """
        # extra placeholders to log stuff from python
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="avg_reward")
        self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="max_reward")
        self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="std_reward")

        self.avg_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="avg_q")
        self.max_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="max_q")
        self.std_q_placeholder  = tf.placeholder(tf.float32, shape=(), name="std_q")

        self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(), name="eval_reward")

        # add placeholders from the graph
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("grads_norm", self.grad_norm)

        # extra summaries from python -> placeholders
        tf.summary.scalar("Avg_Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max_Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std_Reward", self.std_reward_placeholder)

        tf.summary.scalar("Avg_Q", self.avg_q_placeholder)
        tf.summary.scalar("Max_Q", self.max_q_placeholder)
        tf.summary.scalar("Std_Q", self.std_q_placeholder)

        tf.summary.scalar("Eval_Reward", self.eval_reward_placeholder)
            
        # logging
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path, 
                                                self.sess.graph)



    def save(self):
        """
        Saves session
        """
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)

        self.saver.save(self.sess, self.config.model_output)


    def get_best_action(self, state):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        action_values = self.sess.run(self.q, feed_dict={self.s: [state]})[0]
        return np.argmax(action_values), action_values

    def get_best_action_with_memory(self, state, prev_memory):
        """
        Return best action

        Args:
            state: 4 consecutive observations from gym
            prev_memory: memory unit from previous iteration
        Returns:
            action: (int)
            action_values: (np array) q values for all actions
        """
        action_values, next_memory = self.sess.run([self.q_bottom, self.next_memory], feed_dict={
            self.s1: [state], self.s2: [np.zeros_like(state)], self.memory1: prev_memory
        })

        return np.argmax(action_values), action_values, next_memory

    def update_memory(self, state, prev_memory):
        next_memory = self.sess.run(self.next_memory, feed_dict={
            self.s1: state, self.s2: np.zeros_like(state), self.memory1: prev_memory
        })[0]
        return next_memory

    def update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """
        if self.config.use_memory:
            return self.memory_update_step(t,replay_buffer,lr)
        
        s_batch, a_batch, r_batch, sp_batch, done_mask_batch = replay_buffer.sample(
            self.config.batch_size, use_memory=False)


        fd = {
            # inputs
            self.s: s_batch,
            self.a: a_batch,
            self.r: r_batch,
            self.sp: sp_batch, 
            self.done_mask: done_mask_batch,
            self.lr: lr, 
            # extra info
            self.avg_reward_placeholder: self.avg_reward, 
            self.max_reward_placeholder: self.max_reward, 
            self.std_reward_placeholder: self.std_reward, 
            self.avg_q_placeholder: self.avg_q, 
            self.max_q_placeholder: self.max_q, 
            self.std_q_placeholder: self.std_q, 
            self.eval_reward_placeholder: self.eval_reward, 
        }

        loss_eval, grad_norm_eval, summary, _ = self.sess.run([self.loss, self.grad_norm, 
                                                 self.merged, self.train_op], feed_dict=fd)


        # tensorboard stuff
        self.file_writer.add_summary(summary, t)
        
        return loss_eval, grad_norm_eval


    def memory_update_step(self, t, replay_buffer, lr):
        """
        Performs an update of parameters by sampling from replay_buffer

        Args:
            t: number of iteration (episode and move)
            replay_buffer: ReplayBuffer instance .sample() gives batches
            lr: (float) learning rate
        Returns:
            loss: (Q - Q_target)^2
        """
        if t % self.config.memory_update_freq == 0:
            replay_buffer.reset_recently_updated_episodes()

        s_batch1, a_batch1, r_batch1, sp_batch1, done_mask_batch1, memory_batch1, \
        s_batch2, a_batch2, r_batch2, sp_batch2, done_mask_batch2, target_memory_batch = replay_buffer.sample(
            self.config.batch_size, use_memory=True, update_memory_func=self.update_memory)

        fd = {
            # inputs
            self.s1: s_batch1,
            self.a1: a_batch1,
            self.r1: r_batch1,
            self.sp1: sp_batch1,
            self.done_mask1: done_mask_batch1,
            self.memory1: memory_batch1,
            self.s2: s_batch2,
            self.a2: a_batch2,
            self.r2: r_batch2,
            self.sp2: sp_batch2,
            self.done_mask2: done_mask_batch2,
            self.target_memory: target_memory_batch,
            self.lr: lr,
            # extra info

            self.avg_reward_placeholder: self.avg_reward,
            self.max_reward_placeholder: self.max_reward,
            self.std_reward_placeholder: self.std_reward,
            self.avg_q_placeholder: self.avg_q,
            self.max_q_placeholder: self.max_q,
            self.std_q_placeholder: self.std_q,
            self.eval_reward_placeholder: self.eval_reward,
        }

        loss_eval, grad_norm_eval, summary, _, next_memory = self.sess.run([self.loss, self.grad_norm,
                                                               self.merged,
                                                               self.train_op,
                                                               self.next_memory,
                                                               ], feed_dict=fd)

        # tensorboard stuff
        self.file_writer.add_summary(summary, t)

        return loss_eval, grad_norm_eval


    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """
        self.sess.run(self.update_target_op)


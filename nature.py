import tensorflow as tf
import tensorflow.contrib.layers as layers

from utils.general import get_logger
from utils.test_env import EnvTest
from schedule import LinearExploration, LinearSchedule
from linear import Linear


from configs.nature import config


class NatureQN(Linear):
    """
    Implementing DeepMind's Nature paper. Here are the relevant urls.
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
    """
    def get_q_values_op(self, state, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n

        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
                https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

              you may find the section "model architecture" of the appendix of the 
              nature paper particulary useful.

              store your result in out of shape = (batch_size, num_actions)

        HINT: 
            - You may find the following functions useful:
                - tf.layers.conv2d
                - tf.layers.flatten
                - tf.layers.dense

            - Make sure to also specify the scope and reuse

        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################ 
        print('Loading Q network')
        with tf.variable_scope(scope, reuse=reuse):
            cnn_output = state
            for i in range(len(self.config.cnn_filters)):
                cnn_output = tf.layers.conv2d(cnn_output, self.config.cnn_filters[i], self.config.cnn_kernel[i],
                                              padding='valid',
                                              activation=tf.nn.relu)
            cnn_output = tf.layers.flatten(cnn_output)

            h = tf.layers.dense(cnn_output, self.config.hidden_size, activation=tf.nn.relu)

            out = tf.layers.dense(h, num_actions)

        ##############################################################
        ######################## END YOUR CODE #######################
        return out


    def get_q_values_and_memory_op(self, state1, state2, mem1, scope, reuse=False):
        """
        Returns Q values for all actions

        Args:
            state: (tf tensor) 
                shape = (batch_size, img height, img width, nchannels)
            scope: (string) scope name, that specifies if target network or not
            reuse: (bool) reuse of variables in the scope
            prev_memory: (tf tensor)
            	shape = (batch_size, memory_unit_size)

        Returns:
            out: (tf tensor) of shape = (batch_size, num_actions)
        """
        # this information might be useful
        num_actions = self.env.action_space.n
        print('Loading memory Q network')
        ##############################################################
        """
        TODO: implement the computation of Q values like in the paper
                https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
                https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

              you may find the section "model architecture" of the appendix of the 
              nature paper particulary useful.

              store your result in out of shape = (batch_size, num_actions)

        HINT: 
            - You may find the following functions useful:
                - tf.layers.conv2d
                - tf.layers.flatten
                - tf.layers.dense

            - Make sure to also specify the scope and reuse

        """
        ##############################################################
        ################ YOUR CODE HERE - 10-15 lines ################ 

        with tf.variable_scope(scope, reuse=reuse):
            #### Bottom network
            cnn_output = state1

            if self.config.atari:
                for i in range(len(self.config.cnn_filters)):
                    cnn_output = tf.layers.conv2d(cnn_output, self.config.cnn_filters[i], self.config.cnn_kernel[i],
                    	                          stride=self.config.cnn_strides[i]
                                                  padding='same',
                                                  activation=tf.nn.relu)
            else:
                for i in range(len(self.config.cnn_filters)):
                    cnn_output = tf.layers.conv2d(cnn_output, self.config.cnn_filters[i], self.config.cnn_kernel[i],
                                                  padding='valid',
                                                  activation=tf.nn.relu)
            # for i in range(len(self.config.cnn_filters)):
                # cnn_output = tf.layers.conv2d(cnn_output, self.config.cnn_filters[i], self.config.cnn_kernel[i], padding='valid',
                                              # activation=tf.nn.relu)
            cnn_output = tf.layers.flatten(cnn_output)

            concat_memory = tf.concat([cnn_output, mem1], 1)
            h = tf.layers.dense(concat_memory, self.config.hidden_size, activation=tf.nn.relu)

            q_vals_bottom = tf.layers.dense(h, num_actions)
            next_memory = tf.layers.dense(h, self.config.memory_unit_size)

            ###### EXPERIMENT different path to generate memory
            # cnn_output = state1
            # for i in range(len(self.config.cnn_filters)):
            #     cnn_output = tf.layers.conv2d(cnn_output, self.config.cnn_filters[i], self.config.cnn_kernel[i],
            #                                   padding='valid',
            #                                   activation=tf.nn.relu)
            # cnn_output = tf.layers.flatten(cnn_output)
            # concat_memory = tf.concat([cnn_output, mem1], 1)

            # h_mem = tf.layers.dense(concat_memory, self.config.hidden_size, activation=tf.nn.relu)
            # next_memory = tf.layers.dense(h_mem, self.config.memory_unit_size)

            #### Top Network
            cnn_output = state2
            if self.config.atari:
                for i in range(len(self.config.cnn_filters)):
                    cnn_output = tf.layers.conv2d(cnn_output, self.config.cnn_filters[i], self.config.cnn_kernel[i],
                    	                          stride=self.config.cnn_strides[i]
                                                  padding='same',
                                                  activation=tf.nn.relu)
            else:
                for i in range(len(self.config.cnn_filters)):
                    cnn_output = tf.layers.conv2d(cnn_output, self.config.cnn_filters[i], self.config.cnn_kernel[i],
                                                  padding='valid',
                                                  activation=tf.nn.relu)
            # for i in range(len(self.config.cnn_filters)):
                # cnn_output = tf.layers.conv2d(cnn_output, self.config.cnn_filters[i], self.config.cnn_kernel[i], padding='valid',
                                              # activation=tf.nn.relu)
            cnn_output = tf.layers.flatten(cnn_output)

            concat_memory = tf.concat([cnn_output, next_memory], 1)

            h = tf.layers.dense(concat_memory, self.config.hidden_size, activation=tf.nn.relu)

            q_vals_top = tf.layers.dense(h, num_actions)

        ##############################################################
        ######################## END YOUR CODE #######################
        return q_vals_bottom, q_vals_top, next_memory


"""
Use deep Q network for test environment.
"""
if __name__ == '__main__':
    env = EnvTest((80, 80, 1))

    # exploration strategy
    exp_schedule = LinearExploration(env, config.eps_begin, 
            config.eps_end, config.eps_nsteps)

    # learning rate schedule
    lr_schedule  = LinearSchedule(config.lr_begin, config.lr_end,
            config.lr_nsteps)

    # train model
    model = NatureQN(env, config)
    model.run(exp_schedule, lr_schedule)

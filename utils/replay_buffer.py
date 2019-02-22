import numpy as np
import random

class ReplayBuffer(object):
    """
    Taken from Berkeley's Assignment
    """
    def __init__(self, size, frame_history_len):
        """This is a memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        """
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx      = 0
        self.num_in_buffer = 0
        self.next_episode_idx = 0
        # marks index where new buffer begins to overwrite old buffer. So if old buffer starts at i, then
        # buffer_divide_idx = i - 1, which is the most recent element of the new buffer
        self.buffer_divide_idx = 0

        # all are lists of lists. Outer list stores list of obs/action/reward lists that occur in each episode
        self.obs      = None
        self.mem      = None
        self.action   = None
        self.reward   = None
        self.done     = None

    def can_sample(self, batch_size):
        # TODO check
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes, mem_idxes):
        obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)
        mem_batch      = np.array([self._encode_memory(idx) for idx in mem_idxes])

        return [obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, mem_batch]


    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = self.sample_n_unique(batch_size)
        # idxes should be batch_size x 2 array of indices
        sample1 = self._encode_sample(idxes[:,0], idxes[:,0] - 1)
        sample2 = self._encode_sample(idxes[:,1], idxes[:,0])
        return sample1 + sample2

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        # if len(self.obs.shape) <= 2:
        #     return self.obs[end_idx-1]
        # if there weren't enough frames ever in the buffer for context
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)
        # if zero padding is needed for missing context
        # or we are on the boundry of the buffer
        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 2)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            return self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)

    def _encode_memory(self, idx):
        '''

        :param idx: must be the index of the memory you want. Either candidate1 - 1 or candidate1 (in the case of
        candidate2
        :return:
        '''
        if idx < 0 or idx == self.buffer_divide_idx:
            memory = np.zeros_like(self.mem[0])
        else:
            memory = self.mem[idx]
        return memory


    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        if self.obs is None:
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action   = np.empty([self.size],                     dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        if self.next_idx + 1 == self.size:
            self.buffer_divide_idx = (self.next_idx + 1) % self.size
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.

        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done

    def store_memory(self, idx, memory):
        if self.mem is None:
            self.mem = np.empty([self.size] + list(memory.shape), dtype=np.float32)
        self.mem[idx] = memory

    def sample_n_unique(self, n):
        res = []
        while len(res) < n:
            candidate1 = random.randint(0, self.num_in_buffer - 2)
            episode_start = candidate1
            episode_end = candidate1

            if self.done[episode_start]:
                continue

            while episode_start >= 0 and not self.done[episode_start]:
                episode_start -= 1
            assert (episode_start == -1 or self.done[episode_start])
            episode_start += 1

            while episode_end <= self.num_in_buffer - 2 and not self.done[episode_end]:
                episode_end += 1
            assert (episode_end == self.num_in_buffer - 2 or self.done[episode_end])

            candidate2 = random.randint(episode_start, episode_end)
            while candidate2 == candidate1:
                candidate2 = random.randint(episode_start, episode_end)
            assert(candidate1 != candidate2)

            if candidate2 < candidate1:
                temp = candidate2
                candidate2 = candidate1
                candidate1 = temp

            if (candidate1, candidate2) in res:
                continue

            res.append((candidate1, candidate2))

        return np.array(res)

def test1():
    buffer = ReplayBuffer(10, 2)
    assert(not buffer.can_sample(3))

    buffer.store_frame

if __name__ == '__main__':
    test1()
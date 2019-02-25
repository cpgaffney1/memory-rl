import numpy as np
import random

class ReplayBuffer(object):
    """
    Taken from Berkeley's Assignment
    """
    def __init__(self, size, frame_history_len, memory_size=None):
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
        self.mem_size = memory_size
        self.recently_updated_episodes = []
        self.sample_consecutive = True

        self.next_idx      = 0
        self.num_in_buffer = 0
        self.next_episode_idx = 0
        #
        self.buffer_boundary_idx = 0

        # all are lists of lists. Outer list stores list of obs/action/reward lists that occur in each episode
        self.obs      = None
        self.mem      = None
        self.action   = None
        self.reward   = None
        self.done     = None

    def reset_recently_updated_episodes(self):
        self.recently_updated_episodes = []

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes, mem_idxes):
        obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)
        if mem_idxes is None:
            return [obs_batch, act_batch, rew_batch, next_obs_batch, done_mask]
        else:
            mem_batch = np.array([self._encode_memory(idx) for idx in mem_idxes])
            mem_batch = np.squeeze(mem_batch)
            return [obs_batch, act_batch, rew_batch, next_obs_batch, done_mask, mem_batch]


    def sample(self, batch_size, use_memory=True, update_memory_func=None):
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
        '''
        while True:
            try:
                assert self.can_sample(batch_size)
                idxes = self.sample_n_unique(batch_size, update_memory_func=update_memory_func)
                if use_memory:
                    assert (update_memory_func is not None)
                    # idxes should be batch_size x 2 array of indices
                    sample1 = self._encode_sample(idxes[:,0], (idxes[:,0] - 1) % self.size)
                    sample2 = self._encode_sample(idxes[:,1], idxes[:,0])
                    sample = sample1 + sample2
                else:
                    idxes = idxes[:,0]
                    sample = self._encode_sample(idxes, None)
                break
            except Exception as inst:
                print(type(inst))
                print(inst.args)
                print(inst)
        '''

        assert self.can_sample(batch_size)
        idxes = self.sample_n_unique(batch_size, update_memory_func=update_memory_func)
        if use_memory:
            # idxes should be batch_size x 2 array of indices
            sample1 = self._encode_sample(idxes[:, 0], (idxes[:, 0] - 1) % self.size)
            sample2 = self._encode_sample(idxes[:, 1], idxes[:, 0])
            sample = sample1 + sample2
        else:
            idxes = idxes[:, 0]
            sample = self._encode_sample(idxes, None)

        return sample

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

    def encode_recent_memory(self):
        assert (self.num_in_buffer > 0)
        # next_idx - 2 to get prev memory
        prev_memory = np.expand_dims(self._encode_memory((self.next_idx - 2) % self.size), axis=0)
        return prev_memory

    def _encode_observation(self, idx):
        original_idx = idx
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

        if self.next_idx == original_idx:
            start_idx = original_idx
            end_idx = original_idx + 1
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
        if idx < 0 or self.done[idx] or (idx + 1) % self.size == self.next_idx or self.mem is None:
            if self.mem is None:
                assert(self.num_in_buffer == 1)
            memory = np.zeros(self.mem_size)
        else:
            memory = self.mem[idx]
        assert(len(memory) == self.mem_size)
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

    def _lazy_update_memory(self, episode_start, until_idx, update_memory_func):
        if (episode_start, until_idx) in self.recently_updated_episodes:
            return
        for i in range(episode_start, until_idx + 1):
            prev_memory = np.expand_dims(self._encode_memory(i-1), axis=0)
            obs_input = self._encode_observation(i)
            if i == episode_start:
                assert(np.array_equal(prev_memory, np.zeros((1, self.mem_size))))
            _, _, next_memory = update_memory_func(obs_input, prev_memory)
            self.mem[i] = np.squeeze(next_memory)
        self.recently_updated_episodes.append((episode_start, until_idx))

    def _batch_lazy_update_memory(self, episode_idxes_to_update, update_memory_func):
        start_idxes, end_idxes = zip(*episode_idxes_to_update)
        episode_lens = [tup[1] - tup[0] for tup in episode_idxes_to_update]
        for i in range(max(episode_lens) + 1):
            idxes = np.array([start + i for start in start_idxes])
            prev_mem_batch = np.concatenate([np.expand_dims(self._encode_memory(idx-1), axis=0) for idx in idxes], 0)
            obs_batch = np.squeeze(np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0))
            _, _, next_memory = update_memory_func(obs_batch, prev_mem_batch)
            self.mem[idxes] = np.squeeze(next_memory)

    def update_memory(self, update_memory_func):
        for i in range(self.num_in_buffer):
            if i == 0:
                prev_memory = np.zeros((1, self.mem_size))
            else:
                prev_memory = np.expand_dims(self._encode_memory(i - 1), axis=0)
                if self.done[i-1]:
                    assert (np.array_equal(prev_memory, np.zeros((1, self.mem_size))))
            obs_input = self._encode_observation(i)
            _, _, next_memory = update_memory_func(obs_input, prev_memory)
            self.mem[i] = np.squeeze(next_memory)

    def sample_n_unique(self, n, update_memory_func=None):
        res = []
        episode_idxes_to_update = []
        while len(res) < n:
            candidate1 = random.randint(0, self.num_in_buffer - 2)
            episode_start = candidate1
            episode_end = candidate1

            if self.done[episode_start]:
                continue

            while episode_start >= 0 and not self.done[episode_start] and episode_start != self.next_idx - 1:
                episode_start -= 1
                # TODO make this wrap around!
                #if episode_start < 0 and self.num_in_buffer == self.size:
                #    episode_start =
            assert (episode_start == -1 or self.done[episode_start] or episode_start == self.next_idx - 1)
            episode_start += 1

            while episode_end < self.num_in_buffer - 1 and not self.done[episode_end] and episode_end != self.next_idx - 1:
                episode_end += 1
            assert (episode_end == self.num_in_buffer - 1 or self.done[episode_end] or episode_end == self.next_idx - 1)
            assert (episode_end != candidate1)

            if episode_end - episode_start <= 1:
                continue

            if self.sample_consecutive:
                candidate2 = candidate1 + 1
            else:
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

            '''
            # update episode memory
            if update_memory_func is not None:
                self._lazy_update_memory(episode_start, episode_end, update_memory_func)
            '''

            res.append((candidate1, candidate2))
            if (episode_start, episode_end) not in self.recently_updated_episodes:
                episode_idxes_to_update.append((episode_start, episode_end))

        # update episode memory
        if update_memory_func is not None:
            self._batch_lazy_update_memory(episode_idxes_to_update, update_memory_func)
        return np.array(res)

def test1():
    buffer = ReplayBuffer(10, 2, 5)
    assert(not buffer.can_sample(3))

    x = np.zeros((3, 3, 1))
    y = np.zeros((5,))

    '''
    Episodes are:
    1 2 3 4 
    5 6 7 8
    9 10 11 12 13 14 15 16
    '''

    idx = buffer.store_frame(np.full_like(x, 1))
    assert(not buffer.can_sample(3))
    buffer.store_effect(idx, 1, 0, False)
    buffer.store_memory(idx, np.full_like(y, 1))

    idx = buffer.store_frame(np.full_like(x, 2))
    buffer.store_effect(idx, 2, 0, False)
    buffer.store_memory(idx, np.full_like(y, 2))
    idx = buffer.store_frame(np.full_like(x, 3))
    buffer.store_effect(idx, 3, 0, False)
    buffer.store_memory(idx, np.full_like(y, 3))
    idx = buffer.store_frame(np.full_like(x, 4))
    buffer.store_effect(idx, 4, 0, True)
    buffer.store_memory(idx, np.full_like(y, 4))

    idx = buffer.store_frame(np.full_like(x, 5))
    buffer.store_effect(idx, 5, 0, False)
    buffer.store_memory(idx, np.full_like(y, 5))
    idx = buffer.store_frame(np.full_like(x, 6))
    buffer.store_effect(idx, 6, 0, False)
    buffer.store_memory(idx, np.full_like(y, 6))
    idx = buffer.store_frame(np.full_like(x, 7))
    buffer.store_effect(idx, 7, 0, False)
    buffer.store_memory(idx, np.full_like(y, 7))
    idx = buffer.store_frame(np.full_like(x, 8))
    buffer.store_effect(idx, 8, 0, True)
    buffer.store_memory(idx, np.full_like(y, 8))

    idx = buffer.store_frame(np.full_like(x, 9))
    buffer.store_effect(idx, 9, 0, False)
    buffer.store_memory(idx, np.full_like(y, 9))
    idx = buffer.store_frame(np.full_like(x, 10))
    buffer.store_effect(idx, 10, 0, False)
    buffer.store_memory(idx, np.full_like(y, 10))

    assert(buffer.can_sample(2))
    '''
    print(s[0])
    print(s[5])
    print(s[6])
    print(s[-1])
    '''

    idx = buffer.store_frame(np.full_like(x, 11))
    buffer.store_effect(idx, 11, 0, False)
    buffer.store_memory(idx, np.full_like(y, 11))
    idx = buffer.store_frame(np.full_like(x, 12))
    buffer.store_effect(idx, 12, 0, False)
    buffer.store_memory(idx, np.full_like(y, 12))
    idx = buffer.store_frame(np.full_like(x, 13))
    buffer.store_effect(idx, 13, 0, False)
    buffer.store_memory(idx, np.full_like(y, 13))
    idx = buffer.store_frame(np.full_like(x, 14))
    buffer.store_effect(idx, 14, 0, False)
    buffer.store_memory(idx, np.full_like(y, 14))
    idx = buffer.store_frame(np.full_like(x, 15))
    buffer.store_effect(idx, 15, 0, False)
    buffer.store_memory(idx, np.full_like(y, 15))
    idx = buffer.store_frame(np.full_like(x, 16))
    buffer.store_effect(idx, 16, 0, True)
    buffer.store_memory(idx, np.full_like(y, 16))
    assert (buffer.next_idx == 6)

    s = buffer.sample(1)
    print(s[0])
    print(s[5])
    print(s[6])
    print(s[-1])




if __name__ == '__main__':
    test1()
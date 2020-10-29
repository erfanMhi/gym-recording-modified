import os
import time
import logging
import numpy as np
import gym
import pickle
from gym import error

logger = logging.getLogger(__name__)


class TraceRecording(object):
    _id_counter = 0
    def __init__(self, directory=None, batch_size=10000, only_reward=False):
        """
        Create a TraceRecording, writing into directory
        """

        if directory is None:
            directory = os.path.join('/tmp', 'openai.gym.{}.{}'.format(time.time(), os.getpid()))
            os.mkdir(directory)

        self.directory = directory
        self.file_prefix = 'openaigym.trace.{}.{}'.format(self._id_counter, os.getpid())
        TraceRecording._id_counter += 1

        self.closed = False

        self.actions = []
        self.observations = []
        self.rewards = []
        self.episode_id = 0

        self.buffered_step_count = 0
        self.buffer_batch_size = batch_size
        self.only_reward = only_reward

        self.episodes_first = 0
        self.episodes = []
        self.batches = []


    def add_reset(self, observation):
        assert not self.closed
        self.end_episode()
        self.observations.append(observation)

    def add_step(self, action, observation, reward):
        assert not self.closed
        if not self.only_reward:
            self.actions.append(action)
            self.observations.append(observation)
        self.rewards.append(reward)
        self.buffered_step_count += 1

    def end_episode(self):
        """
        if len(observations) == 0, nothing has happened yet.
        If len(observations) == 1, then len(actions) == 0, and we have only called reset and done a null episode.
        """
        if len(self.observations) > 0:
            if len(self.episodes)==0:
                self.episodes_first = self.episode_id
            
            if self.only_reward:
                self.episodes.append({
                    'rewards': self.rewards,
                })
            else:
                self.episodes.append({
                    'actions': self.actions,
                    'observations': self.observations,
                    'rewards': self.rewards,
                })
 
            self.actions = []
            self.observations = []
            self.rewards = []
            self.episode_id += 1

            if self.buffered_step_count >= self.buffer_batch_size:
                self.save_complete()

    def save_complete(self):
        """
        Save the latest batch and write a manifest listing all the batches.
        We save the arrays as raw binary, in a format compatible with np.load.
        We could possibly use numpy's compressed format, but the large observations we care about (VNC screens)
        don't compress much, only by 30%, and it's a goal to be able to read the files from C++ or a browser someday.
        """

        # Each batch of data will be saved in a different file
        batch_fn = '{}.ep{:09}'.format(self.file_prefix, self.episodes_first)
        self.save_to_file(os.path.join(self.directory, batch_fn), self.episodes)
        
        self.episodes = []
        self.episodes_first = None
        self.buffered_step_count = 0

    def save_to_file(self, path, data):
        with open(path + '.pkl', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


    def close(self):
        """
        Flush any buffered data to disk and close. It should get called automatically at program exit time, but
        you can free up memory by calling it explicitly when you're done
        """
        if not self.closed:
            self.end_episode()
            if len(self.episodes) > 0:
                self.save_complete()
            self.closed = True
            logger.info('Wrote traces to %s', self.directory)


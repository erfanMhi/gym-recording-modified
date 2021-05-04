import os
import time
import logging
import numpy as np
import gym
import pickle
from gym import error
from gym_recording_modified.utils import constants
logger = logging.getLogger(__name__)

class TraceRecording(object):
    _id_counter = 0
    def __init__(self, directory=None, batch_size=None, only_reward=True, save_type='episodic_return', log_interval=10000, logger=None):
        """
        Create a TraceRecording, writing into directory
        """

        self.logger = logger
        self.save_type = save_type
        self.log_interval = log_interval

        if directory is None:
            directory = os.path.join('/tmp', 'openai.gym.{}.{}'.format(time.time(), os.getpid()))
            os.mkdir(directory)

        self.directory = directory

        self.file_prefix = constants.FILE_IDENTIFIER + '.trace.{}.{}.{}'
        self.episode_steps_file_prefix = self.file_prefix.format('episode_steps', self._id_counter, os.getpid())
        self.episode_returns_file_prefix = self.file_prefix.format('episode_returns', self._id_counter, os.getpid())
        self.reward_file_prefix = self.file_prefix.format('rewards', self._id_counter, os.getpid())
        self.observation_file_prefix = self.file_prefix.format('observations', self._id_counter, os.getpid())
        self.action_file_prefix = self.file_prefix.format('actions', self._id_counter, os.getpid())
        self.eep_file_prefix = self.file_prefix.format('episodes_end_point', self._id_counter, os.getpid())
        TraceRecording._id_counter += 1

        self.closed = False
        
        self.reset_values()
        self.episodes_end_point.append(0)
        self.episode_id = 0

        self.buffered_step_count = 0
        self.buffer_batch_size = batch_size if batch_size is not None else float('+inf')
        self.only_reward = only_reward

    def reset_values(self):
        self.actions = []
        self.observations = []
        self.rewards = []
        self.episode_returns = []
        self.episode_steps = []
        self.ep_returns_list = [] # used for storing aggregated rewards per episode each self.log_interval steps
        self.ep_steps_list = [] # used for storing aggregated steps per episode each self.log_interval steps
        self.episodes_end_point = []

    def add_reset(self, observation):
        assert not self.closed
        self.observations.append(observation)

    def add_step(self, action, observation, reward):
        assert not self.closed

        if not self.only_reward:
            self.actions.append(action)
            self.observations.append(observation)

        self.rewards.append(reward)
        self.buffered_step_count += 1
        
        if self.buffered_step_count%self.log_interval == 0:
            if self.save_type == 'episodic_return':
                self.ep_returns_list.append(np.mean(self.episode_returns))
                if self.logger is not None:
                    self.logger.info('timestep: {} - Episodic aggregated return: max: {}, min: {}, mean: {}, std: {}'.format(self.buffered_step_count, np.max(self.episode_returns), np.min(self.episode_returns), self.ep_returns_list[-1], np.std(self.episode_returns)))
                self.episode_returns = []
            elif self.save_type == 'reward_per_step':
                pass
            elif self.save_type == 'episodic_steps':
                self.ep_steps_list.append(np.mean(self.episode_steps))
                if self.logger is not None:
                    self.logger.info('timestep: {} - Episodic aggregated steps: max: {}, min: {}, mean: {}, std: {}'.format(self.buffered_step_count, np.max(self.episode_steps), np.min(self.episode_steps), self.ep_steps_list[-1], np.std(self.episode_steps)))
                self.episode_steps = []
            else:
                raise ValueError('Save_type is not defined')
 
    def end_episode(self):
        """
        if len(observations) == 0, nothing has happened yet.
        If len(observations) == 1, then len(actions) == 0, and we have only called reset and done a null episode.
        """

        if self.save_type == 'episodic_return':
            self.episode_returns.append(np.sum(self.rewards))
            self.rewards = []
        elif self.save_type == 'reward_per_step':
            pass
        elif self.save_type == 'episodic_steps':
            self.episode_steps.append(len(self.rewards))
            self.rewards = []
        else:
            raise ValueError('Save_type is not defined')
        
        self.episodes_end_point.append(self.buffered_step_count)
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

        # Creating the path to rewards, actions, and observations, and episodes end points
        episode_returns_batch_fn = '{}.ep{:09}'.format(self.episode_returns_file_prefix, self.buffered_step_count)
        episode_steps_batch_fn = '{}.ep{:09}'.format(self.episode_steps_file_prefix, self.buffered_step_count)
        rewards_batch_fn = '{}.ep{:09}'.format(self.reward_file_prefix, self.buffered_step_count)
        actions_batch_fn = '{}.ep{:09}'.format(self.action_file_prefix, self.buffered_step_count)
        observations_batch_fn = '{}.ep{:09}'.format(self.observation_file_prefix, self.buffered_step_count)
        eep_batch_fn = '{}.ep{:09}'.format(self.eep_file_prefix, self.buffered_step_count)
        
        if not self.only_reward:
            self.save_to_file(os.path.join(self.directory, observations_batch_fn), self.observations)
            self.save_to_file(os.path.join(self.directory, actions_batch_fn), self.actions)

        # Saving data
        if self.save_type == 'episodic_return':
            self.save_to_file(os.path.join(self.directory, episode_returns_batch_fn), self.ep_returns_list)
        elif self.save_type == 'reward_per_step':
            self.save_to_file(os.path.join(self.directory, rewards_batch_fn), self.rewards)
            self.save_to_file(os.path.join(self.directory, eep_batch_fn), self.episodes_end_point)
        elif self.save_type == 'episodic_steps':
            self.save_to_file(os.path.join(self.directory, episode_steps_batch_fn), self.ep_steps_list)
        else:
            raise ValueError('Save_type is not defined')
 
        self.reset_values()
        self.buffered_step_count = self.buffered_step_count % self.log_interval

    def save_to_file(self, path, data, saving_type='numpy'):
        if saving_type=='pickle':
            with open(path + '.pkl', 'wb') as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        elif saving_type=='numpy':
            np.save(path, np.array(data))
        else:
            raise ValueError('saving_type value cannot be identified: {}'.format(saving_type))

    def close(self):
        """
        Flush any buffered data to disk and close. It should get called automatically at program exit time, but
        you can free up memory by calling it explicitly when you're done
        """
        if not self.closed:
            self.save_complete()
            if len(self.rewards) > 0:
                self.save_complete()
            self.closed = True
            logger.info('Wrote traces to %s', self.directory)


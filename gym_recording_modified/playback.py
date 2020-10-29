import os
import time
import pickle
import logging
import numpy as np
from gym import error
logger = logging.getLogger(__name__)


class TraceRecordingReader:
    def __init__(self, directory):
        self.directory = directory
        self.recordings = None

    def load_pickle_file(self, path):
        with open(path + '.pkl', 'rb') as f:
            return pickle.load(f)

    def get_recorded_episodes(self):
        files = os.listdir(self.directory)
        files.sort()
        self.recordings = []
        for f in files[1:]: # Based on naming convention, first file would be args.csv
            self.recordings += self.load_pickle_file(os.path.join(self.directory, f[:-4]))

        return self.recordings

def scan_recorded_traces(directory, episode_cb=None, max_episodes=None, only_reward=False):
    """
    Go through all the traces recorded to directory, and call episode_cb for every episode.
    Set max_episodes to end after a certain number (or you can just throw an exception from episode_cb
    if you want to end the iteration early)
    """
    rdr = TraceRecordingReader(directory)
    recorded_episodes = rdr.get_recorded_episodes()
    added_episode_count = 0
    for ep in recorded_episodes:
        if not only_reward:
            assert 'observations' in ep
            assert 'actions' in ep
            episode_cb(ep['observations'], ep['actions'], ep['rewards'])
        else:
            episode_cb(ep['rewards'])
        added_episode_count += 1
        if max_episodes is not None and added_episode_count >= max_episodes: return

def get_recordings(directory, max_episodes=None, only_reward=False):
    observations = []
    actions = []
    rewards = []
    rdr = TraceRecordingReader(directory)
    recorded_episodes = rdr.get_recorded_episodes()
    added_episode_count = 0
    for ep in recorded_episodes:
        if not only_reward:
            assert 'observations' in ep
            assert 'actions' in ep
            observations.append(ep['observations'])
            actions.append(ep['actions'])
            rewards.append(ep['rewards'])
        else:
            rewards.append(ep['rewards'])
        added_episode_count += 1
        if max_episodes is not None and added_episode_count >= max_episodes: 
            if only_reward:
                return rewards
            return observations, actions, rewards
    if only_reward:
        return rewards
    return observations, actions, rewards

import os
import time
import pickle
import logging
import numpy as np
from gym import error
from typing import Union
from gym_recording_modified.utils import constants

logger = logging.getLogger(__name__)

FULL_EXTRACT = ['reward', 'observation', 'action', 'episodes_end_point', 'episode_returns', 'episode_steps'] # A list of all the extractable information

class TraceRecordingReader:

    def __init__(self, directory: str):
        self.directory = directory
        self.recordings = None

    def _load_file(self, file_path: str):
        extension = file_path[file_path.rfind('.')+1:]
        if extension == 'npy':
            return np.load(file_path)
        elif extension == 'pkl':
            with open(file_path, 'rb') as f:
                return pickle.load(f)

    def _get_files(self, extract: Union[list, str] = FULL_EXTRACT):
        files = os.listdir(self.directory)
        files_stuctured = [[] for _ in range(len(extract))]
        
        for i, t in enumerate(extract):
            for f in files:
                if constants.FILE_IDENTIFIER in f:
                    if t in f:
                        files_stuctured[i].append(f)
            files_stuctured[i].sort()

        return files_stuctured

    def get_recorded_trajectories(self, extract: Union[list, str] = FULL_EXTRACT):
        if isinstance(extract, str):
            extract = [extract]
        assert isinstance(extract, list)
        files = self._get_files(extract)
        recordings = {t:[] for t in extract}
        for i, t_files in enumerate(files):
            for f in t_files:
                recordings[extract[i]].append(self._load_file(os.path.join(self.directory, f)))
            recordings[extract[i]] = np.concatenate(recordings[extract[i]], axis=0)

        return recordings

def get_recordings(directory: str, extract: Union[list, str] = FULL_EXTRACT):
    rdr = TraceRecordingReader(directory)
    recorded_trajectories = rdr.get_recorded_trajectories(extract=extract)
    return recorded_trajectories

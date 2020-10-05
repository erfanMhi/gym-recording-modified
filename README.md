# Installation
By running the following line of code you can install this repo:
```Bash
pip install git+https://github.com/erfanMhi/gym-recording-modified.git
```


# gym-recording

A Python package to capture the sequences of actions and observations on a [Gym](https://github.com/openai/gym) environment
by wrapping it in a `TraceRecordingWrapper`, like this:

```Python
import gym
from gym_recording.wrappers import TraceRecordingWrapper

def main():
    env = gym.make('CartPole-v0')
    env = TraceRecordingWrapper(env)
    # ... exercise the environment
```

It will save recorded traces in a directory, which it will print with `logging`.
You can get the directory name from your code as `env.directory`.

Later you can play back the recording with code like the following, which runs a callback for every episode.

```Python
from gym_recording import playback

def handle_ep(observations, actions, rewards):
  # ... learn a model

playback.scan_recorded_traces(directory, handle_ep)
```

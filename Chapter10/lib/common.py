import sys
import time
import numpy as np

import torch
import torch.nn as nn


class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


class AtariPGN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(AtariPGN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)


import collections
import copy

import numpy as np

import gym
from gym import spaces
from gym import ObservationWrapper

STATE_KEY = 'state'


class PixelObservationWrapper(ObservationWrapper):
    """Augment observations by pixel values."""

    def __init__(self,
                 env,
                 pixels_only=True,
                 render_kwargs=None,
                 pixel_keys=('pixels',),
                 boxify=True):
        """Initializes a new pixel Wrapper.
        Args:
            env: The environment to wrap.
            pixels_only: If `True` (default), the original observation returned
                by the wrapped environment will be discarded, and a dictionary
                observation will only include pixels. If `False`, the
                observation dictionary will contain both the original
                observations and the pixel observations.
            render_kwargs: Optional `dict` containing keyword arguments passed
                to the `self.render` method.
            pixel_keys: Optional custom string specifying the pixel
                observation's key in the `OrderedDict` of observations.
                Defaults to 'pixels'.
            boxify: (fastrl change) Default to True where instead of a Dict, return a Box
        Raises:
            ValueError: If `env`'s observation spec is not compatible with the
                wrapper. Supported formats are a single array, or a dict of
                arrays.
            ValueError: If `env`'s observation already contains any of the
                specified `pixel_keys`.
        """

        super(PixelObservationWrapper, self).__init__(env)

        if render_kwargs is None:
            render_kwargs = {}

        for key in pixel_keys:
            render_kwargs.setdefault(key, {})

            render_mode = render_kwargs[key].pop('mode', 'rgb_array')
            assert render_mode == 'rgb_array', render_mode
            render_kwargs[key]['mode'] = 'rgb_array'

        wrapped_observation_space = env.observation_space

        if isinstance(wrapped_observation_space, spaces.Box):
            self._observation_is_dict = False
            invalid_keys = set([STATE_KEY])
        elif isinstance(wrapped_observation_space,
                        (spaces.Dict, collections.MutableMapping)):
            self._observation_is_dict = True
            invalid_keys = set(wrapped_observation_space.spaces.keys())
        else:
            raise ValueError("Unsupported observation space structure.")

        if not pixels_only:
            # Make sure that now keys in the `pixel_keys` overlap with
            # `observation_keys`
            overlapping_keys = set(pixel_keys) & set(invalid_keys)
            if overlapping_keys:
                raise ValueError("Duplicate or reserved pixel keys {!r}."
                                 .format(overlapping_keys))
            if boxify:
                raise ValueError("boxify cannot be True of pixels_only is False.")

        if pixels_only:
            self.observation_space = spaces.Dict()
        elif self._observation_is_dict:
            self.observation_space = copy.deepcopy(wrapped_observation_space)
        else:
            self.observation_space = spaces.Dict()
            self.observation_space.spaces[STATE_KEY] = wrapped_observation_space

        # Extend observation space with pixels.

        pixels_spaces = {}
        for pixel_key in pixel_keys:
            pixels = self.env.render(**render_kwargs[pixel_key])

            if np.issubdtype(pixels.dtype, np.integer):
                low, high = (0, 255)
            elif np.issubdtype(pixels.dtype, np.float):
                low, high = (-float('inf'), float('inf'))
            else:
                raise TypeError(pixels.dtype)

            pixels_space = spaces.Box(
                shape=pixels.shape, low=low, high=high, dtype=pixels.dtype)

            if boxify:
                self.observation_space = pixels_space
                break

            pixels_spaces[pixel_key] = pixels_space

        if not boxify: self.observation_space.spaces.update(pixels_spaces)

        self._env = env
        self._pixels_only = pixels_only
        self._render_kwargs = render_kwargs
        self._pixel_keys = pixel_keys
        self._boxify = boxify

    def observation(self, observation):
        pixel_observation = self._add_pixel_observation(observation)
        return pixel_observation

    def _add_pixel_observation(self, observation):
        if not self._boxify:
            if self._pixels_only:
                observation = collections.OrderedDict()
            elif self._observation_is_dict:
                observation = type(observation)(observation)
            else:
                observation = collections.OrderedDict()
                observation[STATE_KEY] = observation

            pixel_observations = {
                pixel_key: self.env.render(**self._render_kwargs[pixel_key])
                for pixel_key in self._pixel_keys
            }

            observation.update(pixel_observations)
        else:
            observation = self.env.render(**self._render_kwargs[self._pixel_keys[0]])

        return observation
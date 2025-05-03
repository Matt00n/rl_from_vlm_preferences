"""Wrapper for augmenting observations by pixel values."""
import collections
import copy
from typing import Any, Dict, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
import gymnasium as gym


GymObs = Union[Tuple, Dict[str, Any], np.ndarray, int]
GymStepReturn = Tuple[GymObs, float, bool, bool, Dict]

class PixelObservationWrapper(gym.Wrapper):
    """Augment observations by pixel values.

    Observations of this wrapper will be dictionaries of images.
    You can also choose to add the observation of the base environment to this dictionary.
    In that case, if the base environment has an observation space of type :class:`Dict`, the dictionary
    of rendered images will be updated with the base environment's observation. If, however, the observation
    space is of type :class:`Box`, the base environment's observation (which will be an element of the :class:`Box`
    space) will be added to the dictionary under the key "state".

    Example:
        >>> import gym
        >>> env = PixelObservationWrapper(gym.make('CarRacing-v1', render_mode="rgb_array"))
        >>> obs = env.reset()
        >>> obs.keys()
        odict_keys(['pixels'])
        >>> obs['pixels'].shape
        (400, 600, 3)
        >>> env = PixelObservationWrapper(gym.make('CarRacing-v1', render_mode="rgb_array"), pixel_keys=('obs',))
        >>> obs = env.reset()
        >>> obs.keys()
        odict_keys(['obs'])
        >>> obs['obs'].shape
        (400, 600, 3)
    """

    def __init__(
        self,
        env: gym.Env,
        render_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        pixel_keys: Tuple[str, ...] = ("pixels",),
    ):
        """Initializes a new pixel Wrapper.

        Args:
            env: The environment to wrap.
            render_kwargs (dict): Optional dictionary containing that maps elements of ``pixel_keys``to
                keyword arguments passed to the :meth:`self.render` method.
            pixel_keys: Optional custom string specifying the pixel
                observation's key in the ``OrderedDict`` of observations.
                Defaults to ``(pixels,)``.

        Raises:
            AssertionError: If any of the keys in ``render_kwargs``do not show up in ``pixel_keys``.
            ValueError: If ``env``'s observation space is not compatible with the
                wrapper. Supported formats are a single array, or a dict of
                arrays.
            ValueError: If ``env``'s observation already contains any of the
                specified ``pixel_keys``.
            TypeError: When an unexpected pixel type is used
        """
        super().__init__(env)

        # Avoid side-effects that occur when render_kwargs is manipulated
        render_kwargs = copy.deepcopy(render_kwargs)
        # self.render_history = []

        if render_kwargs is None:
            render_kwargs = {}

        for key in render_kwargs:
            assert key in pixel_keys, (
                "The argument render_kwargs should map elements of "
                "pixel_keys to dictionaries of keyword arguments. "
                f"Found key '{key}' in render_kwargs but not in pixel_keys."
            )

        default_render_kwargs = {}
        if not env.render_mode:
            raise AttributeError(
                "env.render_mode must be specified to use PixelObservationWrapper:"
                "`gym.make(env_name, render_mode='rgb_array')`."
            )

        for key in pixel_keys:
            render_kwargs.setdefault(key, default_render_kwargs)

        self._render_kwargs = render_kwargs
        self._pixel_keys = pixel_keys
    
    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        pixel_obs = self._add_pixel_observation_to_info()
        info.update(pixel_obs)
        return observation, reward, terminated, truncated, info
    
    def _add_pixel_observation_to_info(self):
        observation = collections.OrderedDict()
        pixel_observations = {
            pixel_key: jnp.array(self._render(**self._render_kwargs[pixel_key]))
            for pixel_key in self._pixel_keys
        }
        observation.update(pixel_observations)
        return observation

    def _render(self, *args, **kwargs):
        render = self.env.render(*args, **kwargs)
        # if isinstance(render, list):
        #     self.render_history += render
        return render
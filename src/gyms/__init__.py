import gymnasium as gym
from .tracking import TrackingEnv
from ..common.global_vars import TRACKING_GYM, TRACKING_PATH_LEN

if TRACKING_GYM not in gym.envs.registry:
    gym.register(
        id=TRACKING_GYM, entry_point=TrackingEnv, max_episode_steps=TRACKING_PATH_LEN
    )

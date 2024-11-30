import gymnasium as gym
from .tracking import TrackingEnv
from ..common.global_vars import TRACKING_GYM

if TRACKING_GYM not in gym.envs.registry:
    gym.register(id=TRACKING_GYM, entry_point=TrackingEnv, max_episode_steps=600)

"""Registers the internal gym envs then loads the env plugins for module using the entry point."""

from typing import Any

from gymnasium.envs.registration import make, pprint_registry, register, registry, spec


register(
    id="CartPendulum-v0",
    entry_point="gymnasium.envs.custom.cart_pendulum:CartPendulumEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)



# --- For shimmy compatibility
def _raise_shimmy_error(*args: Any, **kwargs: Any):
    raise ImportError(
        'To use the gym compatibility environments, run `pip install "shimmy[gym-v21]"` or `pip install "shimmy[gym-v26]"`'
    )


# When installed, shimmy will re-register these environments with the correct entry_point
register(id="GymV21Environment-v0", entry_point=_raise_shimmy_error)
register(id="GymV26Environment-v0", entry_point=_raise_shimmy_error)

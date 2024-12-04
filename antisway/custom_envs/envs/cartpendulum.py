import numpy as np 
import os

from gymnasium import utils 
from gymnasium.envs.mujoco import MujocoEnv 
from gymnasium.spaces import Box 

from typing import Dict, Union

from return_handler import return_handler

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 5,
    "lookat": np.array([0, 0, 1]),
    "elevation": 5,
}

class CartPendulumEnv(MujocoEnv, utils.EzPickle):
    """
    ### Description

    ### Action Space
    The agent take a 1-element vector for actions.

    The action space is a continuous `(action)` in `[-3, 3]`, where `action` represents
    the numerical force applied to the cart (with magnitude representing the amount of
    force and sign representing the direction)

    | Num | Action                    | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit      |
    |-----|---------------------------|-------------|-------------|----------------------------------|-------|-----------|
    | 0   | Force applied on the cart | -3          | 3           | slider                           | slide | Force (N) |

    ### Observation Space

    The state space consists of positional values of different body parts of
    the pendulum system, followed by the velocities of those individual parts (their derivatives)
    with all the positions ordered before all the velocities.

    The observation is a `ndarray` with shape `(4,)` where the elements correspond to the following:

    | Num | Observation                                   | Min  | Max | Name (in corresponding XML file) | Joint | Unit                      |
    | --- | --------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------- |
    | 0   | position of the cart along the linear surface | -Inf | Inf | slider                           | slide | position (m)              |
    | 1   | vertical angle of the pole on the cart        | -Inf | Inf | hinge                            | hinge | angle (rad)               |
    | 2   | linear velocity of the cart                   | -Inf | Inf | slider                           | slide | velocity (m/s)            |
    | 3   | angular velocity of the pole on the cart      | -Inf | Inf | hinge                            | hinge | anglular velocity (rad/s) |


    ### Rewards


    ### Starting State
    All observations start in state
    (0.0, 0.0, 0.0, 0.0) with a uniform noise in the range
    of [-0.01, 0.01] added to the values for stochasticity.

    ### Episode End
    The episode ends when any of the following happens:

    1. Truncation: The episode duration reaches 1000 timesteps.
    2. Termination: Any of the state space values is no longer finite.
    3. Termination: The absolutely value of the vertical angle between the pole and the cart is greater than 0.2 radian.
    """


    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }
    


    def __init__(
        self,
        xml_file: str = os.path.join(os.path.dirname(__file__), "assets", "cart_pendulum.xml"),
        frame_skip: int = 1,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        reset_noise_scale: float = 0.01,
        **kwargs,
    ):
        utils.EzPickle.__init__(self, xml_file, frame_skip, reset_noise_scale, **kwargs)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

        self._reset_noise_scale = reset_noise_scale

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        self.observation_structure = {
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel.size,
        }




    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()

        cart_position = observation[0]  # Not used in this reward
        pole_angle = observation[1]
        cart_velocity = observation[2]
        pole_angular_velocity = observation[3]

        terminated = bool(
            not np.isfinite(observation).all() or (np.abs(pole_angle) > .2) or cart_position > 1.5
            #not np.isfinite(observation).all() or (np.abs(pole_angle) > 0.05236) or cart_position > 1.5
        )

        reward = self.reward_function(action, cart_position, pole_angle, cart_velocity, pole_angular_velocity)  


        info = {"reward_survive": reward}

        if self.render_mode == "human":
            self.render()
        
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`

        return_handler(velocity_rl = cart_velocity, angle_rl = pole_angle, action_rl = action)

        return observation, reward, terminated, False, info



    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=noise_low, high=noise_high
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=noise_low, high=noise_high
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return_handler(velocity_rl=observation[2],angle_rl=observation[1])
        #return_handler(plot = True)

        return self._get_obs()



    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()
    



    def reward_function(self, action, cart_position, pole_angle, cart_velocity, pole_angular_velocity):

        target_velocity = 0.2
        
        angle_penalty_factor = 4.0
        velocity_penalty_factor = 0.5
        target_velocity_penalty_factor = 2.0
        angular_velocity_penalty_factor = 1.0

        # Penalize large angle
        angle_penalty = abs(pole_angle) * angle_penalty_factor

        # Penalize angular velocity (swing)
        angular_velocity_penalty = abs(pole_angular_velocity) * angular_velocity_penalty_factor

        # Reward for reaching and maintaining target velocity
        velocity_penalty = abs(cart_velocity - target_velocity) * velocity_penalty_factor
        target_velocity_penalty = target_velocity_penalty_factor * max(0, 1 - velocity_penalty)

        reward = -angle_penalty - angular_velocity_penalty - velocity_penalty + target_velocity_penalty

        # Small negative reward for energy efficiency or large actions
        #energy_penalty = -abs(action) * 0.1  # Penalize large actions
        #reward += energy_penalty

        return reward




    '''
    This reward function is working but can prob be improved

    def reward_function(self, action, cart_position, pole_angle, cart_velocity, pole_angular_velocity):

        target_velocity = 0.2
        
        angle_penalty_factor = 2.0
        velocity_penalty_factor = 0.5
        target_velocity_penalty_factor = 2.0
        angular_velocity_penalty_factor = 1.0

        # Penalize large angle
        angle_penalty = abs(pole_angle) * angle_penalty_factor

        # Penalize angular velocity (swing)
        angular_velocity_penalty = abs(pole_angular_velocity) * angular_velocity_penalty_factor

        # Reward for reaching and maintaining target velocity
        velocity_penalty = abs(cart_velocity - target_velocity) * velocity_penalty_factor
        target_velocity_penalty = target_velocity_penalty_factor * max(0, 1 - velocity_penalty)

        reward = -angle_penalty - angular_velocity_penalty - velocity_penalty + target_velocity_penalty

        # Small negative reward for energy efficiency or large actions
        energy_penalty = -abs(action) * 0.01  # Penalize large actions
        reward += energy_penalty

        return reward
        



    This one works better than the previous one

    def reward_function(self, action, cart_position, pole_angle, cart_velocity, pole_angular_velocity):

        target_velocity = 0.2
        
        angle_penalty_factor = 4.0
        velocity_penalty_factor = 0.5
        target_velocity_penalty_factor = 2.0
        angular_velocity_penalty_factor = 1.0

        # Penalize large angle
        angle_penalty = abs(pole_angle) * angle_penalty_factor

        # Penalize angular velocity (swing)
        angular_velocity_penalty = abs(pole_angular_velocity) * angular_velocity_penalty_factor

        # Reward for reaching and maintaining target velocity
        velocity_penalty = abs(cart_velocity - target_velocity) * velocity_penalty_factor
        target_velocity_penalty = target_velocity_penalty_factor * max(0, 1 - velocity_penalty)

        reward = -angle_penalty - angular_velocity_penalty - velocity_penalty + target_velocity_penalty

        # Small negative reward for energy efficiency or large actions
        energy_penalty = -abs(action) * 0.1  # Penalize large actions
        reward += energy_penalty

        return reward
        '''

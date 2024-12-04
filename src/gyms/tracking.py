import gymnasium as gym
from gymnasium import spaces
import numpy as np

from ..common.global_vars import TRACKING_PATH_LEN, TRACKING_MAX_VEL


class TrackingParameters:
    GRAVITY = 9.81

    def __init__(
        self,
        l: float = 0.47,
        frac_supp: float = 1.0,
        M_t: float = 2.092,
        M_p: float = 0.765,
    ):
        self.l = l
        self.F_supp = frac_supp * M_p * self.GRAVITY
        self.M_t = M_t
        self.M_p = M_p


class TrackingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        trk_params: TrackingParameters = TrackingParameters(),
        reference: np.ndarray = None,
        T: float = 0.005,
        render_mode=None,
        scramble_trk_params=False,
        reset_reference=False,
    ):
        # Tracking Parameters
        self.trk_params = trk_params

        # A state is just [x_t, v_t, a_t, theta] + F_supp
        # There are 3 levels for tustin transform
        self.state = np.zeros((3, 4))

        # The action is just F_out (again, 3 levels for calulcation)
        self.action = np.zeros((3, 1))

        # Timestep
        self.T = T

        # Reference
        self.reference = (
            np.array(reference)
            if reference is not None
            else self._generate_reference_()
        )

        # Iteration
        self.i = -1

        # Reset Options
        self.reset_reference = reset_reference
        self.scramble_trk_params = scramble_trk_params  #

        self.render_mode = render_mode

        # Finally, set the action and observation spaces
        self.action_space = spaces.Box(-80, 80, [1])
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            [
                5,
            ],
        )

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.i = -1
        if options is not None:
            reference = options["reference"] if "reference" in options else None
            trk_params = options["trk_params"] if "trk_params" in options else None
        if reference is not None:
            self.reference = reference
        elif self.reset_reference:
            self.reference = self._generate_reference_()
        if trk_params is not None:
            self.trk_params = trk_params
        elif self.scramble_trk_params:
            self.trk_params = TrackingParameters(
                self.np_random.uniform(0.1, 2.0),
                self.np_random.random(),
                self.np_random.uniform(1.5, 4.0),
                self.np_random.uniform(0.01, 1.5),
            )

        # A state is just [x_t, v_t, a_t, theta] + F_supp
        # There are 3 levels for tustin transform
        # Assumption: theta_init = 0
        self.state = np.zeros((3, 4))

        # The action is just F_out (again, 3 levels for calulcation)
        self.action = np.zeros((3, 1))

        return (
            np.array([*self.state[0], self.trk_params.F_supp], dtype=np.float32),
            dict(),
        )

    def step(self, action):
        # Step 0, update the iteration
        self.i += 1

        while hasattr(action, "__iter__"):
            assert len(action) == 1
            action = action[0]

        # Step 1, push action into vector and push state vector
        self.action = np.roll(self.action, 1, axis=0)
        self.action[0][0] = action
        self.state = np.roll(self.state, 1, axis=0)

        # Step 2, calculate the RHS
        rhs = self._calculate_rhs_()
        rhs = rhs[0] + 2 * rhs[1] + rhs[2]

        # Step 3, calculate the past state residuals
        C_0 = 4 * self.trk_params.M_t / (self.T * self.T)
        C_1 = self.trk_params.F_supp / self.trk_params.l
        lh_res = 2 * (-C_0 + C_1) * self.state[1, 0] + (C_0 + C_1) * self.state[2, 0]

        # Step 3, calculate the new position (state[0][0])
        # self.state[0][0] = np.clip(
        #     (rhs - lh_res) / (C_0 + C_1),
        #     self.reference[self.i] - self.trk_params.l + 1e-6,
        #     self.reference[self.i] + self.trk_params.l - 1e-6,
        # )
        self.state[0][0] = (rhs - lh_res) / (C_0 + C_1)

        # Step 4, propagate the position into velocity and acceleration
        truncated = not self._propogate_state_()

        # Step 5, prepare the observation and return the result
        observation = np.array(
            [*self.state[0, 1:], self.action[0, 0], self.trk_params.F_supp],
            dtype=np.float32,
        )
        reward = self._calculate_reward_()
        terminated = self.i == len(self.reference) - 1

        return observation, reward, terminated, truncated, dict()

    def render(self):
        pass

    def close(self):
        pass

    def _calculate_reward_(self):
        # The reward is as follows:
        # [Penalty] |trolley state - prev reference state|
        # [Penalty] |Force|
        # [Reward] |previous angle| - |angle now|
        # [Reward] [(trolley state - last trolley state)*sign(reference - trolley state) - abs(reference - trolley state)] / l

        PEN_TO_REF_FACTOR = 5.0
        PEN_FORCE_FACTOR = 2.5
        REW_ANG_FACTOR = 30.0
        REW_CLOSE_FACTOR = 20.0
        REW_TRACK_FACTOR = 5.0

        pen_to_ref = -np.abs(self.state[0, 0] - self.reference[self.i - 1])
        pen_force_factor = -np.abs(self.action[0, 0])
        rew_ang = np.deg2rad(3) - np.abs(self.state[0, 3])

        rew_close = np.abs(self.state[0, 3]) - np.abs(self.state[1, 3])
        rew_track = self.trk_params.l - np.abs(
            self.reference[self.i] - self.state[0, 0]
        )

        return (
            PEN_TO_REF_FACTOR * pen_to_ref
            + PEN_FORCE_FACTOR * pen_force_factor
            + REW_ANG_FACTOR * rew_ang
            + REW_CLOSE_FACTOR * rew_close
            + REW_TRACK_FACTOR * rew_track
        )

        # return -(((self.state[0, 0] - self.reference[self.i]) / self.trk_params.l) ** 2)
        # return -np.abs(self.state[0, 3])

    def _calculate_rhs_(self):
        reference_states = np.array(
            [
                self.reference[j] if j >= 0 else 0
                for j in reversed(range(self.i - 2, self.i + 1))
            ]
        )
        # F_in + F_supp * tan(arcsin((x_t - x_p)/l))
        return (
            self.action[:, 0]
            + self.trk_params.F_supp / self.trk_params.l * reference_states
        )

    def _propogate_state_(self):
        # Calculate x'(t) = 2/T * (x(t) - x(t-1)) - x'(t-1)
        self.state[0, 1] = (
            2 / self.T * (self.state[0, 0] - self.state[1, 0]) - self.state[1, 1]
        )
        self.state[0, 2] = (
            2 / self.T * (self.state[0, 1] - self.state[1, 1]) - self.state[1, 2]
        )
        # Calculate theta = arcsin((x_p-x_t)/l)
        if np.abs(self.reference[self.i] - self.state[0, 0]) > self.trk_params.l:
            self.state[0, 3] = (
                np.sign(self.reference[self.i] - self.state[0, 0]) * np.pi / 2
            )
            return False
        self.state[0, 3] = np.arcsin(
            (self.reference[self.i] - self.state[0, 0]) / self.trk_params.l
        )
        return True

    def _generate_reference_(self, p: float = 0.5, size=TRACKING_PATH_LEN):
        samples = self.np_random.normal(0.1, 1) * TRACKING_MAX_VEL * self.T
        samples[0] = 0
        for i in range(1, len(samples)):
            samples[i] += samples[i - 1]
        return samples

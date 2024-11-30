import numpy as np

from ..gyms.tracking import TrackingParameters

from typing import Tuple, Iterable


class TrackingReferencePolicy:
    def __init__(
        self,
        trk_params: TrackingParameters,
        settling_time: float = 0.1,
        overshoot: float = 0.05,
    ):
        """Initializes a tuned Impedance-Proportional Controller (PI Controller) for Tracking Mode

        Args:
            trk_params (TrackingParameters): The Tracking Parameters
            settling_time (float, optional): The setting time to tune to. Defaults to 0.1.
            overshoot (float, optional): The overshoot fraction to tune to. Defaults to 0.05.
        """
        self.optimize(trk_params, settling_time, overshoot)

    def __call__(self, observation: Iterable[float]) -> float:
        """Executes the policy

        Args:
            obs (Iterable[float, float]): The observation, which
            should be a vector of [x_t, v_t, a_t, theta, F_supp]
            for tracking mode (obtained from the
            gyms.tracking.TrackingEnv)

        Returns:
            float: The action corresponding to the observation
        """
        return -(self.K * observation[-2] + self.B * observation[1])

    def optimize(
        self,
        trk_params: TrackingParameters,
        settling_time: float = 0.1,
        overshoot: float = 0.05,
    ) -> Tuple[float, float, float]:
        """
        Optimizes Parameters for System

        :param settling_time: The desired settling time
        :param overshoot: The desired overshoot, between 0.0
        and 1.0
        :return: The damping, inner, and outer loop gains
        for the system
        """
        # The poles are at 0 and -B_m/M_t
        # The meeting point is at -B_m/(2*M_t),
        # so we must control that to control the
        # settling time. We can then introduce
        # a gain to then cause the system to have
        # a particular overshoot.
        B = 8 * trk_params.M_t / settling_time

        zeta = -np.log(overshoot) / (np.pi**2 + np.log(overshoot) ** 2) ** 0.5

        # Desired Gains
        K_inner = (B / (2 * trk_params.M_t * zeta)) ** 2 * trk_params.M_t
        K_outer = K_inner * trk_params.l

        K = trk_params.l - K_outer + trk_params.F_supp

        self.B, self.K = B, K

import gymnasium as gym
import numpy as np

class TrackingParameters:
    GRAVITY = 9.81
    def __init__(self, l : float = 0.47, frac_supp : float = 0.0, M_t : float = 2.092, M_p : float = 0.765):
        self.l = l
        self.F_supp = frac_supp * M_p * self.GRAVITY
        self.M_t = M_t
        self.M_p = M_p

class TrackingEnv(gym.Env):
    metadata = {"render_modes" : []}
    
    def __init__(self, trk_params : TrackingParameters, reference : np.ndarray | None = None, T : float = 0.005, render_mode = None, scramble_trk_params = True):
        # Tracking Parameters
        self.trk_params = trk_params
        
        # A state is just [x_t, v_t, a_t, theta] + F_supp
        # There are 3 levels for tustin transform
        self.state = np.zeros((3, 4))
        
        # The action is just F_out (again, 3 levels for calulcation)
        self.action = np.zeros((3, 1))
        
        # Reference
        self.reference = reference if reference is not None else self._generate_reference_()
        
        # Timestep
        self.T = T
        
        # Iteration
        self.i = -1
        
        # Reset Options
        self.generated_reference = reference is None
        self.scramble_trk_params = scramble_trk_params#
        
        self.render_mode = render_mode
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.i = -1
        if self.generated_reference:
            self.reference = self._generate_reference_()
        if self.scramble_trk_params:
            self.trk_params = TrackingParameters(*self.np_random.uniform(0.0, 3.0, 4))
    
    def step(self, action):
        # Step 0, update the iteration
        self.i += 1
        
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
        lh_res = 2 * (-C_0+C_1) * self.state[1, 0] + (C_0 + C_1) * self.state[2, 0]
        
        # Step 3, calculate the new position (state[0][0])
        self.state[0][0] = (rhs - lh_res) / (C_0 + C_1)
        
        # Step 4, propagate the position into velocity and acceleration
        self._propogate_state_()
        
        # Step 5, prepare the observation and return the result        
        observation = np.array([*self.state[0], self.trk_params.F_supp])
        reward = -np.abs(self.reference[self.i] - self.state[0, 0])
        truncated = self.i == len(self.reference) - 1
        
        return observation, reward, False, truncated, None
    
    def render(self):
        pass

    def close(self):
        pass
    
    def _calculate_rhs_(self):
        reference_states = np.array([self.reference[j] if j >= 0 else 0 for j in reversed(range(self.i - 2, self.i + 1))])
        # F_in + F_supp * tan(arcsin((x_t - x_p)/l))
        return self.action[:, 0] + self.trk_params.F_supp / self.trk_params.l * reference_states
        
    
    def _propogate_state_(self):
        # Calculate x'(t) = 2/T * (x(t) - x(t-1)) - x'(t-1)
        self.state[0, 1] = 2 / self.T * (self.state[0, 0] - self.state[1, 0]) - self.state[1, 1]
        self.state[0, 2] = 2 / self.T * (self.state[0, 1] - self.state[1, 1]) - self.state[1, 2]
        # Calculate theta = arcsin((x_p-x_t)/l)
        self.state[0, 3] = np.arcsin((self.reference[self.i] - self.state[0, 0]) / self.trk_params.l)

    def _generate_reference_(self, p : float = 0.5, size = 2000):
        samples = 2 * self.np_random.binomial(n=1, p=p, size=size) - 1.0
        for i in range(1, len(samples)):
            samples[i] -= samples[i - 1]
        return samples
import matplotlib.pyplot as plt

from .gyms.tracking import TrackingEnv, TrackingParameters
from .tracking.reference_generator import gen_ramp_disturbance
from .tracking.reference_policy import TrackingReferencePolicy

# This currently runs a basic diagnostic on Tracking Mode

# Generate our System
trk_params = TrackingParameters()

# Generate the reference (position of person moving on the ground)
t, reference = gen_ramp_disturbance(
    [(0.5, 0.5), (1.0, 1.0), (1.0, 2.0), (3.0, 2.5), (0.0, 3.0)]
)

# Generate the environment
env = TrackingEnv(trk_params, reference=reference)

# Generate the policy
policy = TrackingReferencePolicy(trk_params)

# Begin Evaluation
observation, _, _, truncated, _ = env.step(0)  # Generate first step

x_t = [0]  # Keep this buffer for seeing the system position over time

while not truncated:
    action = policy(observation)  # Get the action

    x_t.append(observation[0])  # Record position

    observation, _, _, truncated, _ = env.step(action)  # Get next observation

# Plot Response
plt.figure()
plt.plot(t, reference, label="Reference")
plt.plot(t, x_t, label="Response")
plt.legend()
plt.show()

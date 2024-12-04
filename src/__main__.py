# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from .gyms import *
# from .gyms.tracking import TrackingEnv, TrackingParameters
# from .tracking.reference_generator import gen_ramp_disturbance
# from .tracking.reference_policy import TrackingReferencePolicy
# from .common.global_vars import MODEL_DIR, device
# from .policies.policy_gradient import PGPolicy
# from .policies.q_learning import ActorCriticAgent

# # This currently runs a basic diagnostic on Tracking Mode

# # # Generate our System
# trk_params = TrackingParameters()

# # Generate the reference (position of person moving on the ground)
# t, reference = gen_ramp_disturbance([(0.5, 0.5), (1.0, 1.0), (1.0, 2.0), (0.0, 3.0)])

# # Generate the environment
# # env = TrackingEnv(trk_params, reference=reference)
# env = gym.make(TRACKING_GYM, reference=reference)

# env.reset()

# # Generate the policy
# # policy = TrackingReferencePolicy(trk_params)
# # policy = torch.load(MODEL_DIR + "policy_gradient.pth")
# # policy = PGPolicy(5, 2, reference_controller=TrackingReferencePolicy())
# agent = ActorCriticAgent(
#     5,
#     1,
#     [
#         float(env.action_space.low.min()),
#         float(env.action_space.high.max()),
#     ],
#     device,
# )
# agent.load(MODEL_DIR + "actor_critic.pth")
# policy = agent.actor

# # Begin Evaluation
# observation, reward, terminated, truncated, _ = env.step(0)  # Generate first step

# x_t = [observation[0]]  # Keep this buffer for seeing the system position over time
# rewards = [reward]

# while not (terminated or truncated):
#     # action = policy(observation)  # Get the action
#     action, _, _ = policy(torch.from_numpy(observation))
#     # action = policy.mu
#     action = action.item()
#     observation, reward, terminated, truncated, _ = env.step(
#         action
#     )  # Get next observation

#     x_t.append(env.env.env.env.state[0, 0])  # Record position
#     # x_t.append(env.state[0, 0])
#     rewards.append(reward)

# print(f"Terminated: {terminated}, Truncated: {truncated}")

# print(f"Average Reward {np.mean(np.array(rewards))}")
# print(x_t[-1])

# # Plot Response
# plt.figure()
# plt.title("Policy Gradient Response")
# plt.xlabel("Time (s)")
# plt.ylabel("Position (m)")
# plt.plot(env.env.env.env.reference, label="Reference")
# # plt.plot(env.reference, label="Reference")
# plt.plot(x_t, label="Response")
# # plt.plot(rewards, label="Reward")
# plt.legend()
# plt.savefig("stats.png")
# plt.show()

from .experiment.train import train_bc, train_pg, train_ac, train_sac
from .gyms import *
from .experiment.evaluate import evaluate_bc

evaluate_bc()
# train_bc()
# train_pg()
# train_ac()
# train_sac()

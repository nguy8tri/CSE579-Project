import torch
import gymnasium as gym

from ..gyms import *
from ..policies.policy_gradient import PGPolicy, PGBaseline, PGTrainer
from ..common.global_vars import MODEL_DIR, TRACKING_GYM


def train_pg():
    env = gym.make(TRACKING_GYM)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]

    policy = PGPolicy(input_size, output_size)
    baseline = PGBaseline(input_size, output_size)
    trainer = PGTrainer(env, policy, baseline)

    trainer.train_model()

    torch.save(policy, MODEL_DIR + "policy_gradient.pth")

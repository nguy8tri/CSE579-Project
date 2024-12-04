import torch
import gymnasium as gym
import os

from ..policies.behavior_cloning import BehaviorCloningPolicy, BehaviorCloningTrainer

from ..tracking.reference_policy import TrackingReferencePolicy

from ..policies.q_learning import ActorCriticAgent, ActorCriticTrainer, SACAgent

print(os.path.abspath("."))

from ..gyms import *
from ..policies.policy_gradient import PGPolicy, PGBaseline, PGTrainer
from ..common.global_vars import MODEL_DIR, TRACKING_GYM, device
from ..tracking.reference_generator import gen_ramp_disturbance
from ..common.policy_defs import ReplayBuffer

t, reference = gen_ramp_disturbance([(0.5, 0.5), (1.0, 1.0), (1.0, 2.0), (0.0, 3.0)])


def train_bc():
    env = gym.make(TRACKING_GYM, reference=reference, scramble_trk_params=True)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]

    try:
        policy = torch.load(MODEL_DIR + "behavior_cloning.pth")
    except:
        print("Loading new Behavior Cloning Policy")
        policy = BehaviorCloningPolicy()
    trainer = BehaviorCloningTrainer(env, policy)

    trainer.train_model()

    torch.save(policy, MODEL_DIR + "behavior_cloning.pth")


def train_pg():
    env = gym.make(TRACKING_GYM, reference=reference)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]

    try:
        policy = torch.load(MODEL_DIR + "policy_gradient.pth")
    except Exception as e:
        print("Loading new Policy")
        policy = PGPolicy(input_size, output_size)

    try:
        baseline = torch.load(MODEL_DIR + "policy_gradient_baseline.pth")
    except:
        print("Loading New Baseline")
        baseline = PGBaseline(input_size, output_size)
    trainer = PGTrainer(env, policy, baseline)

    try:
        trainer.train_model()
    except KeyboardInterrupt:
        pass
    finally:
        print("Saving PG Model")
        torch.save(policy, MODEL_DIR + "policy_gradient.pth")
        torch.save(baseline, MODEL_DIR + "policy_gradient_baseline.pth")


def train_ac():
    env = gym.make(TRACKING_GYM, reference=reference)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max()),
    ]

    num_train_steps = 100_000

    buffer = ReplayBuffer(input_size, output_size, num_train_steps, device)

    agent = ActorCriticAgent(input_size, output_size, action_range, device)

    try:
        agent.load(MODEL_DIR + "actor_critic.pth")
        print("Loaded a saved agent")
    except:
        pass

    trainer = ActorCriticTrainer(agent, env, buffer, num_train_steps)

    try:
        trainer.train_agent()
    except KeyboardInterrupt:
        pass
    finally:
        agent.save(MODEL_DIR + "actor_critic.pth")


def train_sac():
    env = gym.make(TRACKING_GYM, reference=reference)
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.shape[0]
    action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max()),
    ]

    num_train_steps = 100_000

    buffer = ReplayBuffer(input_size, output_size, num_train_steps, device)

    agent = SACAgent(
        input_size,
        output_size,
        action_range,
        device,
        alpha_lr=3e-4,
        init_temperature=0.1,
        target_entropy=-output_size,
        critic_tau=0.05,
        double_critic=True,
        temperature=True,
    )

    try:
        agent.load(MODEL_DIR + "soft_actor_critic.pth")
        print("Loaded a saved agent")
    except:
        pass

    trainer = ActorCriticTrainer(agent, env, buffer, num_train_steps)

    try:
        trainer.train_agent()
    except KeyboardInterrupt:
        pass
    finally:
        agent.save(MODEL_DIR + "soft_actor_critic.pth")

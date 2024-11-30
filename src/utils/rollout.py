import torch
import numpy as np

from utils.layers import eval_mode

from ..common.global_vars import device
from ..policies.policy_gradient import PGPolicy


def rollout(
    env,
    agent,
    replay_buffer=None,
    render=False,
):
    # Collect the following data
    raw_obs = []
    raw_next_obs = []
    actions = []
    rewards = []
    dones = []
    images = []
    done = False
    o, _ = env.reset()
    episode_step = 0
    if render:
        env.render()

    while not done:
        o_for_agent = o.copy()
        if isinstance(agent, PGPolicy):
            o_for_agent = torch.from_numpy(o_for_agent[None]).to(device).float()
            action, _, _ = agent(o_for_agent)
            action = action.cpu().detach().numpy()[0]
        else:
            with eval_mode(agent):
                action = agent.act(o_for_agent, sample=False)
        # Step the simulation forward
        next_o, r, terminated, truncated, _ = env.step(copy.deepcopy(action))
        done = terminated or truncated
        done_no_max = done_no_max = (
            0 if episode_step + 1 == env.spec.max_episode_steps else done
        )
        if replay_buffer is not None:
            replay_buffer.add(o, action, r, next_o, done, done_no_max)

        # Render the environment
        if render:
            env.render()
        raw_obs.append(o)
        raw_next_obs.append(next_o)
        actions.append(action)
        rewards.append(r)
        dones.append(done)
        if done:
            break
        o = next_o
        episode_step += 1

    # Prepare the items to be returned
    observations = np.array(raw_obs)
    next_observations = np.array(raw_next_obs)
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    rewards = np.array(rewards)
    if len(rewards.shape) == 1:
        rewards = rewards.reshape(-1, 1)
    dones = np.array(dones).reshape(-1, 1)

    # Return in the following format
    return dict(
        observations=observations,
        next_observations=next_observations,
        actions=actions,
        rewards=rewards,
        dones=np.array(dones).reshape(-1, 1),
        images=np.array(images),
    )

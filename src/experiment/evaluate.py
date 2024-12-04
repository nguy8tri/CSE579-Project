import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt

from ..gyms.tracking import TrackingParameters

from ..tracking.reference_policy import TrackingReferencePolicy

from ..common.global_vars import (
    MODEL_DIR,
    RESULTS_DIR,
    TRACKING_GYM,
    TRACKING_MAX_VEL,
    TRACKING_PATH_LEN,
    reference,
)


def get_timestamp(T=0.005):
    return np.linspace(0, TRACKING_PATH_LEN * T, TRACKING_PATH_LEN)


def generate_testing_reference(function, T=0.005):
    return function(get_timestamp(T))


def generate_random_reference(T=0.005):
    samples = np.random.normal(0, 3, TRACKING_PATH_LEN) * T
    samples[0] = 0
    for i in range(1, len(samples)):
        samples[i] += samples[i - 1]
    return samples


functions = [
    lambda x: x,
    lambda x: x * x,
    lambda x: np.sin(x),
    lambda x: np.tan(0.4 * x),
]

references = (
    [reference]
    + [generate_testing_reference(fun) for fun in functions]
    + [generate_random_reference() for _ in range(4)]
)


def generate_trk_params():
    return TrackingParameters(
        np.random.uniform(0.1, 2.0),
        np.random.random(),
        np.random.uniform(1.5, 4.0),
        np.random.uniform(0.01, 1.5),
    )


def get_response(env, policy, reference, trk_params):
    state, _ = env.reset(options={"reference": reference, "trk_params": trk_params})

    response = []
    reward = 0
    maximum_angle = 0

    action = policy(torch.from_numpy(state))

    truncated, terminated = False, False

    while not (truncated or terminated):
        action = action.item()
        state, r, terminated, truncated, _ = env.step(action)

        response.append(env.env.env.env.state[0, 0])

        action = policy(torch.from_numpy(state))
        reward += r
        if np.abs(env.env.env.env.state[0, 3]) > maximum_angle:
            maximum_angle = np.abs(env.env.env.env.state[0, 3])

    return np.array(response), r / len(response), np.rad2deg(maximum_angle)


def evaluate(policy_path, plot_path, title):
    # Get initial States
    env = gym.make(TRACKING_GYM, reference=reference, scramble_trk_params=True)
    t = get_timestamp()

    # Get initial tracking params
    trk_params = TrackingParameters()

    fig, axs = plt.subplots(3, 3)
    fig.suptitle(title)
    fig.supylabel("Position (m)")
    fig.supxlabel("Time (s)")
    # Get policies
    policy_ref = TrackingReferencePolicy()
    policy_test = torch.load(policy_path)

    for i, r in enumerate(references):
        print(f"============================")
        print(f"Reference {i}")
        response_ref, reward_ref, max_ang_ref = get_response(
            env, policy_ref, r, trk_params
        )
        response_test, reward_test, max_ang_tst = get_response(
            env, policy_test, r, trk_params
        )

        print(
            f"System Parameters -> M_t: {trk_params.M_t}, M_p: {trk_params.M_p}, F_supp: {trk_params.F_supp}, l: {trk_params.l}"
        )
        print(f"Reference -> Reward: {reward_ref}, Angle: {max_ang_ref}")
        print(f"Test -> Reward: {reward_test}, Angle: {max_ang_tst}")

        row = i // 3
        col = i % 3

        axs[row, col].plot(t, r, label="Reference")
        axs[row, col].plot(t, response_ref, "-", label="Control Policy")
        axs[row, col].plot(t, response_test, "--", label="BC Policy")
        if i == 0:
            axs[row, col].legend()

        # Regenerate Parameters
        trk_params = generate_trk_params()
        policy_ref.optimize(trk_params)
        print(f"============================")
    plt.show()
    fig.savefig(plot_path)


def evaluate_bc():
    evaluate(
        MODEL_DIR + "behavior_cloning.pth",
        RESULTS_DIR + "behavior_cloning.png",
        "Evaluation for Behavior Cloned Model",
    )

import numpy as np

from typing import List, Tuple


def gen_ramp_disturbance(positions: List[Tuple[float, float]], X0=0, T=0.005):
    """Generates a function that travels from position to position through ramps

    Args:
        positions (List[Tuple[float, float]]): A list of tuples that has 1) The
            position to get to, and 2) When to get to that position
        X0 (int, optional): The initial position. Defaults to 0.
        T (float, optional): The timestep. Defaults to 0.005.

    Returns:
        _type_: _description_
    """
    last_pos = X0
    last_t = 0
    result = np.array([])
    for next_pos, next_t in positions:
        next_values = (
            np.arange(last_pos, next_pos, (next_pos - last_pos) / (next_t - last_t) * T)
            if not next_pos == last_pos
            else np.array([next_pos] * int((next_t - last_t) / T))
        )
        result = np.append(result, next_values, axis=-1)
        last_pos = next_pos
        last_t = next_t
    return np.arange(0.0, positions[-1][-1], T), result

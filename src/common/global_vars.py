import torch

from ..tracking.reference_generator import gen_ramp_disturbance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SRC_DIR = "src/"

MODEL_DIR = SRC_DIR + "model_images/"
RESULTS_DIR = SRC_DIR + "results/"

# Gym Names
TRACKING_GYM = "Tracking-v0"

# Tracking Hyperparameters
TRACKING_PATH_LEN = 600
TRACKING_MAX_VEL = 3

t, reference = gen_ramp_disturbance([(0.5, 0.5), (1.0, 1.0), (1.0, 2.0), (0.0, 3.0)])

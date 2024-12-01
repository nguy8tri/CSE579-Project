import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SRC_DIR = "src/"

MODEL_DIR = SRC_DIR + "model_images/"

# Gym Names
TRACKING_GYM = "Tracking-v0"

# Tracking Hyperparameters
TRACKING_PATH_LEN = 60
TRACKING_MAX_VEL = 10

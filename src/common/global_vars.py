import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "model_images/"

# Gym Names
TRACKING_GYM = "Tracking-v0"

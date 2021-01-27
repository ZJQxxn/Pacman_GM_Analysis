import pickle as pkl
import pandas as pd
import numpy as np


# data = pd.read_csv("all_trial.csv")
# num = data.shape[0]
#
# local_dir = [np.random.choice(["left","right","up","down"], 1).item() for _ in range(num)]
# global_dir = [np.random.choice(["left","right","up","down"], 1).item() for _ in range(num)]
# evade_blinky_dir = [np.random.choice(["left","right","up","down"], 1).item() for _ in range(num)]
# evade_clyde_dir = [np.random.choice(["left","right","up","down"], 1).item() for _ in range(num)]
# approach_dir = [np.random.choice(["left","right","up","down"], 1).item() for _ in range(num)]
# energizer_dir = [np.random.choice(["left","right","up","down"], 1).item() for _ in range(num)]
# multi_dir = [np.random.choice(["left","right","up","down"], 1).item() for _ in range(num)]
#
# fitted_label = [np.random.choice(["global","local","evade(Blinky)","evade(Clyde)","Approach","Energizer","Vague"], 1) for _ in range(num)]
#
# data["local_dir"] = local_dir
# data["global_dir"] = global_dir
# data["evade_blinky_dir"] = evade_blinky_dir
# data["evade_clyde_dir"] = evade_clyde_dir
# data["approach_dir"] = approach_dir
# data["energizer_dir"] = energizer_dir
# data["multi_dir"] = multi_dir
# data["fitted_label"] = fitted_label
#
# data.to_csv("test_data.csv")

# data = pd.read_csv("test_data.csv")
# num = data.shape[0]
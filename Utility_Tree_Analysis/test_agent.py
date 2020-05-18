import pandas as pd
import json
import numpy as np
from MultiAgentInteractor import MultiAgentInteractor


data = pd.read_csv("diary.csv")
data = data.iloc[-1]

multiagent = MultiAgentInteractor(config_file = "config.json")

multiagent.resetStatus(
    eval(data.pacmanPos),
    eval(data.energizers),
    eval(data.beans),
    [eval(data.ghost1_pos), eval(data.ghost2_pos)],
    np.nan,
    np.nan,
    [data.ghost1_status, data.ghost2_status]
)
multiagent.last_dir = "left"

print(multiagent.estimateDir())
import pandas as pd
import numpy as np
import pickle as pkl

with open("evade_to_local.pkl", "rb") as file:
    data = pkl.load(file)
print()

evades = data[["file", "label_evade1", "label_evade2", "label_evade", "beans"]]
evades["reward_num"] = evades.apply(lambda x: len(x.beans) if not isinstance(x.beans, float) else 0, axis = 1)
evades = evades.fillna(0)
evades = evades[["file", "reward_num", "label_evade"]]
print()
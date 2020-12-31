import numpy as np


planned = np.load("8000_trial_data_Omega-with_Q-descriptive-record.npy", allow_pickle=True)
accidental = np.load("7000_trial_data_Patamon-with_Q-descriptive-record.npy", allow_pickle=True)

all = np.concatenate([planned, accidental])

np.save("all_trial_data-descriptive-record.npy", all)
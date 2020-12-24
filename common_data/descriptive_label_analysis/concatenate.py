import numpy as np


planned = np.load("planned_200_trial_data_Omega-with_Q-descriptive-record.npy", allow_pickle=True)
accidental = np.load("accidental_200_trial_data_Omega-with_Q-descriptive-record.npy", allow_pickle=True)

all = np.concatenate([planned, accidental])

np.save("all_200_trial_data_Omega-descriptive-record.npy", all)
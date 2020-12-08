import pickle
import pandas as pd

with open("1000_trial_data_Omega-with_Q-with_weight-window3-new_agents.pkl", "rb") as file:
    data_1000 = pickle.load(file)
    print("1000 Trial Data Shape : ", data_1000.shape)

with open("2000_trial_data_Omega-with_Q-with_weight-window3-new_agents.pkl", "rb") as file:
    data_2000 = pickle.load(file)
    print("2000 Trial Data Shape : ", data_2000.shape)

all_data = pd.concat([data_1000, data_2000])
print("All Trial Data Shape : ", all_data.shape)

with open("3000_trial_data_Omega-with_Q-with_weight-window3-new_agents.pkl", "wb") as file:
    pickle.dump(all_data, file)
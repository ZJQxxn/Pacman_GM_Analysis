import pickle
import pandas as pd

with open("8000_trial_data_Omega-with_Q-descriptive-with_weight-window3-new_agents.pkl", "rb") as file:
    data_1000 = pickle.load(file)
    print("8000 Trial Data Shape : ", data_1000.shape)

with open("7000_trial_data_Patamon-with_Q-descriptive-with_weight-window3-new_agents.pkl", "rb") as file:
    data_2000 = pickle.load(file)
    print("7000 Trial Data Shape : ", data_2000.shape)

all_data = pd.concat([data_1000, data_2000])
print("All Trial Data Shape : ", all_data.shape)

with open("all_trial_data-window3-descriptive.pkl", "wb") as file:
    pickle.dump(all_data, file)

# with open("suicide_100_trial_data_Omega-with_Q.pkl", "rb") as file:
#     data = pickle.load(file)
# print(data.columns.values)
#
# data = data.drop(columns = ["global_Q", "local_Q", "pessimistic_Q", "planned_hunting_Q", "suicide_Q"])
# with open("suicide_100_trial_data_Omega.pkl", "wb") as file:
#     pickle.dump(data, file)

# with open("global_100_trial_data_Omega-with_Q.pkl", "rb") as file:
#     data = pickle.load(file)
# print(data.columns.values)
#
# data = data.drop(columns = ["global_Q", "local_Q", "pessimistic_Q", "planned_hunting_Q", "suicide_Q"])
# with open("global_100_trial_data_Omega.pkl", "wb") as file:
#     pickle.dump(data, file)
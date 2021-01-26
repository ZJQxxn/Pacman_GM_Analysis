import pickle

with open("/home/qlyang/Documents/pacman/constants/all_data_new.pkl", "rb") as f:
    data = pickle.load(f)
    print(list(data.keys()))
import pickle

with open("diary.pkl", 'rb') as file:
    data = pickle.load(file)

# pos = data['pacmanPos'][70:80]
# dir = data['pacman_dir'][70:80]
# moving_dir = data['possible_dir'][70:80]

for each in data:
    if each not in ["date", "trialid", "energizers", "beans"]:
        print(each)
        print(data[each][190:200])
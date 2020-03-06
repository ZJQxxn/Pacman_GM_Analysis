'''
Description:
    Extract data. 
    
uthor:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    2020/3/5
'''
import csv


def extractData(filename):
    '''
    Extract required features for every time step.
    :return: VOID
    '''
    data = []
    with open('df_total_GM.csv', 'r') as file:
        reader = csv.DictReader(file)
        count = 0
        for row in reader:
            features = [
                        count, # index
                        row['file'], # File name (different trials)
                        row['remain_scared_time1'], # Remained scared time for ghost 1
                        row['remain_scared_time2'], # Remained scared time for ghost 2
                        row['distance1'], # Distance between Pacman and ghost 1
                        row['distance2'], # Distance between Pacman and ghost 2
                        row['rwd_pac_distance'], # Distance between Pacman and all the normal dots
                        row['energizers'], # Distance between Pacman and all the big dots
                        row['ifscared1'], # If ghost 1 is scared (1 for normal; 2 for scared)
                        row['ifscared2'], # If ghost 2 is scared (1 for normal; 2 for scared)
                        row['pacmanPos'], # Current position of Pacman
                        row['ghost1Pos'], # Current location of ghost 1
                        row['ghost2Pos'], # Current location of ghost 2
                        row['ghost1_dir'], # Ghost 1 moving direction (up/down/left/right)
                        row['ghost2_dir'] # Ghost 2 moving direction (up/down/left/right)
                        ]
            data.append(features)
            count += 1
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def determineLabel(filename, label_filename):
    '''
    Determine their labels based on the mode of the next time step ([1, 0] for grazing model and [0, 1] for hunting mode). 
    Therefore, the last sample won't be used in training because we can't determine its mode.
    :param cur_time: Features for current time step.
    :param next_time: Features for the next time step.
    :return: VOID
    '''
    data = []
    label = []
    hunt_count = 0
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    for index in range(len(data)-1):
        # Determine the label based on the next time step
        next_time = data[index+1]
        next_time_mode = [0, 1, 'hunting'] if int(float(next_time[8])) == 2 or int(float(next_time[9])) == 2 else [1, 0, 'grazing']
        if [0,1,'hunting'] == next_time_mode:
            hunt_count += 1
        label.append(next_time_mode)
    print("The number of hunting Pacman is {} \n The number of grazing Pacman is {}".format(
                        hunt_count, len(data)-1-hunt_count
    ))
    with open(label_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(label)


def determineMode(filename, mode_filename):
    '''
    Determine the mode of the current time step ([1, 0] for grazing model and [0, 1] for hunting mode). 
    :param filename: Features filename.
    :param label_filename: Modes filename
    :return: VOID
    '''
    data = []
    modes = []
    hunt_count = 0
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    for index in range(len(data)):
        # Determine the modes for the current time step
        cur_time = data[index]
        cur_time_mode = [0, 1, 'hunting'] if int(float(cur_time[8])) == 2 or int(float(cur_time[9])) == 2 else [1, 0, 'grazing']
        if [0, 1, 'hunting'] == cur_time_mode:
            hunt_count += 1
        modes.append(cur_time_mode)
    print("The number of hunting Pacman is {} \n The number of grazing Pacman is {}".format(
                        hunt_count, len(data)-1-hunt_count
    ))
    with open(mode_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(modes)


if __name__ == '__main__':
    filename = 'data/extract_feature.csv'
    label_filename = 'data/all_labels.csv'
    mode_filename = 'data/all_modes.csv'

    # # Extract features
    # extractData(filename)

    # # Determine lables based on the next time step
    # determineLabel(filename, label_filename)
    #
    # # Determine modes based for the current time step
    # determineMode(filename, mode_filename)
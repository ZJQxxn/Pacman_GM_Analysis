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
    with open('data/df_total_GM.csv', 'r') as file:
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
                        row['ghost2_dir'], # Ghost 2 moving direction (up/down/left/right)1
                        row['status_g'], # Whether grazing (1 for true, 0 for false)
                        row['status_h1'], # Whether hunting ghost 1 (1 for true, 0 for false)
                        row['status_h2'], # Whether hunting ghost 2 (1 for true, 0 for false)
                        row['Step'] # No. of this time step 
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
    hunt1_count = 0
    hunt2_count = 0
    grazing_count = 0
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    for index in range(len(data)-1):
        # Determine the label based on the next time step
        next_time = data[index+1]
        next_time_mode = whichMode(next_time[15], next_time[16], next_time[17])
        if [0,1,0, 'hunting1'] == next_time_mode:
            hunt1_count += 1
        elif [0, 0, 1, 'hunting2'] == next_time_mode:
            hunt2_count += 1
        elif [1, 0, 0, 'grazing'] == next_time_mode:
            grazing_count += 1
        label.append(next_time_mode)
    print("The number of hunting1 Pacman is {} \n "
          "The number of hunting2 Pacman is {} \n "
          "The number of grazing Pacman is {}".format(
        hunt1_count, hunt2_count, grazing_count
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
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    for index in range(len(data)):
        # Determine the modes for the current time step
        cur_time = data[index]
        cur_time_mode = whichMode(cur_time[15], cur_time[16], cur_time[17])
        modes.append(cur_time_mode)
    with open(mode_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(modes)


def whichMode(status_g, status_h1, status_h2):
    #TODO: change for different settings
    '''
    Determin the mode based on some status.
    :param status_g: Whether in the grazing status.
    :param status_h1: Whether in the hunting ghost 1 status.
    :param status_h2: Whether in the hunting ghost 2 status.
    :return: The mode. 
             [0,0,0,'escaping'], 
             [1,0,0,'grazing'], 
             [0,1,0,'hunting1']
             or [0,0,1,'hunting2'].
    '''
    status_g = int(float(status_g))
    status_h1 = int(float(status_h1))
    status_h2 = int(float(status_h2))
    mode = None
    if not status_g and  not status_h1 and not status_h2:
        mode = [0, 0, 0, 'escaping']
    elif status_g:
        mode = [1, 0, 0, 'grazing']
    elif status_h1:
        mode = [0, 1, 0, 'hunting1']
    else:
        mode = [0, 0, 1, 'hunting2']
    return mode



if __name__ == '__main__':
    filename = 'data/extract_feature.csv'
    label_filename = 'data/split_all_labels.csv'
    mode_filename = 'data/split_all_modes.csv'

    # # Extract features
    # extractData(filename)

    # Determine lables based on the next time step
    determineLabel(filename, label_filename)

    # Determine modes based for the current time step
    determineMode(filename, mode_filename)
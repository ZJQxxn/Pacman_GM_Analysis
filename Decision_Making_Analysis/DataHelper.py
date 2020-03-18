'''
Description:
    Extract data. 
    
uthor:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    2020/3/5
'''
import csv
from util import str2List


def extractData(filename):
    '''
    Extract required features for every time step.
    :return: VOID
    '''
    data = []
    with open('../common_data/df_total_GM.csv', 'r') as file:
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
    if 1 == status_h1 or 1 == status_h2:
        if status_h1:
            mode = [0, 1, 0, 'hunting1']
        else:
            mode = [0, 0, 1, 'hunting2']
    else :
        if 1 == status_g:
            mode = [1, 0, 0, 'grazing']
        else:
            mode = [0, 0, 0, 'escaping']
    return mode


# ==================================================================
#               EXTRACT ONLY USEFUL G2H DATA
# ==================================================================

def extractGHData(feature_filename, label_filename, mode_filename):
    '''
    Extract data that grazing first and then hunting after has eaten an energizer.
    :param filename: Feature filename.
    :return: VOID
    '''
    data = {}
    mode = {}
    with open('../common_data/df_total_GM.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            trial_name = row['file']
            if trial_name not in data:
                data[trial_name] = []
                mode[trial_name] = []
            data[trial_name].append(row)
            mode[trial_name].append(whichGHMode(row))
    selected_data = []
    selected_label = []
    selected_mode = []
    for each_trial in data:
        trial_data, trial_label, trial_mode = filterDirectHunting(data[each_trial], mode[each_trial])
        selected_data.extend(trial_data)
        selected_label.extend(trial_label)
        selected_mode.extend(trial_mode)
    # Save features, labels, and modes
    with open(feature_filename, 'w', newline = '') as file:
        writer = csv.DictWriter(file, fieldnames = [each for each in selected_data[0].keys()])
        writer.writeheader()
        writer.writerows(selected_data)
    with open(label_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(selected_label)
    with open(mode_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(selected_mode)


def filterDirectHunting(trial_data, trial_mode):
    '''
    Filter out data that directly hunting after energizers, for each trial.
    :param trial_data: A dict of trial data.
    :return: Processed trial data.
    '''
    # The Pacman is not escaping denotes it has eaten energizers.
    step_num = len(trial_data)
    # Find out the time step where energizers are eaten
    energizer_point = []
    for index in range(step_num-1):
        step_data = trial_data[index]
        next_step_data = trial_data[index + 1]
        if 0 == len(str2List(step_data['energizers'])):
            break
        if len(str2List(next_step_data['energizers'])) < len(str2List(step_data['energizers'])):
            energizer_point.append(index)
     # Split data and filter out directly hunting time
    if 0 == len(energizer_point):
        return [], [], []
    else:
        # Take scared data after each generizer point (block data)
        energizer_point.append(step_num - 1) # discard the last time step
        blk_data = {}
        blk_label = {}
        blk_mode = {}
        for index in range(len(energizer_point) - 1):
            blk_data[index] = trial_data[
                                energizer_point[index]:energizer_point[index+1]
                              ]
            blk_label[index] = trial_mode[
                                energizer_point[index] + 1:energizer_point[index + 1] + 1
                              ]
            blk_mode[index] = trial_mode[
                                energizer_point[index]:energizer_point[index + 1]
                              ]
        # Filter out data that directly hunting after energizer
        processed_data = []
        processed_label = []
        processed_mode = []
        for index in range(len(energizer_point) - 1):
            start_index = 0
            end_index = 0
            step = 0
            # Find where the pure grazing starts
            while ('1.0' == blk_data[index][step]['status_h1']
                or '1.0' == blk_data[index][step]['status_h2']):
                step += 1
                if (step >= len(blk_data[index])):
                    break
            start_index = step
            step += 1
            # Find where the hunting ends (i.e., ghosts becomes normal or dead)
            while step < len(blk_data[index]) \
                    and ('' != blk_data[index][step]['remain_scared_time1']
                    or '' != blk_data[index][step]['remain_scared_time2']):
                step += 1
            end_index = step

            # Select out data between this phase. And filter out escaping data, if any
            for inner_idnex in range(start_index, end_index):
                if 'escaping' != blk_label[index][inner_idnex][3] \
                        and 'escaping' != blk_mode[index][inner_idnex][3] \
                        and 'grazing' == blk_mode[index][inner_idnex][3]:
                    processed_data.append(blk_data[index][inner_idnex])
                    processed_label.append(blk_label[index][inner_idnex])
                    processed_mode.append(blk_mode[index][inner_idnex])
        return processed_data, processed_label, processed_mode


def whichGHMode(time_step_data):
    status_g = int(float(time_step_data['status_g']))
    status_h1 = int(float(time_step_data['status_h1']))
    status_h2 = int(float(time_step_data['status_h2']))
    mode = None
    if 1 == status_h1 or 1 == status_h2:
        if 0 == status_h1:
            mode = [0, 0, 1, 'hunting2']
        elif 0 == status_h2:
            mode = [0, 1, 0, 'hunting1']
        else:
            mode = [0, 1, 1, 'hunting_all']
    else:
        if 1 == status_g:
            mode = [1, 0, 0, 'grazing']
        else:
            mode = [0, 0, 0, 'escaping']
    return mode


if __name__ == '__main__':
    filename = 'extracted_data/extract_feature.csv'
    label_filename = 'extracted_data/split_all_labels.csv'
    mode_filename = 'extracted_data/split_all_modes.csv'
    less_feature_filename = 'extracted_data/less_G2H_feature.csv'
    less_label_filename = 'extracted_data/less_G2H_label.csv'
    less_mode_filename = 'extracted_data/less_G2H_mode.csv'

    # # Extract features
    # extractData(filename)

    # # Determine lables based on the next time step
    # determineLabel(filename, label_filename)
    #
    # # Determine modes based for the current time step
    # determineMode(filename, mode_filename)

    extractGHData(less_feature_filename, less_label_filename, less_mode_filename)
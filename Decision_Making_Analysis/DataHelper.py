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
            features = [count, # index
                        row['file'], # File name (different trials)
                        row['remain_scared_time1'], # Remained scared time for ghost 1
                        row['remain_scared_time2'], # Remained scared time for ghost 2
                        row['distance1'], # Distance between Pacman and ghost 1
                        row['distance2'], # Distance between Pacman and ghost 2
                        row['rwd_pac_distance'], # Distance between Pacman and all the normal dots
                        row['energizers'], # Distance between Pacman and all the big dots
                        row['ifscared1'], # If ghost 1 is scared (1 for normal; 2 for scared)
                        row['ifscared2'], # If ghost 2 is scared (1 for normal; 2 for scared)
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


def determineMode(cur_time, next_time):
    '''
    Determine their labels based on the mode of the next time step ([1, 0] for grazing model and [0, 1] for hunting mode).
    :param cur_time: Features for current time step.
    :param next_time: Features for the next time step.
    :return: VOID
    '''
    # if 0 == count:
    #     pre_time = features
    #     continue
    # else:
    #     cur_time = features
    #     cur_label = determineMode(pre_time, cur_time)
    #     pre_time = cur_time
    pass


def preprocess():
    '''
    Preprocessing for features.
    :return: VOID
    '''
    pass

if __name__ == '__main__':
    extractData('extract_feature.csv')
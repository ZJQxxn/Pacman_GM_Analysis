import pandas as pd
import sys
sys.path.append('./')
from MultiAgentInteractor import MultiAgentInteractor
import numpy as np
multiagent = MultiAgentInteractor(config_file = 'config.json')
df=pd.read_csv('diary.csv')
last_row=df.iloc[-1] # 116; 229; 286; 332
print(last_row[['pacmanPos','ghost1_pos','ghost2_pos','possible_dir']])
multiagent.resetStatus(eval(last_row.pacmanPos), 
eval(last_row.energizers), 
eval(last_row.beans), 
np.array([last_row.ghost1_distance,last_row.ghost2_distance]),
last_row.fruit_type, 
eval(last_row.fruit_pos),
np.array([last_row.ghost1_status, last_row.ghost2_status]))
estimated=multiagent.estimateDir()
print(estimated)



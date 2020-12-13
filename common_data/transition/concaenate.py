import pandas as pd
import pickle

# For local, global, attack agents
with open("local_to_global-with_Q.pkl", "rb") as file:
    local2global = pickle.load(file)
    local2global.file = local2global.file.apply(lambda x: x + "-local2global")

with open("global_to_local-with_Q.pkl", "rb") as file:
    global2local = pickle.load(file)
    global2local.file = global2local.file.apply(lambda x: x + "-global2local")

with open("local_to_planned-with_Q.pkl", "rb") as file:
    local2planned = pickle.load(file)
    local2planned.file = local2planned.file.apply(lambda x: x + "-local2planned")

local_global_attack_data = pd.concat([local2global, global2local, local2planned]).reset_index(drop = True)
with open("../trial/local_global_attack-with_Q.pkl", "wb") as file:
    pickle.dump(local_global_attack_data, file)
print("Finished saving local+global+attack data ", local_global_attack_data.shape)
# ======================================

# For local, evade, suicide agents
with open("local_to_evade-with_Q.pkl", "rb") as file:
    local2evade = pickle.load(file)
    local2evade.file = local2evade.file.apply(lambda x: x + "-local2evade")

with open("evade_to_local-with_Q.pkl", "rb") as file:
    evade2local = pickle.load(file)
    evade2local.file = evade2local.file.apply(lambda x: x + "-evade2local")

with open("local_to_suicide-with_Q.pkl", "rb") as file:
    local2suicide = pickle.load(file)
    local2suicide.file = local2suicide.file.apply(lambda x: x + "-local2suicide")

local_evade_suicide_data = pd.concat([local2evade, evade2local, local2suicide]).reset_index(drop = True)
with open("../trial/local_evade_suicide-with_Q.pkl", "wb") as file:
    pickle.dump(local_evade_suicide_data, file)
print("Finished saving local+evade+suicide data ", local_evade_suicide_data.shape)
# ======================================

# For all agents
all_agents_data = pd.concat([local2global, global2local, local2planned, local2evade, evade2local, local2suicide]).reset_index(drop = True)
with open("../trial/all_agent-with_Q.pkl", "wb") as file:
    pickle.dump(all_agents_data, file)
print("Finished saving all agent data ", all_agents_data.shape)
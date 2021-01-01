import pandas as pd
import numpy as np

data1 = pd.read_csv("1-1-Omega-09-Jul-2019-1.csv")
data2 = pd.read_csv("1-1-Patamon-14-Jul-2019-1.csv")
data3 = pd.read_csv("10-4-Omega-18-Jul-2019-2.csv")
data4 = pd.read_csv("23-1-Omega-21-Aug-2019-1.csv")
res = pd.concat([data1, data2, data3, data4])
res_col = res.columns.values
res = res.rename(columns={"label_true_planned_hunting":"label_planned_hunting"})
res["label_planned_hunting"][res["label_planned_hunting"] == 0.0] = np.nan
print(len(res.columns.values))

out_data = pd.read_csv("./vedio_code/vedio_code/data/out.csv")
out_column = out_data.columns.values
print(len(out_data.columns.values))
#
# temp = []
# for each in res_col:
#     if each not in out_column:
#         temp.append(each)
# print(temp)

# res = res.drop(columns = temp)

res.to_csv("all_trial_simplify.csv")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def eval_df(df_total, l):
    """
    读取数据的时候，把一些string格式的column转化为本身的数据format
    """
    for c in l:
        df_total[c] = df_total[c].apply(
            lambda x: eval(x) if isinstance(x, str) else np.nan
        )
    return df_total



MAP_INFO = eval_df(pd.read_csv("../common_data/map_info_brian.csv"), ["pos", "pos_global"])

T, F = True, False
ARRAY = np.asarray(
    MAP_INFO.pivot_table(columns="Pos1", index="Pos2")
    .iswall.reindex(range(MAP_INFO.Pos2.max() + 1))
    .replace({1: F, np.nan: F, 0: T})
)
ARRAY = np.concatenate((ARRAY, np.array([[False] * 30])))
ARRAY= ARRAY[1:, 1:-1]
COSTS = np.where(ARRAY, 1, 1000)
print("Finished reading data.")

f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(ARRAY, ax=ax, linewidth=0.5, annot=False, cbar=False, cmap="bone", xticklabels=range(1, ARRAY.shape[1]+1), yticklabels=range(1, ARRAY.shape[0]+1))
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
# ax.set_title(k)
plt.show()
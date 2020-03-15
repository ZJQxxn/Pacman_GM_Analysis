###==========preps===============
from pacmanutils import *

df_total = pd.read_csv("df_total_GM.csv", index_col=0).rename(
    columns={"Unnamed: 0": "index"}
)

for c in [
    "ghost1Pos",
    "ghost2Pos",
    "pacmanPos",
    "nearrwdPos",
    "before_last",
    "after_first",
    "pos",
    "previousPos",
    "possible_dirs",
    "next_eat_rwd",
    "nearbean_dir",
    "energizers",
    "ghost1_wrt_pacman",
]:
    df_total[c] = df_total[c].apply(lambda x: eval(x) if isinstance(x, str) else np.nan)

Rewards_dict = pickle.load(open("Rewards_dict.p", "rb"))
Process_dict = pickle.load(open("Process_dict.p", "rb"))


#####==========ploting===============
sel_file = "16-2-Omega-21-Aug-2019" ## !!! DO NOT add .csv to file name
df_explore = df_total[df_total.file == sel_file + ".csv"].set_index("index")

w = interact(
    plot_eating_all,
    k=fixed(sel_file),
    df_pos=fixed(df_explore),
    Rewards_dict=fixed(Rewards_dict),
    idx=(
        widgets.IntSlider(
            min=df_explore.index.min(),
            max=df_explore.index.max(),
            step=1,
            value=59,
        )
    ),
)
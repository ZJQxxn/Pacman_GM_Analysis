#TODO: some undefined variables

energizer_data = (
    df_total.reset_index()
    .merge(energizer_data, on=["file", "next_eat_energizer"])
    .set_index("level_0")
)

energizer_data = energizer_data[energizer_data.apply(lambda x: x.name in x[0], 1)]

energizer_last_step = [list(j)[-1] for j in consecutive_groups(energizer_data.index)]
energizer_step = [list(j) for j in consecutive_groups(energizer_data.index)]


pre_hunt_index = [
    energizer_step[energizer_last_step.index(i[0] - 1)]
    for i in index_list
    if df_total.loc[i, "status_h" + which_ghost].mean() >= 0.95
    and i[0] - 1 in energizer_last_step
]

df_total.loc[
    list(itertools.chain(*pre_hunt_index)), "status_h" + which_ghost
] = "prehunt"
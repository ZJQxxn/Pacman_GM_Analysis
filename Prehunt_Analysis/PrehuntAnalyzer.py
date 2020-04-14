'''
Description:
    Analyzing the pre-hunt data.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date: 
    Apr. 3 2020
'''

import numpy as np
import pandas as pd
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append('./')
from AnalysisUtils import locs_df, cross_pos
from AnalysisUtils import determine_region, dijkstra_distance


class PrehuntAnalyzer:

    def __init__(self, feature_file):
        '''
        Initialization.
        :param feature_file: Filename for features. 
        '''
        self.data = pd.read_csv(feature_file)
        for c in [
            "pacmanPos",
            "energizers",
            "beans"
        ]:
            self.data[c] = self.data[c].apply(lambda x: eval(x) if not isinstance(x, float) else np.nan)
        # Only take data with normal ghosts.
        # Because we want to analyze why the Pacman choose to eat the energizer when ghosts are normal
        self.data = self.data[(self.data.ifscared1 <= 2) & (self.data.ifscared2 <= 2)]


    def _TurnRatio(self, trial_data, cross_pos):
        '''
        Given a sequence of time steps, determine whether the Pacman turns at a certain crossroads 
        :param trial_data: 
        :return: 
        '''
        # No enough data to determine whether the Pacman passes by the cross
        if trial_data.shape[0] <= 1:
            return np.nan
        # Determine the cross position
        is_pass_cross = trial_data.apply(
            lambda x: x.pacmanPos in cross_pos, axis = 1
        ).rename("is_cross")
        # If the Pacman hasn't walk on the cross position or the Pacman pass the cross at the beginning of pre-hunt
        if np.all(is_pass_cross == False) or (is_pass_cross.iloc[0] and np.all(is_pass_cross.iloc[1:] == False)) :
            return np.nan
        # Determine the number of cross positions where the Pacman choose to turn its direction
        turn_num = 0
        pre_dir = trial_data.shift_pacman_dir.iloc[0]
        for index in range(1, trial_data.shape[0]):
            cur_dir = trial_data.shift_pacman_dir.iloc[index]
            if is_pass_cross.iloc[index]:
                if cur_dir != pre_dir:
                    turn_num += 1
            pre_dir = cur_dir
        return turn_num / is_pass_cross.sum()


    def analyzeInertia(self):
        pre_hunt_data = self.data.dropna(subset = ["shift_pacman_dir"])
        # The number of cross positions on the path from Pacman to the energizer
        path_cross_num_data = (
            pre_hunt_data.groupby(["file"]).apply(
                lambda x: np.sum([each in cross_pos for each in x.pacmanPos])
            )
                .rename("cross_num")
                .reset_index()
        )
        # The number of cross positions where the Pacman chooses to turn its direction, on the path from Pacman to the energizer
        turn_num_data = (
            pre_hunt_data.groupby(["file"]).apply(
                lambda x: self._TurnRatio(x, cross_pos)
            )
                .rename("turn_percent")
                .reset_index()
        )
        # The mode of Pacman after eating an energizer (hunt / graze)
        prehunt_mode_data = (
            self.data.groupby(["file"]).apply(
                lambda x: x.after_energizer_mode.iloc[0]
            )
                .rename("prehunt_mode")
                .reset_index()
        )
        # Integrate data
        integrated_data = (
            prehunt_mode_data.merge(
                turn_num_data,
                on = ["file"],
                how = "left"
            ).merge(
                path_cross_num_data,
                on = ['file'],
                how = "left"
            ).dropna(subset = ['turn_percent'])
        )
        # # The number of data with very few cross positions and very few turning numbers
        # few_cross_few_turn = integrated_data[(integrated_data.cross_num <= 1) & (integrated_data.turn_percent <= 0.1)]
        # much_cross_much_turn = integrated_data[(integrated_data.cross_num >= 5) & (integrated_data.turn_percent >= 0.9)]
        print()
        # Plot the histogram of cross number
        plt.figure(figsize=(25, 10))
        plt.subplot(1, 3, 1)
        plt.title("All Pre-Hunt", fontsize=20)
        sns.distplot(integrated_data.cross_num, kde=False, bins=9,color="#1BA3F9")
        plt.xlabel('Number of Cross', fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.yticks(fontsize=20)
        plt.subplot(1, 3, 2)
        plt.title("Directly Hunt [median = {}]".format(
            integrated_data.turn_percent[integrated_data.prehunt_mode == "hunt"].median().round(decimals = 2)
            ),
            fontsize = 20)
        sns.distplot(integrated_data.turn_percent[integrated_data.prehunt_mode == "hunt"], kde = False, bins = 10, color="#1BA3F9")
        sns.distplot(
            integrated_data[(integrated_data.cross_num <= 2) & (integrated_data.turn_percent <= 0.1) & (integrated_data.prehunt_mode == "hunt")].turn_percent,
            bins = np.arange(0, 1.0, 0.1),
            kde = False,
            color = "#ad5050",
            label = "# cross $\leq$ 2, turn percent $\leq$ 0.1",
            hist_kws = {"alpha" :0.8, "hatch": '\\',"edgecolor": "w"}
        )
        plt.xlabel('Turn Percentage', fontsize=20)
        plt.xticks(np.arange(0,1.1,0.1), fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.yticks(np.arange(0, 51, 10), np.arange(0, 51, 10), fontsize=20)
        plt.legend(fontsize = 17)

        plt.subplot(1, 3, 3)
        plt.title("Graze Firstly [median = {}]".format(
            integrated_data.turn_percent[integrated_data.prehunt_mode == "graze"].median().round(decimals=2)
        ), fontsize = 20)
        sns.distplot(integrated_data.turn_percent[integrated_data.prehunt_mode == "graze"], kde=False, bins = 10, color="#1BA3F9")
        sns.distplot(
            integrated_data[(integrated_data.cross_num <= 2) & (integrated_data.turn_percent <= 0.1) & (
            integrated_data.prehunt_mode == "graze")].turn_percent,
            bins=np.arange(0, 1.0, 0.1),
            kde=False,
            color="#ad5050",
            label = "# cross $\leq$ 2, turn percent $\leq$ 0.1",
            hist_kws={"alpha": 0.8, "hatch": '\\', "edgecolor": "w"}
        )
        plt.xlabel('Turn Percentage', fontsize=20)
        plt.xticks(np.arange(0,1.1,0.1), fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig("inertia_analysis.pdf")
        plt.legend(fontsize = 17)
        plt.show()


    def analyzeEscaping(self):
        all_data = pd.read_csv("../common_data/df_total_new.csv")
        all_data_nearest_ghost_dist = all_data[["distance1", "distance2"]].apply(
            lambda x: np.min([x.distance1, x.distance2]),
            axis=1
        )
        prehunt_nearest_ghost_dist =  self.data[["distance1", "distance2"]].apply(
                lambda x: np.min([x.distance1, x.distance2]),
                axis = 1
        )
        pre_hunt_data = self.data.assign(
            nearest_ghost_dist = prehunt_nearest_ghost_dist
        )
        # Compute the chi-square
        all_hist = np.histogram(all_data_nearest_ghost_dist, density=True)
        prehunt_hunt_hist = np.histogram(
            pre_hunt_data.nearest_ghost_dist[pre_hunt_data.after_energizer_mode == "hunt"],
            bins=all_hist[1],
            density=True)
        prehunt_graze_hist = np.histogram(
            pre_hunt_data.nearest_ghost_dist[pre_hunt_data.after_energizer_mode == "graze"],
            bins=all_hist[1],
            density=True)
        prehunt_hunt_chi2_stat = chisquare(
            prehunt_hunt_hist[0] / np.sum(prehunt_hunt_hist[0]),
            all_hist[0] / np.sum(all_hist[0])
        )
        prehunt_graze_chi2_stat = chisquare(
            prehunt_graze_hist[0] / np.sum(prehunt_graze_hist[0]),
            all_hist[0] / np.sum(all_hist[0])
        )
        # Plot the histogram of nearest ghost distance
        plt.figure(figsize = (25, 10))
        plt.subplot(1, 3, 1)
        plt.title("All Data", fontsize = 20)
        sns.distplot(all_data_nearest_ghost_dist, kde = True, color="#1BA3F9")
        plt.xlabel('Nearest Ghosts Distance', fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('Gaussian KDE', fontsize=20)
        plt.yticks(fontsize=20)
        plt.subplot(1, 3, 2)
        plt.title("Directly Hunt [$\\chi^2$ = {}]".format(str(prehunt_hunt_chi2_stat[1])[:5]), fontsize = 20)
        sns.distplot(pre_hunt_data.nearest_ghost_dist[pre_hunt_data.after_energizer_mode == "hunt"], kde=False, color="#1BA3F9")
        plt.fill_between([34, 40], 0, 250, color='#f26755', alpha=0.2, hatch='\\', edgecolor='w')
        plt.xlabel('Nearest Ghosts Distance', fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.yticks(fontsize=20)
        plt.subplot(1, 3, 3)
        plt.title("Graze Firstly [$\\chi^2$ = {}]".format(str(prehunt_graze_chi2_stat[1])[:5]), fontsize = 20)
        sns.distplot(pre_hunt_data.nearest_ghost_dist[pre_hunt_data.after_energizer_mode == "graze"], kde=False, color="#1BA3F9")
        plt.fill_between([34, 40], 0, 140, color='#f26755', alpha=0.2, hatch='\\', edgecolor='w')
        plt.xlabel('Nearest Ghosts Distance', fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.yticks(np.arange(0, 141, 20), np.arange(0, 141, 20), fontsize=20)
        plt.savefig("escaping-analysis.pdf")
        plt.show()


    def analyzeNoBeans(self):
        # The bean data
        bean_data = self.data[["file", "pacmanPos", "beans", "after_energizer_mode"]]
        bean_data = bean_data.assign(
            bean_distance=bean_data[["pacmanPos", "beans"]].apply(
                lambda x: [len(dijkstra_distance(x.pacmanPos, each)) for each in x.beans],
                axis=1
            )
        )
        bean_data = bean_data.assign(
            near_bean_num = bean_data[["bean_distance"]].apply(
                lambda x: np.sum(np.array(x.bean_distance) <= 5),
                axis=1
            )
        )
        # Plot near bean nums
        plt.figure(figsize=(15,10))
        plt.subplot(1, 2, 1)
        plt.title("Directly Hunt", fontsize = 20)
        sns.distplot(
            bean_data[bean_data.after_energizer_mode == "hunt"].near_bean_num,
            kde = False,
            color="#1BA3F9",
            bins = 8
        )
        plt.xlabel('Nearby Bean Num', fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.yticks(fontsize=20)

        plt.subplot(1, 2, 2)
        plt.title("Graze Firstly", fontsize=20)
        sns.distplot(
            bean_data[bean_data.after_energizer_mode == "graze"].near_bean_num,
            kde=False,
            color="#1BA3F9",
            bins=8
        )
        plt.xlabel('Nearby Bean Num', fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig("nearby_bean_analysis.pdf")
        plt.show()


    def _lookRatio(self, eye_region, energizer_region):
        count = 0
        for each in eye_region:
            if each == energizer_region:
                count += 1
        return count / len(eye_region)


    def decisionPoint(self):
        # Read data
        after_energizer_data = pd.read_csv("extracted_data/after_energizer_data.csv")
        for c in ["energizer_pos", "after_energizer_eye_pos", "after_energizer_eye_region"]:
            after_energizer_data[c] = after_energizer_data[c].apply(
                lambda x: eval(x) if not isinstance(x, float) else np.nan
            )
        before_prehunt_data = pd.read_csv("extracted_data/before_prehunt_data.csv")
        for c in ["energizer_pos", "start_prehunt_eye_pos", "start_prehunt_eye_region"]:
            before_prehunt_data[c] = before_prehunt_data[c].apply(
                lambda x: eval(x) if not isinstance(x, float) else np.nan
            )
        # After energizer look ratio
        after_look_ratio = after_energizer_data.apply(
            lambda x: self._lookRatio(x.after_energizer_eye_region, x.energizer_region),
            axis=1
        )
        # Before pre=hunt look ratio
        before_look_ratio = before_prehunt_data.apply(
            lambda x: self._lookRatio(x.start_prehunt_eye_region, x.energizer_region),
            axis = 1
        )
        # Integrate data
        all_look_ratio = after_energizer_data[['file']].assign(
            before_look_ratio = before_look_ratio,
            after_look_ratio = after_look_ratio
        )
        all_look_ratio = all_look_ratio[(all_look_ratio.before_look_ratio > 0) & (all_look_ratio.after_look_ratio > 0)]
        # Plot look ratio
        plt.scatter(np.arange(all_look_ratio.shape[0]), all_look_ratio.before_look_ratio, label = "before")
        plt.scatter(np.arange(all_look_ratio.shape[0]), all_look_ratio.after_look_ratio, label = "after")
        # plt.savefig("eye_look_at_analysis.pdf")
        plt.show()


if __name__ == '__main__':
    analyzer = PrehuntAnalyzer("extracted_data/all_prehunt_data.csv")
    analyzer.analyzeInertia()
    # analyzer.analyzeEscaping()
    # analyzer.analyzeNoBeans()
    # analyzer.decisionPoint()
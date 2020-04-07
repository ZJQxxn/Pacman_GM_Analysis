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


    def analyzeInertia(self):
        #TODO: due to the way we select pre-hunt data,  there shouldn't be cross on the path; but there should be corners
        # Build the path for a period of pre-hunt steps
        path_data = (
            self.data.groupby(["file"]).apply(
                lambda x: [each for each in x.pacmanPos]
            )
                .rename("move_path")
                .reset_index()
        )
        # The number of cross passed on the path from Pacman to the energizer
        #TODO: analyze whether change the direction
        is_pass_cross = path_data.move_path.apply(
                lambda x: [each in cross_pos for each in x]
        ).rename("is_pass_cross")
        cross_num = is_pass_cross.apply(
                lambda x: np.sum(x)
        )
        # Plot the histogram of cross number
        sns.distplot(cross_num, kde=True, color="#1BA3F9")
        plt.xlabel('Path Cross Num', fontsize=30)
        plt.xticks(fontsize=20)
        # plt.ylabel('Count', fontsize=30)
        # plt.yticks(fontsize=20)
        plt.yticks([], [])
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
        # Save prehunt data with minimal ghost distance larger than 34
        greedy_prehunt = self.data.apply(
            lambda x: np.min([x.distance1, x.distance2]) > 34, axis = 1
        )
        self.data[greedy_prehunt].to_csv("extracted_data/greedy_prehunt_data.csv")
        # Compute similarities between two histograms
        all_hist = np.histogram(all_data_nearest_ghost_dist, density=True)
        prehunt_hist = np.histogram(prehunt_nearest_ghost_dist, bins=all_hist[1], density=True)
        diff_norm = np.linalg.norm(
            prehunt_hist[0] / np.sum(prehunt_hist[0]) - all_hist[0] / np.sum(all_hist[0])
        )
        chi2_stat = chisquare(
            prehunt_hist[0] / np.sum(prehunt_hist[0]),
            all_hist[0] / np.sum(all_hist[0])
        )
        print("Histogram norm: ", diff_norm)
        print("Histogram chi-square p-value", chi2_stat[1])
        # Plot the histogram of combined distance for all the data
        plt.subplot(1, 2, 1)
        all_hist = sns.distplot(all_data_nearest_ghost_dist, kde = True, color="#1BA3F9")
        # plt.fill_between([34, 50], 0, 1, color='#f26755', alpha=0.2, hatch='\\', edgecolor='w')
        plt.xlabel('Nearest Ghosts Distance', fontsize=30)
        plt.xticks(fontsize=20)
        # plt.ylabel('Count', fontsize=30)
        # plt.yticks(fontsize=20)
        plt.yticks([],[])
        # Plot the histogram of combined distance for prehunt data
        plt.subplot(1, 2, 2)
        prehunt_hist = sns.distplot(prehunt_nearest_ghost_dist, kde=True, color="#1BA3F9")
        plt.fill_between([34, 40], 0, 0.08, color='#f26755', alpha=0.2, hatch='\\', edgecolor='w')
        plt.xlabel('Nearest Ghosts Distance', fontsize=30)
        plt.xticks(fontsize=20)
        # plt.ylabel('Count', fontsize=30)
        # plt.ylim((0, 0.08))
        # plt.yticks(np.arange(0, 0.081, 0.02), np.arange(0, 0.081, 0.02), fontsize=20)
        plt.yticks([],[])
        plt.show()


    def analyzeGreedyPrehunt(self):
        '''
        Analyze the data that the Pacman choose to eat energizer when ghosts are too far away.
        :return: 
        '''
        # TODO: useless for now
        # Read greedy pre-hunt data
        prehunt_data = pd.read_csv("extracted_data/greedy_prehunt_data.csv")[
            ["file", "index", "pacmanPos", "energizers", "beans", "ifscared1", "ifscared2"]
        ]
        for c in [
            "pacmanPos",
            "energizers",
            "beans"
        ]:
            prehunt_data[c] = prehunt_data[c].apply(lambda x: eval(x) if not isinstance(x, float) else np.nan)
        # Compute the distance between the Pacman and energizers and beans
        # Note: the Pacman locations in greedy pre-hunt data are not included in the "dij_distance_map.csv" file
        prehunt_data = prehunt_data.assign(
            energizer_distance = prehunt_data[["pacmanPos", "energizers"]].apply(
                lambda x: [len(dijkstra_distance(x.pacmanPos, each)) for each in x.energizers],
                axis = 1
            ),
            bean_distance = prehunt_data[["pacmanPos", "beans"]].apply(
                lambda x: [len(dijkstra_distance(x.pacmanPos, each)) for each in x.beans],
                axis = 1
            )
        )
        # Compute the nearest energizer distance and nearest bean distance
        prehunt_data = prehunt_data.assign(
            nearest_energizer_distance=prehunt_data[["energizer_distance"]].apply(
                lambda x: min(x.energizer_distance),
                axis=1
            ),
            nearest_bean_distance=prehunt_data[["bean_distance"]].apply(
                lambda x: min(x.bean_distance),
                axis=1
            )
        )
        # Plot the histogram of nearest energizer and nearest beans
        plt.subplot(1,2,1)
        sns.distplot(prehunt_data.nearest_energizer_distance, kde=False, color="#1BA3F9")
        plt.xlabel('Nearest Energizer Distance', fontsize=30)
        plt.xticks(fontsize=20)
        plt.ylabel('Count', fontsize=30)
        plt.yticks(fontsize=20)
        plt.subplot(1, 2, 2)
        sns.distplot(prehunt_data.nearest_bean_distance, kde=False, color="#1BA3F9")
        plt.xlabel('Nearest Bean Distance', fontsize=30)
        plt.xticks(fontsize=20)
        plt.ylabel('Count', fontsize=30)
        plt.yticks(fontsize=20)
        plt.subplots_adjust(hspace = 0.5)
        plt.show()


    def analyzeNoBeans(self):
        # for c in [
        #     "pacmanPos",
        #     "energizers",
        #     "beans"
        # ]:
        #     self.data[c] = self.data[c].apply(lambda x: eval(x) if not isinstance(x, float) else np.nan)
        # The bean data
        bean_data = self.data[["file", "index", "pacmanPos", "beans"]]
        bean_data = bean_data.assign(
            bean_distance=bean_data[["pacmanPos", "beans"]].apply(
                lambda x: [len(dijkstra_distance(x.pacmanPos, each)) for each in x.beans],
                axis=1
            )
        )
        print()
        bean_data = bean_data.assign(
            near_bean_num = bean_data[["bean_distance"]].apply(
                lambda x: np.sum(np.array(x.bean_distance) <= 5),
                axis=1
            )
        )
        # Plot near bean nums
        all_hist = sns.distplot(bean_data.near_bean_num, kde = False, color="#1BA3F9", bins = 8)
        # plt.fill_between([34, 50], 0, 1, color='#f26755', alpha=0.2, hatch='\\', edgecolor='w')
        plt.xlabel('Nearby Bean Num', fontsize=30)
        plt.xticks(fontsize=20)
        # plt.ylabel('Count', fontsize=30)
        plt.yticks(fontsize=20)
        plt.yticks([], [])
        # Plot the histogram of combined distance for prehunt data
        plt.show()


if __name__ == '__main__':
    analyzer = PrehuntAnalyzer("extracted_data/prehunt_data.csv")
    # analyzer.analyzeEscaping()
    # analyzer.analyzeGreedyPrehunt()
    # analyzer.analyzeNoBeans()
    analyzer.analyzeInertia()

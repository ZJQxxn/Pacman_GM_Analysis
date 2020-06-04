'''
Description:
   Analysis for the multi-agent.
   
Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    Jun. 2 2020
'''

import numpy as np
import pandas as pd
import pymc3 as pm
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB, CategoricalNB, GaussianNB



class MultiAgentAnalysis:

    def __init__(self, filename):
        '''
        Initialization.
        :param filename: Data filename. 
        '''
        # self.dir_list = ['left', 'right', 'up', 'down']
        # self.data = pd.read_csv(filename)
        # self.direction_vec = self.data['pacman_dir']
        self.agent_dir = pd.read_pickle("agent_dir.pkl")
        self.pacman_pos = self.agent_dir.pacman_pos
        self.global_dir = self.agent_dir.global_dir.apply(
            lambda x: np.argmax(eval(x)) if not isinstance(x, float) else -1
        )
        self.local_dir = self.agent_dir.local_dir.apply(
            lambda x: np.argmax(eval(x)) if not isinstance(x, float) else -1
        )
        self.lazy_dir = self.agent_dir.lazy_dir.apply(
            lambda x: np.argmax(eval(x)) if not isinstance(x, float) else -1
        )
        self.random_dir = self.agent_dir.random_dir.apply(
            lambda x: np.argmax(eval(x)) if not isinstance(x, float) else -1
        )
        self.integrate_dir = self.agent_dir.integrate_dir.apply(
            lambda x: np.argmax(eval(x)) if not isinstance(x, float) else -1
        )


    def _constructDataset(self):
        X = np.vstack(
            (self.global_dir.values, self.local_dir.values, self.lazy_dir.values, self.random_dir.values)
        ).T
        processed_X = np.zeros((X.shape[0], 2 * X.shape[1]))
        for index, each in enumerate(X):
            for i, agent_dir in enumerate(each):
                if agent_dir == 0:
                    processed_X[index][i * 2 : i * 2 + 2] = [-1, 0]
                elif agent_dir == 1:
                    processed_X[index][i * 2 : i * 2 + 2] = [1, 0]
                elif agent_dir == 2:
                    processed_X[index][i * 2: i * 2 + 2] = [0, 1]
                elif agent_dir == 3:
                    processed_X[index][i * 2: i * 2 + 2] = [0, -1]
                else:
                    raise ValueError("Undefined direction {}!".format(agent_dir))
        Y = self.integrate_dir.values
        return X, processed_X, Y


    def LogisticAnalysis(self):
        '''
        Extract features and use logistic regression for analysis.
        :return: 
        '''
        X, _, Y = self._constructDataset()
        # Logistic regression
        model = LogisticRegression(
            multi_class = 'ovr',
            solver = "lbfgs"
        )
        model.fit(X, Y)
        # The coefficient
        LR_coeff = model.coef_
        LR_coeff = np.mean(np.abs(LR_coeff), axis = 0)
        # LR_coeff = np.exp(LR_coeff) / np.sum(np.exp(LR_coeff))
        LR_coeff = LR_coeff / np.sum(LR_coeff)
        print("Logistic regression coefficient: ", LR_coeff)


    def LinearAnalysis(self):
        '''
        Extract features and use linear regression.
        :return: 
        '''
        # Construct dataset
        X, _, Y = self._constructDataset()
        # Logistic regression
        model = LinearRegression(fit_intercept=False)
        model.fit(X, Y)
        # The coefficient
        LR_coeff = model.coef_
        # LR_coeff = np.array([np.sum(np.abs(LR_coeff)[i:i + 2]) for i in [0, 2, 4, 6]])
        LR_coeff = LR_coeff / np.sum(abs(LR_coeff))
        print("Linear Regression Coefficient: ", LR_coeff)


    def BayesianAnalysis(self):
        # Direction for each agent
        global_agent = self.global_dir
        global_dir = np.argmax(global_agent)
        local_agent = self.local_dir
        local_dir = np.argmax(local_agent)
        lazy_agent = self.lazy_dir
        lazy_dir = np.argmax(lazy_agent)
        random_agent = self.random_dir
        random_dir = np.argmax(random_agent)
        # Integrated agent
        integrate_agent = self.integrate_dir
        integrate_dir = np.argmax(integrate_agent)
        # Bayesian estimation
        print("Bayesian....")
        with pm.Model() as model:
            beta = pm.Dirichlet("beta", a = np.array([0.25, 0.25, 0.25, 0.25]), shape = 4)
            pred_Y = beta[0] * global_dir + beta[1] * local_dir + beta[2] * lazy_dir + beta[3] * random_dir
            obs_Y = pm.Categorical("obs_Y", p = [1, 1, 1, 1], observed = integrate_dir)
            map_estimate = pm.find_MAP(model = model)
            print(map_estimate)
            # print(pm.fit())




if __name__ == '__main__':
    filename = "diary.csv"
    analysis = MultiAgentAnalysis(filename)
    #
    analysis.LogisticAnalysis()
    analysis.LinearAnalysis()
    # analysis.BayesianAnalysis()
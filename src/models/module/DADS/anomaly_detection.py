import gym
import numpy as np
import pandas as pd
import torch
import random
from gym import spaces
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.hbos import HBOS
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from .Utility_Functions import RSAMPLE

device = 'cuda'


class ad(gym.Env):
    def __init__(self, train_df, valid_df, black_len, white_len, parameter):
        dataset_a = torch.tensor(train_df.iloc[:black_len, :-1].values.astype(float)).float().to(device)
        dataset_n = torch.tensor(
            train_df.iloc[black_len:black_len + white_len, :-1].values.astype(float)).float().to(device)
        dataset_u = torch.tensor(train_df.iloc[black_len + white_len:, :-1].values.astype(float)).float().to(
            device)
        # print("Anomaly num: ", len(dataset_a), "   Unlabeled num: ", len(dataset_u), "Normal num: ", len(dataset_n))

        # dataset_anomaly, dataset_unlabeled, and dataset_temp are three dataset defined in the paper
        # dataset_test and test_label are used for test
        # tempdata_confidence stores the confidence of each data in dataset_temp
        self.dataset_anomaly = dataset_a
        self.dataset_unlabeled = dataset_u
        self.dataset_normal = dataset_n
        self.dataset_anomaly_backup = dataset_a
        self.dataset_unlabeled_backup = dataset_u
        self.dataset_temp = torch.tensor([]).to(device)
        self.valid_df = valid_df
        self.tempdata_confidence = []

        # initialize current data to be an unlabeled data
        self.current_index = random.randint(0, len(self.dataset_unlabeled) - 1)
        self.current_data = self.dataset_unlabeled[self.current_index]
        self.current_class = "unlabeled"

        self.observation_space = spaces.Discrete(self.current_data.size()[0])
        self.action_space = spaces.Discrete(2)
        self.tot_steps = 0

        self.sample_num = parameter["sample_num"]
        self.max_trajectory = parameter["max_trajectory"]
        self.check_num = parameter["check_num"]
        self.reward_list = parameter["reward_list"]
        self.strategy_distribution = parameter["strategy_distribution"]
        self.sampling_method_distribution = parameter["sampling_method_distribution"]

        self.previous_anomaly_num = len(self.dataset_anomaly)

        self.net = None

        # choose the following six methods as unsupervised anomaly datection method
        self.clf_list = [IForest(), HBOS(), OCSVM(), RSAMPLE()]

    def reset(self):
        self.dataset_anomaly = self.dataset_anomaly_backup
        self.dataset_unlabeled = self.dataset_unlabeled_backup
        self.dataset_temp = torch.tensor([]).to(device)
        self.tempdata_confidence = []

        self.current_index = random.randint(0, len(self.dataset_unlabeled) - 1)
        self.current_data = self.dataset_unlabeled[self.current_index]
        self.current_class = 'unlabeled'

        if len(self.dataset_unlabeled) > 5000:
            candidate_index = np.random.choice([i for i in range(len(self.dataset_unlabeled))], size=5000,
                                               replace=False)
            candidate = self.dataset_unlabeled[candidate_index]
        else:
            candidate = self.dataset_unlabeled
        for i in range(len(self.clf_list)):
            if self.sampling_method_distribution[i] > 0:
                clf = self.clf_list[i]
                clf.fit(candidate.cpu())

        if self.previous_anomaly_num > 1.5 * len(self.dataset_anomaly):
            self.check_num = self.check_num + 1
        elif self.previous_anomaly_num == len(self.dataset_anomaly):
            self.check_num = self.check_num - 1
        # print("check num: ", self.check_num)

        return self.current_data

    def unsupervised_index(self, type_index, data):
        """ return anomaly index of a data using one of six unsupervised method defined in __init__
        the returned value is normalized to range [0, 1]
        @param type_index: the index of unsupervised detecto, if type_index>=len(self.clf_list), random sampling method will be applied
        @param data: data needs to be calculated, both single data or multiple data is acceptable
        """
        clf = self.clf_list[type_index]
        if len(data.shape) == 1:
            r = clf.predict_proba(data.cpu().unsqueeze(0))[0][1]
        else:
            r = np.array(clf.predict_proba(data.cpu()))[:, 1]
        data.to(device)

        return r

    def calculate_reward(self, action):
        """ calculate reward based on the class of current data and the action"""
        if self.current_class == 'anomaly':
            if action == 1:
                score = self.reward_list[0]
            else:
                score = self.reward_list[1]
        elif self.current_class == 'normal':
            if action == 0:
                score = self.reward_list[2]
            else:
                score = self.reward_list[3]
        elif self.current_class == 'unlabeled':
            if action == 0:
                score = self.reward_list[4]
            else:
                score = 0
        elif self.current_class == 'temp':
            if action == 1 and self.tempdata_confidence[self.current_index] >= self.check_num:
                score = self.reward_list[5]
            else:
                score = 0
        else:
            assert 0

        return score

    def sample_method_one(self):
        """ random sampling is used in dataset_anomaly"""
        self.current_class = 'anomaly'
        self.current_index = random.randint(0, len(self.dataset_anomaly) - 1)
        self.current_data = self.dataset_anomaly[self.current_index]

    def sample_method_two(self):
        """ random sampling is used in dataset_anomaly"""
        self.current_class = 'normal'
        self.current_index = random.randint(0, len(self.dataset_anomaly) - 1)
        self.current_data = self.dataset_anomaly[self.current_index]

    def sample_method_three(self):
        """ random sampling is used in dataset_temp"""
        self.current_class = 'temp'
        self.current_index = random.randint(0, len(self.dataset_temp) - 1)
        self.current_data = self.dataset_temp[self.current_index]

    def sample_method_four(self):
        """ Use unsupervised-based method to sample data in dataset_unlabeled
        Each time sample a certain amount of data, then select the one with the highest unsupervised index
        """
        self.current_class = 'unlabeled'
        true_sample_num = min(self.sample_num, len(self.dataset_unlabeled))
        candidate_index = np.random.choice([i for i in range(len(self.dataset_unlabeled))], size=true_sample_num,
                                           replace=False)
        candidate = self.dataset_unlabeled[candidate_index]

        choice = np.random.choice([i for i in range(len(self.clf_list))], size=1,
                                  p=self.sampling_method_distribution)[0]
        score = self.unsupervised_index(choice, candidate)
        self.current_index = np.argmax(score)
        self.current_data = self.dataset_unlabeled[self.current_index]

    def refresh_dataset(self, action):
        """ Refresh three dataset according to the data flow rules"""
        if action == 1 and self.current_class == 'unlabeled':
            self.dataset_unlabeled = torch.cat(
                [self.dataset_unlabeled[:self.current_index], self.dataset_unlabeled[self.current_index+1:]])
            self.tempdata_confidence.append(1)
            self.dataset_temp = torch.cat([self.dataset_temp, self.current_data.unsqueeze(0)])
        elif action == 1 and self.current_class == 'temp':
            if self.tempdata_confidence[self.current_index] >= self.check_num:
                self.dataset_temp = torch.cat(
                    [self.dataset_temp[:self.current_index], self.dataset_temp[self.current_index + 1:]])
                self.tempdata_confidence = self.tempdata_confidence[:self.current_index] + \
                                           self.tempdata_confidence[self.current_index + 1:]
                self.dataset_anomaly = torch.cat([self.dataset_anomaly, self.current_data.unsqueeze(0)])
            else:
                self.tempdata_confidence[self.current_index] = self.tempdata_confidence[self.current_index] + 1
        elif action == 0 and self.current_class == 'temp':
            self.dataset_temp = torch.cat(
                [self.dataset_temp[:self.current_index], self.dataset_temp[self.current_index + 1:]])
            self.dataset_unlabeled = torch.cat([self.dataset_unlabeled, self.current_data.unsqueeze(0)])
            self.tempdata_confidence = self.tempdata_confidence[:self.current_index] + \
                                       self.tempdata_confidence[self.current_index + 1:]

    def step(self, action):
        """ Environment takes an action, then returns the current data(regarded as state), reward and done flag"""
        reward = self.calculate_reward(action)

        self.dataset_unlabeled.to(device)
        self.tot_steps = self.tot_steps + 1
        self.refresh_dataset(action)

        done = False
        if self.tot_steps % self.max_trajectory == 0:
            done = True
            self.previous_anomaly_num = len(self.dataset_anomaly)

        while True:   # sample next data according to the probablity distribution
            choice = np.random.choice([0, 1, 2, 3], size=1, p=self.strategy_distribution)[0]
            if choice == 0 and len(self.dataset_anomaly) != 0:
                self.sample_method_one()
                break
            if choice == 1 and len(self.dataset_normal) != 0:
                self.sample_method_two()
                break
            elif choice == 2 and len(self.dataset_temp) != 0:
                self.sample_method_three()
                break
            elif choice == 3 and len(self.dataset_unlabeled) != 0:
                self.sample_method_four()
                break
        return self.current_data, reward, done, " "

    def evaluate(self, df):
        """ Evaluate the agent, return AUC_ROC and AUC_PR"""
        x = torch.tensor(df.iloc[:, :-1].values.astype(float)).float().to(device)
        y = list(df.iloc[:, -1].values.astype(float))

        # q_values = self.net(x)
        # anomaly_score = q_values[:, 0]
        q_values = self.net(x)
        anomaly_score = q_values[:, 1]
        auc_roc = roc_auc_score(y, anomaly_score.cpu().detach())
        precision, recall, _thresholds = precision_recall_curve(y, anomaly_score.cpu().detach())
        auc_pr = auc(recall, precision)

        return auc_roc, auc_pr

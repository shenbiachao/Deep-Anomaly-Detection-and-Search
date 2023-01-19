import gym
import numpy as np
import torch
import random
from gym import spaces
from pyod.models.iforest import IForest
from pyod.models.ocsvm import OCSVM
from pyod.models.hbos import HBOS
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from .Utility_Functions import RSAMPLE, Logger
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import copy
import datetime


class ad(gym.Env):
    def __init__(self, train_df, valid_df, black_len, white_len, parameter):
        self.device = parameter["device"]
        self.dataset_anomaly = torch.tensor(train_df.iloc[:black_len, :-1].values.astype(float)).float().to(self.device)
        self.dataset_unlabeled = torch.tensor(train_df.iloc[black_len + white_len:, :-1].values.astype(float)).float().to(
            self.device)
        self.dataset = self.dataset_unlabeled
        self.confidence = [0] * len(self.dataset_unlabeled)
        self.anomaly_repeat_times = round(len(train_df) * parameter["anomaly_ratio"] / black_len)
        for _ in range(self.anomaly_repeat_times):
            self.dataset = torch.cat([self.dataset, self.dataset_anomaly])
            self.confidence = self.confidence + [parameter["check_num"]] * len(self.dataset_anomaly)
        self.valid_df = valid_df
        print("unlabeled: ", len(self.dataset_unlabeled), "abnormal: ", len(self.dataset_anomaly), "*", self.anomaly_repeat_times)

        self.current_index = random.randint(0, len(self.dataset) - 1)
        self.current_data = self.dataset[self.current_index]

        self.observation_space = spaces.Discrete(self.current_data.size()[0])
        self.action_space = spaces.Discrete(2)
        self.tot_steps = 0

        self.sample_num = parameter["sample_num"]
        self.max_trajectory = parameter["max_trajectory"]
        self.check_num = parameter["check_num"]
        self.reward_list = parameter["reward_list"]
        self.sampling_method_distribution = parameter["sampling_method_distribution"]
        self.score_threshold = parameter["score_threshold"]
        self.eval_interval = parameter["eval_interval"]

        self.searched_anomalies = 0

        self.net = None
        self.best_net = None
        self.best_roc = 0
        self.tot_reward = 0

        # choose the following six methods as unsupervised anomaly datection method
        self.clf_list = [IForest(), HBOS(), OCSVM(), RSAMPLE()]

        self.logger = Logger()
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logger.set_log(SummaryWriter(log_dir="./log/" + current_time))

    def reset(self):
        self.current_index = random.randint(0, len(self.dataset) - 1)
        self.current_data = self.dataset[self.current_index]

        if len(self.dataset_unlabeled) > 5000:
            candidate_index = np.random.choice([i for i in range(len(self.dataset_unlabeled))], size=5000,
                                               replace=False)
            candidate = self.dataset_unlabeled[candidate_index]
        else:
            candidate = self.dataset_unlabeled
        candidate = self.net.process_hidden_layers(candidate).detach().cpu().numpy()
        for i in range(len(self.clf_list)):
            if self.sampling_method_distribution[i] > 0:
                clf = self.clf_list[i]
                clf.fit(candidate)

        if self.searched_anomalies > 0.5 * len(self.dataset_anomaly):
            self.check_num = self.check_num + 1
        elif self.searched_anomalies == 0:
            self.check_num = self.check_num - 1
        print("check num: ", self.check_num)
        self.confidence = [0] * len(self.dataset_unlabeled)
        for _ in range(self.anomaly_repeat_times):
            self.confidence = self.confidence + [self.check_num] * len(self.dataset_anomaly)

        self.logger.base_idx = self.logger.base_idx + self.max_trajectory

        self.distance_matrix = np.zeros([len(self.dataset_anomaly), len(self.dataset)])
        mapped_anomaly = self.net.process_hidden_layers(self.dataset_anomaly).detach().numpy()
        mapped_dataset = self.net.process_hidden_layers(self.dataset).detach().numpy()
        for i in range(len(mapped_anomaly)):
            for j in range(len(mapped_dataset)):
                self.distance_matrix[i][j] = np.linalg.norm(mapped_anomaly[i] - mapped_dataset[j])

        return self.current_data

    def unsupervised_index(self, type_index, data):
        """ return anomaly index of a data using one of six unsupervised method defined in __init__
        the returned value is normalized to range [0, 1]
        @param type_index: the index of unsupervised detecto, if type_index>=len(self.clf_list), random sampling method will be applied
        @param data: data needs to be calculated, both single data or multiple data is acceptable
        """
        mapped_data = self.net.process_hidden_layers(data).detach().cpu().numpy()
        clf = self.clf_list[type_index]
        if len(data.shape) == 1:
            r = clf.predict_proba(mapped_data.reshape(1, -1))[0][1]
        else:
            r = np.array(clf.predict_proba(mapped_data))[:, 1]
        data.to(self.device)

        return r

    def calculate_reward(self, action):
        """ calculate reward based on the class of current data and the action"""
        if self.confidence[self.current_index] >= self.check_num:
            if action == 1:
                score = self.reward_list[0]
            else:
                score = self.reward_list[1]
        elif self.confidence[self.current_index] == self.check_num - 1 and action == 1:
            score = self.reward_list[2]
        else:
            score = 0

        return score

    def sampling_function(self, action):
        # candidate_index = np.random.choice([i for i in range(len(self.dataset))], size=self.sample_num,
        #                                    replace=False)
        # candidate = self.dataset[candidate_index]
        #
        # choice = np.random.choice([i for i in range(len(self.clf_list))], size=1,
        #                           p=self.sampling_method_distribution)[0]
        # score = self.unsupervised_index(choice, candidate)
        # max_index = np.argmax(score)
        # self.current_data = candidate[max_index]
        # self.current_index = candidate_index[max_index]

        candidate_index = np.random.choice([i for i in range(len(self.dataset))], size=self.sample_num,
                                           replace=False)
        distance = self.distance_matrix[:, candidate_index]
        self.current_index = candidate_index[np.argmin(np.min(distance, axis=1))]
        self.current_data = self.dataset[self.current_index]

    def refresh(self, action):
        if self.confidence[self.current_index] < self.check_num:
            if action == 1:
                self.confidence[self.current_index] += 1
            else:
                self.confidence[self.current_index] = 0

    def step(self, s):
        """ Environment takes an action, then returns the current data(regarded as state), reward and done flag"""
        if s > self.score_threshold:
            action = 1
        else:
            action = 0

        reward = self.calculate_reward(action)

        self.tot_steps = self.tot_steps + 1
        self.refresh(action)

        done = False
        if self.tot_steps % self.max_trajectory == 0:
            done = True
            self.searched_anomalies = sum(
                i >= self.check_num for i in self.confidence) - self.anomaly_repeat_times * len(self.dataset_anomaly)

            # roc, pr, p95 = self.evaluate(self.valid_df, False)
            # if roc > self.best_roc:
            #     self.best_net = copy.deepcopy(self.net)
            #     self.best_roc = roc

        self.sampling_function(action)

        if self.tot_steps % self.eval_interval == 0:
            auc_roc, auc_pr, p95 = self.evaluate(self.valid_df, False)
            if auc_roc > self.best_roc:
                self.best_net = copy.deepcopy(self.net)
                self.best_roc = auc_roc
            self.logger.log("result/auc_roc", auc_roc, self.tot_steps)
            self.logger.log("result/auc_pr", auc_pr, self.tot_steps)
            self.logger.log("result/p95", p95, self.tot_steps)

        return self.current_data, reward, done, " "

    def evaluate(self, df, flag):
        """ Evaluate the agent, return AUC_ROC and AUC_PR"""
        x = torch.tensor(df.iloc[:, :-1].values.astype(float)).float().to(self.device)
        y = list(df.iloc[:, -1].values.astype(float))

        if flag:
            q_values = self.best_net(x)
        else:
            q_values = self.net(x)
        anomaly_score = q_values[:, 0]

        auc_roc = roc_auc_score(y, anomaly_score.cpu().detach())
        precision, recall, _thresholds = precision_recall_curve(y, anomaly_score.cpu().detach())
        auc_pr = auc(recall, precision)
        fpr, tpr, thresholds = metrics.roc_curve(y, anomaly_score.cpu().detach(), pos_label=1)
        for idx, _tpr in enumerate(tpr):
            if _tpr > 0.95:
                break

        return auc_roc, auc_pr, fpr[idx]

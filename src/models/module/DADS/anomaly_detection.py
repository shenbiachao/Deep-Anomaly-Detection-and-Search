import gym
import numpy as np
import torch
import random
from gym import spaces
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.ecod import ECOD
from pyod.models.ocsvm import OCSVM
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score
from .Utility_Functions import RSAMPLE, Logger
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
import copy
import datetime
import matplotlib.pyplot as plt


class ad(gym.Env):
    def __init__(self, train_df, valid_df, black_len, white_len, contamination, dataset_name, ground_truth, parameter):
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
        self.ground_truth = ground_truth
        print("unlabeled: ", len(self.dataset_unlabeled), "abnormal: ", len(self.dataset_anomaly), "*", self.anomaly_repeat_times)

        self.current_index = random.randint(0, len(self.dataset) - 1)
        self.current_data = self.dataset[self.current_index]

        self.observation_space = spaces.Discrete(self.current_data.size()[0])
        self.action_space = spaces.Box(0, 1, shape=(1, ), dtype=np.float32)
        self.tot_steps = 0

        self.sample_num = parameter["sample_num"]
        self.max_trajectory = parameter["max_trajectory"]
        self.check_num = parameter["check_num"]
        self.initial_check_num = parameter["check_num"]
        self.reward_list = parameter["reward_list"]
        self.sampling_method_distribution = parameter["sampling_method_distribution"]
        self.score_threshold = parameter["score_threshold"]
        self.search_score_threshold = parameter["search_score_threshold"]
        self.eval_interval = parameter["eval_interval"]
        self.min_steps_before_searching = parameter["min_steps_before_searching"]

        # self.up_search_num = min(contamination, 0.04) * self.max_trajectory * parameter["up_search_percentage"]
        # self.low_search_num = min(contamination, 0.04) * self.max_trajectory * parameter["low_search_percentage"]
        self.up_search_num = min(contamination, 0.04) * self.max_trajectory * parameter["search_percentage"]
        self.low_search_num = self.up_search_num / 2
        print("Search range:", (self.low_search_num, self.up_search_num))

        self.searched_anomalies = 0
        self.correct_search_num = 0
        self.upper_search_num = 0

        self.net = None
        self.best_net = None
        self.best_roc = 0
        self.tot_reward = 0

        self.logger = Logger()
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.logger.set_log(SummaryWriter(log_dir="./log/" + current_time))

        unsup_method = {'annthyroid': ECOD(), 'cardio': IForest(), 'satimage2': ECOD(), 'satellite': ECOD(),
                        'thyroid': IForest(), 'arrhythmia': ECOD(), 'multi_shuttle': ECOD(), 'multi_cardio': OCSVM(),
                        'multi_har': IForest(), 'multi_annthyroid': OCSVM()}
        clf = unsup_method[dataset_name]
        clf.fit(self.dataset.cpu())
        self.anomaly_score_list = np.array(clf.decision_scores_.tolist())
        self.pre_sample_method = "random"
        print("Unsupervised method: ", clf.__class__)

        self.touch_num = [0] * (len(self.dataset_unlabeled) + len(self.dataset_anomaly))

    def reset(self):
        if self.tot_steps < self.min_steps_before_searching:
            self.check_num = 100
        else:
            if self.check_num == 100:
                self.check_num = self.initial_check_num
            else:
                if self.searched_anomalies > self.up_search_num:
                    self.check_num = self.check_num + 1
                elif self.searched_anomalies < self.low_search_num:
                    self.check_num = self.check_num - 1
        print("check num: ", self.check_num)

        self.confidence = [0] * len(self.dataset_unlabeled)
        for _ in range(self.anomaly_repeat_times):
            self.confidence = self.confidence + [self.check_num] * len(self.dataset_anomaly)

        self.logger.base_idx = self.logger.base_idx + self.max_trajectory

        self.current_index = random.randint(0, len(self.dataset) - 1)
        self.current_data = self.dataset[self.current_index]
        self.pre_sample_method = "random"

        self.touch_num = [0] * (len(self.dataset_unlabeled) + len(self.dataset_anomaly))

        return self.current_data

    def calculate_reward(self, action):
        """ calculate reward based on the class of current data and the action"""
        if self.confidence[self.current_index] >= self.check_num:
            if action == 1:
                score = self.reward_list[0]
            else:
                score = self.reward_list[1]
        elif self.confidence[self.current_index] == self.check_num - 1 and action == 1:
            score = self.reward_list[2]
        elif self.confidence[self.current_index] < self.check_num - 1:
            if self.pre_sample_method == "random" and action == 0:
                score = self.reward_list[3]
            elif self.pre_sample_method == "unsup" and action == 1:
                score = self.reward_list[4]
            else:
                score = 0
        else:
            score = 0

        return score

    def sampling_function(self, action):
        choice = np.random.choice([i for i in range(2)], size=1,
                                  p=self.sampling_method_distribution)[0]
        if choice == 0:
            self.pre_sample_method = "random"
            self.current_index = random.randint(0, len(self.dataset)-1)
            self.current_data = self.dataset[self.current_index]
        else:
            self.pre_sample_method = "unsup"
            candidate_index = np.random.choice([i for i in range(len(self.dataset))], size=self.sample_num,
                                               replace=False)
            score = self.anomaly_score_list[candidate_index]
            max_index = np.argmax(score)
            self.current_index = candidate_index[max_index]
            self.current_data = self.dataset[self.current_index]

        if self.current_index > len(self.dataset_unlabeled):
            touch_index = (self.current_index - len(self.dataset_unlabeled)) % len(self.dataset_anomaly)
        else:
            touch_index = self.current_index
        self.touch_num[touch_index] = self.touch_num[touch_index] + 1

    def refresh(self, score):
        if self.confidence[self.current_index] < self.check_num:
            if score > self.search_score_threshold:
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
        self.refresh(s)

        done = False
        if self.tot_steps % self.max_trajectory == 0:
            done = True
            self.searched_anomalies = sum(
                i >= self.check_num for i in self.confidence) - self.anomaly_repeat_times * len(self.dataset_anomaly)

            unlabeled_confidence = self.confidence[:len(self.dataset_unlabeled)]
            self.correct_search_num = sum(
                unlabeled_confidence[i] >= self.check_num and self.ground_truth[i] != 0 for i in
                range(len(unlabeled_confidence)))

            unlabeled_touch = self.touch_num[:len(self.dataset_unlabeled)]
            self.upper_search_num = sum(
                unlabeled_touch[i] >= self.check_num and self.ground_truth[i] != 0 for i in
                range(len(unlabeled_touch)))

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
        # plt.hist(anomaly_score.cpu().detach(), bins=10, color='blue')
        # plt.show()
        # plt.savefig("./figures/distribution"+str(int(self.tot_steps/self.max_trajectory))+".jpg")

        # plt.clf()
        # x_label = [0.1*i for i in range(10)]
        # for temp in x_label:
        #     y_hat = [0 if i < temp * 0.01 else 1 for i in anomaly_score.cpu().detach()]
        #     precision = np.mean([1 if y_hat[i] == y[i] else 0 for i in range(len(y))])
        #     plt.scatter(temp, precision, color='blue')
        # plt.show()
        # plt.savefig("./figures/acc_threshold"+str(int(self.tot_steps/self.max_trajectory))+".jpg")

        auc_roc = roc_auc_score(y, anomaly_score.cpu().detach())
        precision, recall, _thresholds = precision_recall_curve(y, anomaly_score.cpu().detach())
        auc_pr = auc(recall, precision)
        fpr, tpr, thresholds = metrics.roc_curve(y, anomaly_score.cpu().detach(), pos_label=1)
        for idx, _tpr in enumerate(tpr):
            if _tpr > 0.95:
                break

        return auc_roc, auc_pr, fpr[idx]

    def evaluate_unsup(self, clf):
        x = torch.tensor(self.valid_df.iloc[:, :-1].values.astype(float)).float().cpu()
        y = list(self.valid_df.iloc[:, -1].values.astype(float))

        y_hat = [i[1] for i in clf.predict_proba(x)]
        auc_roc = roc_auc_score(y, y_hat)

        return auc_roc

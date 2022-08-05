import pandas as pd
from src.models.module.dads.anomaly_detection import ad
from src.models.module.dads.SAC_Discrete import SAC_Discrete
from src.models.module.dads.SAC import SAC
from src.models.module.dads.DDPG import DDPG
from src.models.module.dads.Trainer import Trainer
import numpy as np


class DADS(object):
    def __init__(self, config):
        """Init DADS instance."""
        self.parameter = config

    def train(self, train_df, valid_df, test_df, black_len, white_len, finetune, former_episode, ori_df, dataset_name):
        # print("\n", self.parameter)
        # if finetune:
        #     result_list = np.array([0]*self.parameter["hyperparameters"]["num_episodes_to_run"])
        #     for i in range(1):
        #         # AGENT = SAC_Discrete
        #         AGENT = SAC
        #         self.environment = ad(train_df, valid_df, black_len, white_len, self.parameter["environment"])
        #         trainer = Trainer(self.parameter["hyperparameters"], AGENT, self.environment)
        #         temp = trainer.run_game_for_agent()
        #         result_list = result_list + np.array([i[0] + i[1] for i in temp])
        #     best_episode = np.argmax(result_list) + 1
        #     # print("best_episode: ", best_episode)
        #     self.parameter["hyperparameters"]["num_episodes_to_run"] = best_episode
        # else:
        #     self.parameter["hyperparameters"]["num_episodes_to_run"] = former_episode

        # AGENT = SAC_Discrete
        AGENT = SAC
        self.environment = ad(train_df, valid_df, test_df, black_len, white_len, self.parameter["environment"], ori_df, dataset_name)
        trainer = Trainer(self.parameter["hyperparameters"], AGENT, self.environment)
        trainer.run_game_for_agent()

        return self.parameter["hyperparameters"]["num_episodes_to_run"]

    def evaluate(self, test_df):
        auc_roc, auc_pr = self.environment.evaluate(test_df, True)

        return auc_roc, auc_pr

    def save_model(self, export_path):
        """Save SSAD model to export_path."""
        pass

    def load_model(self, import_path, device):
        """Load SSAD model from import_path."""
        pass

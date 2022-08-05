import copy
import numpy as np


class Trainer(object):
    """ Train the agent
    @property config: configuration of training
    @property agent: SAC agent to be trained
    @property environment: ad defined in anomaly_detection.py
    """
    def __init__(self, config, agent, environment):
        self.config = config
        self.agent = agent
        self.environment = environment

    def run_game_for_agent(self):
        """Run the training process several times to calculate the mean and variance of AUC_PR and AUC_ROC"""
        agent = self.agent(self.config, self.environment)
        if self.config["Actor"]["pretrain"] == 1:
            agent.pretrain()
        result_list = agent.run_n_episodes()  # run a single training process

        return result_list

    def print_two_empty_lines(self):
        print("-----------------------------------------------------------------------------------")
        print("-----------------------------------------------------------------------------------")
        print(" ")

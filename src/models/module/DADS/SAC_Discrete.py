import torch
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from .Base_Agent import Base_Agent
from .Utility_Functions import Replay_Buffer
from .Utility_Functions import create_actor_distribution
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from pyod.models.iforest import IForest


class SAC_Discrete(Base_Agent):
    """ SAC agent inherited from Base_Agent
    Include two critic networks and one actor network
    """
    agent_name = "SAC"

    def __init__(self, config, environment):
        Base_Agent.__init__(self, config, environment)
        self.hyperparameters = config
        assert self.hyperparameters["Actor"][
                   "final_layer_activation"] == "Softmax", "Final actor layer must be softmax"
        self.critic_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Critic")
        self.critic_local_2 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                             key_to_use="Critic")
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(),
                                                 lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_local_2.parameters(),
                                                   lr=self.hyperparameters["Critic"]["learning_rate"], eps=1e-4)
        self.critic_target = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                            key_to_use="Critic")
        self.critic_target_2 = self.create_NN(input_dim=self.state_size, output_dim=self.action_size,
                                              key_to_use="Critic")
        Base_Agent.copy_model_over(self.critic_local, self.critic_target)
        Base_Agent.copy_model_over(self.critic_local_2, self.critic_target_2)
        self.memory = Replay_Buffer(self.hyperparameters["Critic"]["buffer_size"],
                                    self.hyperparameters["batch_size"], device=self.hyperparameters["device"])

        self.actor_local = self.create_NN(input_dim=self.state_size, output_dim=self.action_size, key_to_use="Actor")
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(),
                                                lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)

        # We set the max possible entropy as the target entropy
        self.target_entropy = -np.log((1.0 / self.action_size)) * 0.98
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.hyperparameters["device"])
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=self.hyperparameters["Actor"]["learning_rate"], eps=1e-4)

        self.environment.net = self.actor_local

    def pretrain(self):
        clf = IForest()
        if len(self.environment.dataset_unlabeled) > 5000:
            candidate_index = np.random.choice([i for i in range(len(self.environment.dataset_unlabeled))], size=5000,
                                               replace=False)
            candidate = self.environment.dataset_unlabeled[candidate_index]
        else:
            candidate = self.environment.dataset_unlabeled
        clf.fit(candidate.cpu())
        y = torch.tensor([[i] for i in np.array(clf.predict_proba(self.environment.dataset_unlabeled.cpu()))[:, 1]])

        dataset = TensorDataset(self.environment.dataset_unlabeled, y)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        optimizer = Adam(self.actor_local.parameters())
        loss = torch.nn.MSELoss()
        for _, (data, score) in enumerate(loader):
            score_hat = self.actor_local(data)[:, 1]
            score = score.squeeze()
            l = loss(score.cpu(), score_hat.cpu())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()


    def produce_action_and_action_info(self, state):
        """
        @return action: selected action
        @return (action_probabilities, log_action_probabilities): distribution of action and log of itself
        @return max_probability_action: action with maxmimum probablity
        """
        action_probabilities = self.actor_local(state)
        max_probability_action = torch.argmax(action_probabilities, dim=-1)
        action_distribution = create_actor_distribution(self.action_types, action_probabilities, self.action_size)
        action = action_distribution.sample().cpu()
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        return action, (action_probabilities, log_action_probabilities), max_probability_action

    def reset_game(self):
        Base_Agent.reset_game(self)

    def step(self):
        """ Run an episode on the game, saving the experience and running a learning step if appropriate"""
        while not self.done:
            self.action = self.pick_action()
            self.conduct_action(self.action)
            if self.time_for_critic_and_actor_to_learn():
                for _ in range(self.hyperparameters["learning_updates_per_learning_session"]):
                    self.learn()
            self.save_experience(experience=(self.state, self.action, self.reward, self.next_state, self.done))
            self.state = self.next_state
            self.global_step_number += 1
        self.environment.net = self.actor_local
        self.episode_number += 1

    def pick_action(self, state=None):
        """ Pick an action using one of two methods:
        1) Randomly if we haven't passed a certain number of steps
        2) Using the actor
        """
        if state is None: state = self.state
        if self.global_step_number < self.hyperparameters["min_steps_before_learning"]:
            action = self.environment.action_space.sample()
        else:
            action = self.actor_pick_action(state=state)
        return action

    def actor_pick_action(self, state=None, eval=False):
        """ Use actor to pick an action in one of two ways:
        1) If eval = False and we aren't in eval mode then it picks
        an action that has partly been randomly sampled
        2) If eval = True then we pick the action that comes directly
        from the network and so did not involve any random sampling
        """
        if state is None: state = self.state
        if len(state.shape) == 1: state = state.unsqueeze(0)

        if eval == False:
            action, _, _ = self.produce_action_and_action_info(state)
        else:
            with torch.no_grad():
                _, z, action = self.produce_action_and_action_info(state)
        action = action.detach().cpu().numpy()
        return action[0]

    def time_for_critic_and_actor_to_learn(self):
        """ Return boolean indicating whether there are enough experiences to learn from, and
        it is time to learn for the actor and critic
        """
        return self.global_step_number > self.hyperparameters["min_steps_before_learning"] and \
               self.enough_experiences_to_learn_from() and self.global_step_number % self.hyperparameters[
                   "update_every_n_steps"] == 0

    def learn(self):
        """ Run a learning iteration for the actor, both critics and (if specified) the temperature parameter"""
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.sample_experiences()
        qf1_loss, qf2_loss = self.calculate_critic_losses(state_batch, action_batch, reward_batch, next_state_batch,
                                                          mask_batch)
        self.update_critic_parameters(qf1_loss, qf2_loss)

        policy_loss, log_pi = self.calculate_actor_loss(state_batch)
        alpha_loss = self.calculate_entropy_tuning_loss(log_pi)
        self.update_actor_parameters(policy_loss, alpha_loss)

    def sample_experiences(self):
        return self.memory.sample()

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch):
        """ Calculate the losses for the two critics.
        This is the ordinary Q-learning loss except the additional entropy term is taken into account
        """
        with torch.no_grad():
            next_state_action, (
            action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(next_state_batch)
            qf1_next_target = self.critic_target(next_state_batch)
            qf2_next_target = self.critic_target_2(next_state_batch)
            min_qf_next_target = action_probabilities * (
                        torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (
                min_qf_next_target)

        qf1 = self.critic_local(state_batch).gather(1, action_batch.long())
        qf2 = self.critic_local_2(state_batch).gather(1, action_batch.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss

    def calculate_actor_loss(self, state_batch):
        """Calculate the loss for the actor.
        This loss includes the additional entropy term
        """
        action, (action_probabilities, log_action_probabilities), _ = self.produce_action_and_action_info(state_batch)
        qf1_pi = self.critic_local(state_batch)
        qf2_pi = self.critic_local_2(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean()
        log_action_probabilities = torch.sum(log_action_probabilities * action_probabilities, dim=1)
        return policy_loss, log_action_probabilities

    def calculate_entropy_tuning_loss(self, log_pi):
        """ Calculate the loss for the entropy temperature parameter.
        This is only relevant if self.automatic_entropy_tuning is True
        """
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss

    def update_critic_parameters(self, critic_loss_1, critic_loss_2):
        """ Update the parameters for both critics"""
        self.take_optimisation_step(self.critic_optimizer, self.critic_local, critic_loss_1,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.take_optimisation_step(self.critic_optimizer_2, self.critic_local_2, critic_loss_2,
                                    self.hyperparameters["Critic"]["gradient_clipping_norm"])
        self.soft_update_of_target_network(self.critic_local, self.critic_target,
                                           self.hyperparameters["Critic"]["tau"])
        self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2,
                                           self.hyperparameters["Critic"]["tau"])

    def update_actor_parameters(self, actor_loss, alpha_loss):
        """Update the parameters for the actor and (if specified) the temperature parameter"""
        self.take_optimisation_step(self.actor_optimizer, self.actor_local, actor_loss,
                                    self.hyperparameters["Actor"]["gradient_clipping_norm"])
        if alpha_loss is not None:
            self.take_optimisation_step(self.alpha_optim, None, alpha_loss, None)
            self.alpha = self.log_alpha.exp()

    def eval(self):
        """ Evaluate the actor"""
        return self.environment.evaluate(self.environment.valid_df)

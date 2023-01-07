from src.models.module.dads.anomaly_detection import ad
from src.models.module.dads.SAC import SAC
from src.models.module.dads.Trainer import Trainer


class DADS(object):
    def __init__(self, config):
        """Init DADS instance."""
        self.parameter = config

    def train(self, train_df, valid_df, black_len, white_len):
        AGENT = SAC
        self.environment = ad(train_df, valid_df, black_len, white_len, self.parameter["Environment"])
        trainer = Trainer(self.parameter["Agent"], AGENT, self.environment)
        trainer.run_game_for_agent()

    def evaluate(self, test_df):
        auc_roc, auc_pr, p95 = self.environment.evaluate(test_df, True)

        return auc_roc, auc_pr, p95

    def save_model(self, export_path):
        """Save SSAD model to export_path."""
        pass

    def load_model(self, import_path, device):
        """Load SSAD model from import_path."""
        pass

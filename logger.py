class Logger:
    def __init__(self):
        pass

    def attach_optuna_trial(self, trial):
        self.trial = trial
    
    def log_training_results(self, engine):
        pass
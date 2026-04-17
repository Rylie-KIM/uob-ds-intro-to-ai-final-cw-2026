class EarlyStopping:
    # Adapted from https://medium.com/biased-algorithms/a-practical-guide-to-implementing-early-stopping-in-pytorch-for-model-training-99a7cbd46e9d
    def __init__(self, patience):
        self.patience = patience
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop = False

    def check(self, test_loss):
        if self.best_loss is None or test_loss < self.best_loss:
            self.best_loss = test_loss
            # have to reset the counter
            self.no_improvement_count = 0
        else:
            self.no_improvement_count +=1
            if self.no_improvement_count > self.patience:
                self.stop = True



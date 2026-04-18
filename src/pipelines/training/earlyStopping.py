class EarlyStopping:
    # Adapted from https://medium.com/biased-algorithms/a-practical-guide-to-implementing-early-stopping-in-pytorch-for-model-training-99a7cbd46e9d
    def __init__(self, patience : int, min_improvement : float = 0.0):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop = False

    def check(self, val_loss : float):
        if self.best_loss is None or val_loss < self.best_loss - self.min_improvement:
            self.best_loss = val_loss
            self.no_improvement_count = 0 # Reset counter
        else:
            self.no_improvement_count +=1
            if self.no_improvement_count > self.patience:
                self.stop = True



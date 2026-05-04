class EarlyStopping:
    # Adapted From:
    # Yadav, A. (2024). A Practical Guide to Implementing Early Stopping in PyTorch for Model Training. 
    # [online] Medium. Available at: https://medium.com/biased-algorithms/a-practical-guide-to-implementing-early-stopping-in-pytorch-for-model-training-99a7cbd46e9d.

    def __init__(self, patience : int, min_improvement : float = 0.0):
        self.patience = patience
        self.min_improvement = min_improvement
        self.lowest_loss = None
        self.no_decrease = 0
        self.stop = False

    def check(self, val_loss : float):
        if self.lowest_loss is None or val_loss < self.lowest_loss - self.min_improvement:
            self.lowest_loss = val_loss
            self.no_decrease = 0 
        else:
            self.no_decrease +=1
            if self.no_decrease > self.patience:
                self.stop = True



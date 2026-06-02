import torch
import numpy as np

class EarlyStopping:
    """
    Validation loss iyileşmediğinde eğitimi erken durduran yardımcı sınıf.
    """
    def __init__(self, patience=5, min_delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        """Validation loss düştüğünde modeli kaydeder."""
        torch.save(model.state_dict(), self.path)

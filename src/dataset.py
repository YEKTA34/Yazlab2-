import torch
from torch.utils.data import Dataset
import numpy as np

class TimeSeriesDataset(Dataset):
    """
    Zaman serisi verisini PyTorch modeline uygun (X, y) formatına çeviren Dataset sınıfı.
    X: [batch_size, window_size, num_features]
    y: [batch_size, 1]
    """
    def __init__(self, data_series, window_size=4):
        self.data_series = data_series
        self.window_size = window_size
        
        # Olası eksik veya hatalı veri boyutlarına karşı kontrol
        if len(self.data_series) <= self.window_size:
            raise ValueError("Veri boyutu pencere boyutundan büyük olmalıdır.")

    def __len__(self):
        return len(self.data_series) - self.window_size

    def __getitem__(self, idx):
        # Pencereyi al
        window = self.data_series[idx : idx + self.window_size]
        
        # Pencerenin hemen sonrasındaki değeri hedef (target) olarak belirle
        # Anomali tespiti için genellikle bir sonraki adımı tahmin etmeye (forecasting) çalışırız
        target = self.data_series[idx + self.window_size]
        
        # Tensor'a dönüştür
        x_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(-1) # [window_size, 1]
        y_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(-1) # [1]
        
        return x_tensor, y_tensor

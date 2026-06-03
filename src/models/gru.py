import torch
import torch.nn as nn

class GRUAnomalyDetector(nn.Module):
    """
    Zaman serisi üzerinde rekonstrüksiyon (forecasting) tabanlı anomali tespiti 
    yapmak için kullanılacak GRU mimarisi.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(GRUAnomalyDetector, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: [batch_size, window_size, input_size]
        
        # GRU layer
        gru_out, _ = self.gru(x)
        
        # Sadece son zaman adımının (t) çıktısını alarak bir sonraki adımı (t+1) tahmin edeceğiz
        last_time_step_out = gru_out[:, -1, :] # shape: [batch_size, hidden_size]
        
        # Fully connected layer
        prediction = self.fc(last_time_step_out) # shape: [batch_size, 1]
        
        return prediction

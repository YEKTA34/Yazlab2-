import torch
import numpy as np
import copy
from config import cfg
from dataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from .trainer import train_model
from models.lstm import LSTMAnomalyDetector
from models.gru import GRUAnomalyDetector
from .logger import ExperimentLogger
from utils.metrics import calculate_metrics

def add_gaussian_noise(data_series, mean=0.0, std=0.1):
    """Veriye Gauss gürültüsü ekler."""
    noise = np.random.normal(mean, std, size=data_series.shape)
    return data_series + noise

def run_dl_experiment(train_data, val_data, test_data, y_test, model_type="LSTM", scenario="Orijinal"):
    """Tek bir derin öğrenme modeli deneyini çalıştırır."""
    window_size = cfg["automata"]["window_size"]
    batch_size = cfg["deep_learning"]["batch_size"]
    epochs = cfg["deep_learning"]["epoch"]
    patience = cfg["deep_learning"]["early_stopping_patience"]
    
    # Dataset ve Dataloader
    train_dataset = TimeSeriesDataset(train_data, window_size)
    val_dataset = TimeSeriesDataset(val_data, window_size)
    test_dataset = TimeSeriesDataset(test_data, window_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model seçimi
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_type == "LSTM":
        model = LSTMAnomalyDetector(input_size=1)
    else:
        model = GRUAnomalyDetector(input_size=1)
        
    print(f"\n--- {model_type} Egitimi Basliyor ({scenario} Senaryosu) ---")
    
    # Modeli eğit
    trained_model = train_model(
        model, train_loader, val_loader, 
        epochs=epochs, patience=patience, device=device
    )
    
    # Test fazı
    trained_model.eval()
    test_predictions = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            preds = trained_model(X_batch)
            test_predictions.extend(preds.cpu().numpy().flatten())
            
    # Hata hesaplama ve anomali threshold (basit eşik)
    # Burada modelin tahmini ile gerçek değer arasındaki farka göre basit bir anomali kararı veriyoruz.
    actuals = test_data[window_size:]
    mse_errors = np.abs(test_predictions - actuals)
    threshold = np.mean(mse_errors) + 2 * np.std(mse_errors) # Basit dinamik eşik
    
    y_pred_binary = (mse_errors > threshold).astype(int)
    
    # Hedef (y_test) değerleri pencere boyutuna göre hizalanmalı
    y_true_aligned = y_test.iloc[window_size:].values
    
    # Metrikleri hesapla
    metrics = calculate_metrics(y_true_aligned, y_pred_binary)
    return metrics

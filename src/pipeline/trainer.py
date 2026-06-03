import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ..utils.early_stopping import EarlyStopping
import os

def train_model(model, train_loader, val_loader, epochs=50, patience=5, lr=0.001, device='cpu', checkpoint_path='checkpoint.pt'):
    """
    Standart PyTorch eğitim döngüsü (EarlyStopping destekli).
    """
    model = model.to(device)
    criterion = nn.MSELoss() # Rekonstrüksiyon için Hata Kareler Ortalaması
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    early_stopping = EarlyStopping(patience=patience, path=checkpoint_path)
    
    for epoch in range(epochs):
        # Eğitim fazı
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # Doğrulama fazı
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early stopping kontrolü
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Erken durdurma (Early Stopping) tetiklendi!")
            break
            
    # En iyi modeli geri yükle
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        
    return model

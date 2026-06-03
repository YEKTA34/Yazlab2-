import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    """Confusion matrix çizimi ve kaydetmesi."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_scores, title="ROC Eğrisi", save_path=None):
    """ROC eğrisi çizimi."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_transition_heatmap(transition_matrix, title="Transition Heatmap", save_path=None):
    """Otomata modeli için durum geçiş olasılıklarını görselleştirir."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(transition_matrix, annot=True, cmap='viridis')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

def calculate_metrics(y_true, y_pred):
    """
    Sınıflandırma (anomali tespiti) için temel metrikleri hesaplar.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1}

def perform_wilcoxon_test(metric_list_1, metric_list_2):
    """
    İki modelin (örn. LSTM ve Otomata) cross-validation (GroupKFold) 
    sonuçları (F1 skorları gibi) arasındaki anlamlılığı test eder.
    """
    stat, p_value = wilcoxon(metric_list_1, metric_list_2)
    return stat, p_value

def perform_mcnemar_test(y_true, y_pred_model1, y_pred_model2):
    """
    İki modelin aynı veri seti üzerindeki hata dağılımlarını karşılaştırır.
    b: Model 1'in yanlış, Model 2'nin doğru bildikleri
    c: Model 1'in doğru, Model 2'nin yanlış bildikleri
    """
    b = sum((y_pred_model1 != y_true) & (y_pred_model2 == y_true))
    c = sum((y_pred_model1 == y_true) & (y_pred_model2 != y_true))
    
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=True)
    return result.statistic, result.pvalue

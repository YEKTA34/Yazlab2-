import json
import os
from datetime import datetime

class ExperimentLogger:
    """
    Deney parametrelerini ve performans sonuçlarını (metrics) merkezi olarak kaydeder.
    """
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Benzersiz bir deney adı oluştur
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.log_dir, f"experiment_{self.experiment_id}.json")
        self.history = []

    def log_run(self, model_name, scenario, params, metrics):
        """
        Tek bir çalıştırmanın sonucunu günlüğe (history) ekler.
        """
        run_data = {
            "model": model_name,
            "scenario": scenario,
            "parameters": params,
            "metrics": metrics
        }
        self.history.append(run_data)
        self.save()

    def save(self):
        """Güncel durumu JSON dosyasına yazar."""
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4)
        
    def summary(self):
        """En iyi sonuçları konsola yazdırır."""
        print(f"Toplam {len(self.history)} deney loglandi: {self.log_file}")

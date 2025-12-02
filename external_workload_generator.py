import pandas as pd
import numpy as np
import os
import random

class ExternalWorkloadGenerator:
    def __init__(self, csv_path: str, load_column: str = 'total_cpu_demand', scale_factor: float = 2000.0):
        self.csv_path = csv_path
        self.load_column = load_column
        self.scale_factor = scale_factor
        
        self.workload_data = self._load_data()
        
        self.start_index = 0 

    def _load_data(self):
        if not os.path.exists(self.csv_path):
            return np.array([500.0] * 100, dtype=np.float32)
        try:
            df = pd.read_csv(self.csv_path)
            raw_data = df[self.load_column].values
            scaled_data = raw_data * self.scale_factor
            cleaned_data = np.maximum(scaled_data, 10.0)
            print(f"✅ Data chargée : {len(cleaned_data)} points.")
            return cleaned_data.astype(np.float32)
        except Exception:
            return np.array([500.0] * 100, dtype=np.float32)

    def get_workload(self, t):
        if len(self.workload_data) == 0: return 500.0

        real_index = (self.start_index + t) % len(self.workload_data)
        
        return self.workload_data[real_index]

    def reset(self):
    
        if len(self.workload_data) > 0:
            self.start_index = random.randint(0, len(self.workload_data) - 1)
        else:
            self.start_index = 0
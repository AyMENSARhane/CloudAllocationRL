import pandas as pd
import numpy as np
import os
import random

class ExternalWorkloadGenerator:
    """
    Workload Generator that reads from a CSV file.
    It supports two modes:
    1. Training Mode (start_index=None): Randomly jumps to any point in the file on reset.
    2. Test Mode (start_index=X): Always starts at the same point (deterministic) for fair comparison.
    """
    def __init__(self, csv_path: str, load_column: str = 'total_cpu_demand', scale_factor: float = 2000.0, 
                 start_index: int = None, end_index: int = None):
        
        self.csv_path = csv_path
        self.load_column = load_column
        self.scale_factor = scale_factor
        self.full_data = self._load_data()
        if start_index is None:
            self.workload_data = self.full_data
            self.random_mode = True
            print("Generator initialized in TRAINING MODE (Random start).")
        else:
            final_end = end_index if end_index is not None else len(self.full_data)
            self.workload_data = self.full_data[start_index:final_end]
            self.random_mode = False
            print(f"Generator initialized in TEST MODE (Fixed range: {start_index}-{final_end}).")

        self.current_start_index = 0 

    def _load_data(self):
        if not os.path.exists(self.csv_path):
            print(f"Error: File not found at {self.csv_path}")
            return np.array([500.0] * 100, dtype=np.float32)
        try:
            df = pd.read_csv(self.csv_path)
            if self.load_column not in df.columns:
                 return np.array([500.0] * 100, dtype=np.float32)
                 
            raw_data = df[self.load_column].values
            scaled_data = raw_data * self.scale_factor
            cleaned_data = np.maximum(scaled_data, 10.0)
            print(f" Data loaded: {len(cleaned_data)} points.")
            return cleaned_data.astype(np.float32)
        except Exception as e:
            print(f"CSV Error: {e}")
            return np.array([500.0] * 100, dtype=np.float32)

    def get_workload(self, t):
        if len(self.workload_data) == 0: return 500.0

        real_index = (self.current_start_index + t) % len(self.workload_data)
        
        return self.workload_data[real_index]

    def reset(self):
        """
        Resets the generator. 
        - In Training: Picks a new random start point.
        - In Testing: Resets to the beginning of the slice (0).
        """
        if len(self.workload_data) > 0:
            if self.random_mode:
                self.current_start_index = random.randint(0, len(self.workload_data) - 1)
            else:
                self.current_start_index = 0
        else:
            self.current_start_index = 0
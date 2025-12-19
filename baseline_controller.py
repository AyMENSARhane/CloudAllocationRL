import numpy as np

class ThresholdAutoScaler:
    """
    Standard Baseline Controller.
    Simple logic: Reactive scaling based on CPU thresholds.
    """
    def __init__(self, high_threshold=80, low_threshold=30):
        self.high = high_threshold
        self.low = low_threshold

    def predict(self, obs, deterministic=True):
        cpu_usage = obs[0]
        
        if cpu_usage > self.high:
            return np.array([1.0], dtype=np.float32), None 
        elif cpu_usage < self.low:
            return np.array([-1.0], dtype=np.float32), None 
        else:
            return np.array([0.0], dtype=np.float32), None
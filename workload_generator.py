import numpy as np

class WorkloadGenerator:
    """
    Simulates incoming HTTP traffic (requests per second - RPS).
    Generates a pattern with daily cycles (sine wave) and random noise.
    """
    def __init__(self, duration=1000, base_load=500, amplitude=300, noise_level=50):
        self.duration = duration
        self.time_step = 0
        self.base_load = base_load
        self.amplitude = amplitude
        self.noise_level = noise_level

    def get_workload(self, t):
        """
        Returns the number of requests at time step t.
        Formula: Base + Sine Wave (Trend) + Gaussian Noise (Randomness)
        """
        trend = self.amplitude * np.sin(2 * np.pi * t / 200) 
        noise = np.random.normal(0, self.noise_level)
        load = max(10, self.base_load + trend + noise)
        return load

    def reset(self):
        self.time_step = 0
        


work = WorkloadGenerator() 
print(work.get_workload(10))
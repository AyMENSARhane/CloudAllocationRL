import gymnasium as gym
import numpy as np
from gymnasium import spaces
from workload_generator import WorkloadGenerator

class CloudResourceEnv(gym.Env):
    """
    Custom Environment for Dynamic Resource Allocation with Queueing Model.
    The number of instances corresponds to the number of active CPU cores.
    
    Observation Space (s_t): [CPU%, Memory%, Latency, InstanceCount, PendingRequests]
    Action Space (a_t): Continuous value [-1, 1] for scaling.
    """

    def __init__(self , workload_gen = None):
        super(CloudResourceEnv, self).__init__()
        
        if workload_gen is None:
            # Si aucun générateur n'est fourni, crée un générateur par défaut
            self.workload_gen = WorkloadGenerator()
        else:
            # Utilise l'instance fournie (injection de dépendances)
            self.workload_gen = workload_gen

        # --- Configuration Parameters ---
        self.min_instances = 1
        self.max_instances = 25
        self.sla_latency_limit = 200  # ms (SLA Threshold)
        self.request_per_instance = 100 # RPS capacity of 1 CPU core

        # Reward weights (Alpha and Beta from the paper, Section 2)
        self.alpha = 1.5  # Higher penalty for SLA violation (strong incentive)
        self.beta = 0.5   # Cost importance
        self.base_latency = 50 # ms

        self.current_step = 0
        self.max_steps = 1000 

        # --- Internal State for Queueing ---
        self.pending_requests = 0.0 # Requests accumulated from past steps (the queue)
        self.current_instances = 5
        
        # --- Action Space ---
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # --- Observation Space ---
        # Vector: [CPU%, Mem%, Latency, InstanceCount, PendingRequests]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([100, 100, 5000, self.max_instances, 5000]), # Max 5000 pending requests
            dtype=np.float32
        )

        self.state = None

    def step(self, action):
        """
        Execute one time step within the environment, processing workload and updating state.
        """
        self.current_step += 1
        
        # 1. APPLY ACTION (Scaling)
        scaling_action = int(action * 5) # Scale up/down by up to 5 cores
        self.current_instances = np.clip(
            self.current_instances + scaling_action, 
            self.min_instances, 
            self.max_instances
        )

        # 2. CALCULATE WORKLOAD AND CAPACITY
        current_arrival = self.workload_gen.get_workload(self.current_step)
        
        # Total work is incoming requests PLUS previously pending requests (the queue)
        total_work_to_process = current_arrival + self.pending_requests
        total_capacity = self.current_instances * self.request_per_instance
        
        # 3. PROCESS WORK & UPDATE QUEUE
        
        # Work that the CPU cores successfully process in this step
        work_processed = min(total_work_to_process, total_capacity)
        
        # Work that remains in the queue for the next step
        self.pending_requests = max(0, total_work_to_process - total_capacity)

        # 4. SIMULATE SYSTEM METRICS
        
        # CPU Usage (Utilization is based on the actual work done vs total capacity)
        utilization_ratio = work_processed / total_capacity if total_capacity > 0 else 1.0
        cpu_usage = utilization_ratio * 100.0
        
        # Memory usage (Simplification: correlated to CPU in a Mono-CPU scenario)
        mem_usage = min(100.0, cpu_usage * 0.8 + np.random.normal(0, 5))
        
        # Latency Simulation (Now highly sensitive to utilization AND queue size)
        
        # Latency due to CPU saturation (as utilization approaches 100%)
        latency_cpu_saturation = self.base_latency / (1.001 - utilization_ratio)
        
        # Latency due to Queue backlog (time spent waiting in line)
        latency_queue_backlog = self.pending_requests * 0.1 # Simple model: 0.1ms per request in queue
        
        latency = latency_cpu_saturation + latency_queue_backlog
        latency = min(5000, latency) # Cap latency

        # Update State [CPU, Mem, Latency, Instances, PendingRequests]
        self.state = np.array([cpu_usage, mem_usage, latency, self.current_instances, self.pending_requests], dtype=np.float32)

        # 5. CALCULATE REWARD (r_t)
        
        # SLA Penalty: non-zero only if latency > threshold
        sla_penalty = max(0, latency - self.sla_latency_limit)
        
        # Cost: Linear cost per running instance (CPU core)
        resource_cost = self.current_instances
        
        # r = - (alpha * SLA_violations + beta * Resource_Cost)
        self.reward = - (self.alpha * sla_penalty + self.beta * resource_cost)

        # 6. CHECK TERMINATION
        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {
            "workload": current_arrival,
            "sla_violations": 1 if latency > self.sla_latency_limit else 0,
            "reward": self.reward 
        }

        return self.state, self.reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to an initial state for a new episode.
        """
        super().reset(seed=seed)
        self.workload_gen.reset()
        self.current_step = 0
        self.current_instances = 5 # Start with 5 CPU cores
        self.pending_requests = 0.0 # Queue must be empty at start
        
        # Initial state: [CPU, Mem, Latency, Instances, PendingRequests]
        self.state = np.array([50.0, 40.0, 60.0, 5.0, 0.0], dtype=np.float32)
        self.reward = 0.0
        
        return self.state, {}

    def render(self):
        """
        Renders the current state of the environment for debugging and visualization.
        """
        if self.state is None:
            return 
            
        cpu, mem, latency, instances, pending = self.state

        # Check for SLA violation
        sla_status = " SLA VIOLATION" if latency > self.sla_latency_limit else "✅ OK"
        
        print(f"\n--- TIME STEP {self.current_step} / {self.max_steps} ---")
        print(f" Reward: {self.reward:.2f}")
        print(f"Core Count: {instances:.0f} CPU(s) / Capacity: {instances * self.request_per_instance:.0f} RPS")
        print(f"==================================================")
        print(f"CPU Usage: {cpu:.1f}%")
        print(f"Latency: {latency:.1f} ms ({sla_status})")
        print(f"Pending Queue: {pending:.0f} requests")
        print("--------------------------------------------------")
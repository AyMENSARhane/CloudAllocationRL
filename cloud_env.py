import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import math
from typing import Optional 

# --- Environment Constants and Hyperparameters ---
# Base unit capacity for one abstract 'instance' or 'core'
CPU_PER_UNIT = 10.0   # 10 normalized CPU units per capacity unit
MEM_PER_UNIT = 8.0    # 8 normalized Memory units per capacity unit

# Minimum and Maximum capacity boundaries (to prevent infinite scaling)
MIN_CAPACITY = 2      # Minimum active units (20 CPU, 16 MEM)
MAX_CAPACITY = 50     # Maximum active units (500 CPU, 400 MEM)

# Hyperparameters for the Reward Function
ALPHA_SLA = 0.001    # High penalty for SLA violation (prioritize performance)
BETA_COST = 0.5       # Moderate penalty for resource cost (minimize waste)
GAMMA_STABILITY = 0.1 # Small penalty for aggressive scaling (encourage stability)

# Scaling Dynamics
MAX_SCALING_STEP = 5  # Max units to add/remove in one step (based on action)
SCALING_DELAY_STEPS = 2 # Scaling action takes effect after 2 steps (Sim2Real feature)


class CloudResourceEnv(gym.Env):
    """
    A custom Gymnasium environment for dynamic cloud resource allocation 
    using the Google Cluster trace as the external demand signal.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, workload_trace: pd.DataFrame, render_mode: Optional[str] = None):
        super(CloudResourceEnv, self).__init__()

        # --- Workload Trace Integration ---
        self.workload_trace = workload_trace
        self.total_steps = len(workload_trace)
        self.render_mode = render_mode

        # --- State and Action Space Definition ---
        
        # Action Space: Continuous value [-1.0, 1.0] for scaling factor
        # -1.0 means max scale-down, 1.0 means max scale-up
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation Space (State s_t): A vector of 6 metrics
        # [CPU_Demand, MEM_Demand, CPU_Utilization, MEM_Utilization, CPU_Capacity, MEM_Capacity]
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, 
            shape=(6,), 
            dtype=np.float32
        )

        # --- Internal Environment Variables ---
        self.current_step = 0
        self.current_units = MIN_CAPACITY # Number of active abstract units
        
        # Resource tracking (used to model scaling delay)
        self.units_provisioning = [] # Stores pending capacity changes
        
        # Internal state metrics
        self.cpu_capacity = self.current_units * CPU_PER_UNIT
        self.mem_capacity = self.current_units * MEM_PER_UNIT

        # History for debugging/rendering
        self.history = []

    def _get_obs(self):
        """Helper to construct the current observation vector."""
        if self.current_step >= self.total_steps:
             # If simulation is over, return padded zeros
             return np.zeros(6, dtype=np.float32)

        # Get demand for the current step from the workload trace
        data = self.workload_trace.iloc[self.current_step]
        
        demand_cpu = data['total_cpu_demand']
        demand_mem = data['total_mem_demand']
        
        # Calculate utilization (usage is capped at 1.0, or 100%)
        util_cpu = np.clip(demand_cpu / self.cpu_capacity, 0.0, 1.0) if self.cpu_capacity > 0 else 1.0
        util_mem = np.clip(demand_mem / self.mem_capacity, 0.0, 1.0) if self.mem_capacity > 0 else 1.0

        return np.array([
            demand_cpu, 
            demand_mem, 
            util_cpu, 
            util_mem, 
            self.cpu_capacity, 
            self.mem_capacity
        ], dtype=np.float32)

    def _apply_action(self, action):
        """
        Processes the agent's action and schedules the capacity change.
        Action is a scaling factor [-1.0, 1.0].
        """
        # Determine the target change in units based on the continuous action
        unit_change = int(action[0] * MAX_SCALING_STEP)
        
        target_units = np.clip(
            self.current_units + unit_change, 
            MIN_CAPACITY, 
            MAX_CAPACITY
        )
        
        # Calculate the actual change to provision
        actual_change = target_units - self.current_units
        
        if actual_change != 0:
            # Schedule the change to occur after the delay
            provision_step = self.current_step + SCALING_DELAY_STEPS
            self.units_provisioning.append((provision_step, actual_change))
            
        # Penalize aggressive scaling regardless of outcome
        stability_penalty = GAMMA_STABILITY * abs(unit_change)
        return stability_penalty

    def _apply_provisioning(self):
        """Applies pending capacity changes that have passed their delay time."""
        
        new_provisioning = []
        capacity_change = 0
        
        for step, change in self.units_provisioning:
            if self.current_step >= step:
                capacity_change += change
            else:
                new_provisioning.append((step, change))
        
        self.units_provisioning = new_provisioning
        
        # Apply the change and clip to ensure boundaries are respected
        self.current_units = np.clip(
            self.current_units + capacity_change, 
            MIN_CAPACITY, 
            MAX_CAPACITY
        )
        
        # Update physical capacities
        self.cpu_capacity = self.current_units * CPU_PER_UNIT
        self.mem_capacity = self.current_units * MEM_PER_UNIT

    def _calculate_reward(self, demand_cpu, demand_mem):
        """Calculates the reward based on cost and SLA penalties."""
        
        # 1. Resource Cost Penalty
        cost_penalty = BETA_COST * self.current_units
        
        # 2. SLA Violation Penalty
        # Violation occurs if demand exceeds capacity for either CPU or Memory
        
        # CPU Violation: max(0, Demand - Capacity)
        cpu_violation = max(0, demand_cpu - self.cpu_capacity)
        
        # Memory Violation: max(0, Demand - Capacity)
        mem_violation = max(0, demand_mem - self.mem_capacity)
        
        # The penalty is based on the worse violation (the bottleneck resource)
        # We use the square to harshly penalize large shortfalls.
        sla_penalty = ALPHA_SLA * (cpu_violation**2 + mem_violation**2)
        
        # Total Reward: Goal is to maximize (i.e., minimize negative penalties)
        total_penalty = cost_penalty + sla_penalty
        reward = -total_penalty
        
        return reward, cost_penalty, sla_penalty

    # --- Core Gym Methods ---

    def step(self, action):
        """
        The agent performs an action, the environment transitions to a new state.
        """
        # 1. Check if the episode is finished
        if self.current_step >= self.total_steps - 1:
            return self._get_obs(), 0.0, True, False, {}

        # 2. Apply Agent's Action and get scaling penalty
        stability_penalty = self._apply_action(action)
        
        # 3. Advance to the next time step
        self.current_step += 1
        
        # 4. Apply capacity changes that are due (handling the scaling delay)
        self._apply_provisioning()

        # 5. Get NEW State and Demand at t+1
        obs = self._get_obs()
        
        # Extract the demand data for the reward calculation
        data = self.workload_trace.iloc[self.current_step - 1] # Use data from the step just completed
        demand_cpu = data['total_cpu_demand']
        demand_mem = data['total_mem_demand']
        
        # 6. Calculate Reward
        reward, cost, sla_pen = self._calculate_reward(demand_cpu, demand_mem)
        reward -= stability_penalty # Apply stability penalty here

        # 7. Check for termination
        # Terminate if the agent fails to provision resources for too long (Optional)
        terminated = False 
        truncated = False
        
        info = {
            "cost_penalty": cost,
            "sla_penalty": sla_pen,
            "total_demand_cpu": demand_cpu,
            "current_units": self.current_units
        }

        # Store history for visualization in Dash later
        self.history.append({
            "cost_penalty": cost,
            "sla_penalty": sla_pen,
            "total_demand_cpu": demand_cpu,
            "current_units": self.current_units,
            "step": self.current_step, 
            "reward": reward # <--- This key MUST be present
        })
        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Resets the environment to the starting state for a new episode.
        """
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_units = MIN_CAPACITY
        self.cpu_capacity = self.current_units * CPU_PER_UNIT
        self.mem_capacity = self.current_units * MEM_PER_UNIT
        self.units_provisioning = []
        self.history = []

        observation = self._get_obs()
        info = {}
        return observation, info

    # Optional: Render mode function (can be implemented later for Dash visualization)
    def render(self):
        if self.render_mode == 'human':
            # This is where you would print or plot the current state
            pass

    def close(self):
        # Clean up any resources (e.g., plot windows)
        pass
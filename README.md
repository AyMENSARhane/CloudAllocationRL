
#  Reinforcement Learning for Dynamic Cloud Resource Allocation

## Project Overview

This project implements a dynamic autoscaling solution for cloud environments using **Reinforcement Learning (RL)**, specifically the **Proximal Policy Optimization (PPO)** algorithm. The goal is to train an intelligent agent to balance two competing objectives:
1.  **Maintain Service Level Agreement (SLA) Compliance** (minimize latency and pending requests).
2.  **Minimize Operational Costs** (optimize the number of active CPU cores/instances).

The project framework relies on the **Gymnasium** standard for simulating the cloud environment as a Markov Decision Process (MDP).

---

##  Architecture and Core Components

The solution is divided into three logical components:

1.  **Workload Generator (`workload_generator.py`)**: Simulates incoming user traffic (Requests Per Second, RPS). It supports sinusoidal patterns (for typical daily load cycles) and configurable **stress events** for testing resilience.
2.  **Cloud Environment (`cloud_env.py`)**: The core simulation, implementing the MDP. It uses a **Queueing Model** to track pending requests and accurately calculates Latency and CPU usage based on the allocated cores.
3.  **RL Agent (`train_agent.py` / `test_model.py`)**: Uses the **Stable-Baselines3** library to implement and train the PPO (or SAC) agent on the observed state and calculated reward.

### MDP Formulation (CloudResourceEnv)

| Component | Description |
| :--- | :--- |
| **State ($s_t$)** | `[CPU%, Memory%, Latency (ms), Instance Count, Pending Requests]`  |
| **Action ($a_t$)** | Continuous value $[-1.0, 1.0]$ where -1.0 is max scale-down and 1.0 is max scale-up. |
| **Reward ($r_t$)** | $r_{t}=-(\alpha \times \text{SLA Violations} + \beta \times \text{Resource Cost})$  |

---

##  Setup and Installation

### Prerequisites

You need Python 3.9+ installed.

### Installation Steps

1.  **Clone the Repository (if applicable) or ensure files are structured:**

    ```bash
    /your_project_directory
    ├── cloud_env.py
    ├── workload_generator.py
    ├── train_agent.py
    └── test_model.py
    ```

2.  **Install Dependencies:**
    The project relies on standard scientific and RL libraries.

    ```bash
    pip install gymnasium stable-baselines3 numpy pandas matplotlib tensorboard
    ```

---

##  Execution Guide

### 1. Training the RL Agent

The `train_agent.py` script executes the PPO training loop. It saves the trained model and logs every session automatically into a time-stamped directory.

```bash
python train_agent.py
````

  * **Output Location**: Logs and models are saved in `./training_sessions/YYYY-MM-DD_HhMmSs/`.

  * **Monitoring**: To view the training progress (Reward Curve, Value Loss, etc.), run TensorBoard:

    ```bash
    tensorboard --logdir=training_sessions
    ```

### 2\. Testing the Trained Model

The `test_model.py` script loads a previously trained model and runs a short simulation to visualize its final policy (scaling decisions).

#### A. Running the Default Test

The script is configured to use a default model path (which you should update after your first successful training run).

```bash
python test_model.py
```

#### B. Running a Specific Model

To test a specific model version (e.g., from a session with optimal hyperparameter tuning), pass the path as an argument:

```bash
python test_model.py training_sessions/2025-11-26_16h00m15s/models/final_ppo_autoscaler_model.zip
```

### 3\. Running Stress Tests

To evaluate the agent's resilience, you can introduce custom peaks without changing the core RL logic:

1.  **Modify `train_agent.py` or `test_model.py`**: Find the lines where `CloudResourceEnv` is initialized.
2.  **Configure Stress**: Create a custom `WorkloadGenerator` instance with a defined stress peak and inject it into the environment.

<!-- end list -->

```python
# Example in your test script:
from workload_generator import WorkloadGenerator

# 1. Define custom extreme scenario
custom_workload = WorkloadGenerator(base_load=300, amplitude=500)
custom_workload.add_stress_peak(start_step=150, duration=10, magnitude=1200)

# 2. Inject into environment
test_env = CloudResourceEnv(workload_gen=custom_workload) 
```

-----

##  Evaluation

Key metrics to analyze in your results:


  * **SLA Violation Rate**: Must be minimized by the RL agent.
  * **Cost Efficiency**: Must be improved compared to static or threshold-based baselines.
  * **Scaling Stability**: Check the frequency and magnitude of scaling actions (smooth transitions are better).

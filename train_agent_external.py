import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from datetime import datetime
from cloud_env import CloudResourceEnv
from external_workload_generator import ExternalWorkloadGenerator

# 1. Configurer le générateur avec VOTRE fichier
# scale_factor=2000 transforme 0.5 (du CSV) en 1000 RPS (pour vos serveurs)



# --- 1. DEFINING A UNIQUE PATH PER SESSION ---
# Creates a timestamp (e.g., 2025-11-26_17h07m18s)
session_id = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

# Parent directory for all training sessions
BASE_LOG_DIR = "training_sessions"

# Specific path for this session's files
SESSION_PATH = os.path.join(BASE_LOG_DIR, session_id)
LOG_DIR = os.path.join(SESSION_PATH, "logs")
MODELS_DIR = os.path.join(SESSION_PATH, "models")

# Create the directory structure
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- 2. ENVIRONMENT AND AGENT INITIALIZATION ---

# 2.1 Initialize the base environment (non-vectorized)
# Creates a list of functions that return the environment instance
def make_env():
    return CloudResourceEnv()

# 2.2 Vectorize the environment and use VecMonitor (best practice for logging VecEnvs)
# DummyVecEnv wraps the single environment for compatibility with SB3 algorithms
env = DummyVecEnv([make_env]) 
env = VecMonitor(env, LOG_DIR)

# 2.3 Define the RL Agent (PPO)
model = SAC(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.0003, 
    gamma=0.99, 
    tensorboard_log=LOG_DIR,
)


# --- 3. TRAINING AND SAVING ---

print(f"Starting training process for session: {session_id}")
print(f"Logs and model will be saved in: {SESSION_PATH}")

model.learn(total_timesteps=50000)

# 3.1 Save the trained model to the session directory
model_path = os.path.join(MODELS_DIR, "final_ppo_autoscaler_model")
model.save(model_path)
print(f"\nModel saved successfully in: {model_path}.zip")

real_workload = ExternalWorkloadGenerator(
    csv_path="train_data.csv", 
    load_column="total_cpu_demand", 
    scale_factor=2000.0
)

env = CloudResourceEnv(workload_gen=real_workload)

test_env = CloudResourceEnv()
obs, _ = test_env.reset()

print("\n--- Testing the trained agent (20 steps) ---")
for i in range(20):
    # 1. The model predicts the action. 'action' is an array (e.g., [[0.75]])
    # The observation must be provided to the vectorized model.
    action_array, _states = model.predict(obs, deterministic=True)
    
    # 2. Robustly extract the scalar value (the float between -1.0 and 1.0) from the action array.
    # This ensures compatibility and fixes the previous IndexError.
    action_value = action_array.flatten()[0] 
    
    # 3. The simple test environment uses this scalar value (action_value).
    obs, reward, terminated, truncated, info = test_env.step(action_value)
    
    test_env.render()
    if terminated or truncated:
        break

print(f"\nTraining session {session_id} complete.")
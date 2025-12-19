import os
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from datetime import datetime
from cloud_env import CloudResourceEnv

# --- 1. DEFINING A UNIQUE PATH PER SESSION ---
session_id = datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")
BASE_LOG_DIR = "training_sessions"
SESSION_PATH = os.path.join(BASE_LOG_DIR, session_id)
LOG_DIR = os.path.join(SESSION_PATH, "logs")
MODELS_DIR = os.path.join(SESSION_PATH, "models")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- 2. ENVIRONMENT AND AGENT INITIALIZATION ---
def make_env():
    return CloudResourceEnv()

env = DummyVecEnv([make_env]) 
env = VecMonitor(env, LOG_DIR)
model = PPO(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=0.0003,
    gamma=0.99 , 
    tensorboard_log=LOG_DIR, 
)

# --- 3. TRAINING AND SAVING ---

print(f"Starting training process for session: {session_id}")
print(f"Logs and model will be saved in: {SESSION_PATH}")

model.learn(total_timesteps=50000)
model_path = os.path.join(MODELS_DIR, "final_ppo_autoscaler_model")
model.save(model_path)
print(f"\nModel saved successfully in: {model_path}.zip")

# --- 4. TEST AND RENDER ---

test_env = CloudResourceEnv()
obs, _ = test_env.reset()

print("\n--- Testing the trained agent (20 steps) ---")
for i in range(20):
    action_array, _states = model.predict(obs, deterministic=True)
    action_value = action_array.flatten()[0] 
    obs, reward, terminated, truncated, info = test_env.step(action_value)
    
    test_env.render()
    if terminated or truncated:
        break

print(f"\nTraining session {session_id} complete.")
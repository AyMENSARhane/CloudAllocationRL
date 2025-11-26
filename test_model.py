import os
import argparse
from stable_baselines3 import PPO
from cloud_env import CloudResourceEnv
import numpy as np
from workload_generator import WorkloadGenerator

# --- Default path 
DEFAULT_MODEL_PATH = "training_sessions/2025-11-26_16h08m04s/models/final_ppo_autoscaler_model.zip"

# ----------------------------------------------------------------------
# 1. TEST EXECUTION FUNCTION
# ----------------------------------------------------------------------

def run_test_simulation(model_path: str, num_steps: int = 20):
    """
    Loads a trained PPO model and runs a short test simulation on the environment.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure the default path or the specified path is correct.")
        return

    # 1. Load the model
    try:
        model = PPO.load(model_path, device="auto", verbose=0)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Check dependencies and file integrity.")
        return

    # 2. Initialize the environment for testing
    Workload = WorkloadGenerator(amplitude= 1500)
    test_env = CloudResourceEnv(Workload)
    obs, _ = test_env.reset()

    print(f"\n--- Testing Trained Agent ({num_steps} steps) ---")
    print(f"Model loaded: {os.path.basename(model_path)}")
    print("--------------------------------------------------")

    # 3. Simulation Loop
    for i in range(num_steps):
        # The model predicts the action
        action_array, _states = model.predict(obs, deterministic=True)
        
        # Extract the scalar action value
        action_value = action_array.flatten()[0] 
        
        # Step the environment
        obs, reward, terminated, truncated, info = test_env.step(action_value)
        
        test_env.render()
        
        if terminated or truncated:
            print("--- Episode Terminated ---")
            break
            
    print("\nTest simulation complete.")

# ----------------------------------------------------------------------
# 2. MAIN EXECUTION BLOCK (Argument Parsing)
# ----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained PPO model using a specified path.")
    
    parser.add_argument(
        'model_path', 
        type=str, 
        nargs='?', 
        default=DEFAULT_MODEL_PATH,
        help="Full path to the PPO model .zip file. Defaults to a generic path if not specified."
    )
    args = parser.parse_args()

    run_test_simulation(args.model_path)
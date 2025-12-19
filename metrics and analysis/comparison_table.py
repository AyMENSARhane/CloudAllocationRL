import os
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC
from cloud_env import CloudResourceEnv
from external_workload_generator import ExternalWorkloadGenerator
from baseline_controller import ThresholdAutoScaler
import pandas as pd 

# --- CONFIGURATION ---
CSV_FILE_PATH =  r"data\test_data.csv" 
CSV_COLUMN_NAME = "total_cpu_demand"
SCALE_FACTOR = 2000.0 

# Paths to trained models
PPO_MODEL_PATH = "training_sessions/2025-12-02_16h54m02s/models/final_ppo_autoscaler_model.zip"
SAC_MODEL_PATH = "training_sessions/2025-12-02_17h01m35s/models/final_sac_autoscaler_model.zip" 

TEST_DURATION = 1500

def calculate_aggregate_metrics(history, sla_threshold=200):
    if not history or len(history["latency"]) == 0:
        return None
    latencies = np.array(history["latency"])
    instances = np.array(history["instances"])
    queues = np.array(history["queue"])
    violation_count = np.sum(latencies > sla_threshold)
    total_steps = len(latencies)
    sla_violation_rate = (violation_count / total_steps) * 100.0
    avg_latency = np.mean(latencies)
    avg_instances = np.mean(instances)
    avg_queue = np.mean(queues)
    stability_std = np.std(instances)
    return {
        "SLA Violation (%)": sla_violation_rate,
        "Avg Latency (ms)": avg_latency,
        "Avg Cost (Instances)": avg_instances,
        "Avg Queue Size": avg_queue,
        "Stability (Std Dev)": stability_std
    }


def print_comparison_table(res_ppo, res_sac, res_base):
    metrics = []
    if res_base:
        m = calculate_aggregate_metrics(res_base)
        m["Agent"] = "Baseline (Threshold)"
        metrics.append(m)
        
    if res_ppo:
        m = calculate_aggregate_metrics(res_ppo)
        m["Agent"] = "PPO Agent"
        metrics.append(m)
        
    if res_sac:
        m = calculate_aggregate_metrics(res_sac)
        m["Agent"] = "SAC Agent"
        metrics.append(m)
    df = pd.DataFrame(metrics)
    cols = ["Agent", "SLA Violation (%)", "Avg Latency (ms)", "Avg Cost (Instances)", "Avg Queue Size", "Stability (Std Dev)"]
    df = df[cols]
    
    print("\n" + "="*80)
    print("FINAL COMPARATIVE RESULTS (Global Metrics)")
    print("="*80)
    print(df.to_string(index=False, float_format="%.2f"))
    print("="*80)
    df.to_csv("final_metrics_summary.csv", index=False, float_format="%.2f")
    print("\n✅ Métriques sauvegardées dans 'final_metrics_summary.csv'")
    
def run_simulation(agent, workload_source, label):
    """
    Runs a single simulation episode with a specific agent and workload.
    """
    print(f"--- Running Simulation: {label} ---")
    workload_source.reset() 
    env = CloudResourceEnv(workload_gen=workload_source)
    obs, _ = env.reset()
    history = {
        "step": [], "latency": [], "cpu": [], 
        "queue": [], "instances": []
    }

    for i in range(TEST_DURATION):
        action, _ = agent.predict(obs, deterministic=True)
        action_scalar = action.flatten()[0] if isinstance(action, np.ndarray) else action[0]
        obs, reward, terminated, truncated, info = env.step(action_scalar)
        history["step"].append(i)
        history["cpu"].append(obs[0])
        history["latency"].append(obs[2])
        history["instances"].append(obs[3])
        history["queue"].append(obs[4])

        if terminated or truncated:
            break
            
    return history

def plot_comparison_results(res_ppo, res_sac, res_base):
    """
    Plots the performance of all available agents on one figure.
    """
    plt.figure(figsize=(14, 12))
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Plot 1: Latency (SLA) ---
    plt.subplot(3, 1, 1)
    if res_base: plt.plot(res_base["latency"], label="Baseline (Threshold)", color="orange", linestyle="--", linewidth=2)
    if res_ppo: plt.plot(res_ppo["latency"], label="PPO Agent", color="blue", linewidth=2, alpha=0.8)
    if res_sac: plt.plot(res_sac["latency"], label="SAC Agent", color="green", linewidth=2, alpha=0.8)
    
    plt.axhline(y=200, color="red", linestyle=":", linewidth=2, label="SLA Limit (200ms)")
    plt.ylabel("Latency (ms)", fontsize=12)
    plt.title("SLA Compliance Comparison", fontsize=14)
    plt.legend()

    # --- Plot 2: Queue Size (Backlog) ---
    plt.subplot(3, 1, 2)
    if res_base: plt.plot(res_base["queue"], label="Baseline Queue", color="orange", linestyle="--")
    if res_ppo: plt.plot(res_ppo["queue"], label="PPO Queue", color="blue")
    if res_sac: plt.plot(res_sac["queue"], label="SAC Queue", color="green")
    
    plt.ylabel("Pending Requests", fontsize=12)
    plt.title("Queue Management (Proactivity Check)", fontsize=14)
    plt.legend()

    # --- Plot 3: Resource Usage (Cost) ---
    plt.subplot(3, 1, 3)
    if res_base: plt.plot(res_base["instances"], label="Baseline Cost", color="orange", linestyle="--")
    if res_ppo: plt.plot(res_ppo["instances"], label="PPO Cost", color="blue")
    if res_sac: plt.plot(res_sac["instances"], label="SAC Cost", color="green")
    
    plt.ylabel("Active Instances (Cost)", fontsize=12)
    plt.xlabel("Simulation Steps (Time)", fontsize=12)
    plt.title("Resource Allocation Strategy", fontsize=14)
    plt.legend()

    plt.tight_layout()
    plt.savefig("final_comparison_results.png", dpi=300)
    print("\nComparison graph saved as: final_comparison_results.png")
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(CSV_FILE_PATH):
        print(f" Error: CSV file not found at {CSV_FILE_PATH}")
        exit()
        
    shared_workload = ExternalWorkloadGenerator(
        csv_path=CSV_FILE_PATH,
        load_column=CSV_COLUMN_NAME,
        scale_factor=SCALE_FACTOR , 
        start_index=0, 
    )

    ppo_agent = PPO.load(PPO_MODEL_PATH) if os.path.exists(PPO_MODEL_PATH) else None
    sac_agent = SAC.load(SAC_MODEL_PATH) if os.path.exists(SAC_MODEL_PATH) else None
    baseline_agent = ThresholdAutoScaler(high_threshold=80, low_threshold=30)

    if ppo_agent is None: print(" Warning: PPO model not found. Skipping.")
    if sac_agent is None: print("ℹInfo: SAC model not found. Skipping.")


    results_ppo = run_simulation(ppo_agent, shared_workload, "PPO") if ppo_agent else None
    results_sac = run_simulation(sac_agent, shared_workload, "SAC") if sac_agent else None
    results_base = run_simulation(baseline_agent, shared_workload, "Baseline")
    print_comparison_table(results_ppo, results_sac, results_base)
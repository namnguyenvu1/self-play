import subprocess
import os
import time
import shlex

# --- CONFIG ---
experiments = [
    "--env simple_adversary --good_algo mbpo --adv_algo ppo",
    "--env simple_adversary --good_algo ppo --adv_algo mbpo",
    "--env simple_crypto --good_algo mbpo --adv_algo ppo",
    "--env simple_crypto --good_algo ppo --adv_algo mbpo",
    "--env simple_push --good_algo mbpo --adv_algo ppo",
    "--env simple_push --good_algo ppo --adv_algo mbpo",
    "--env simple_tag --good_algo mbpo --adv_algo ppo",
    "--env simple_tag --good_algo ppo --adv_algo mbpo",
    "--env simple_world_comm --good_algo mbpo --adv_algo ppo",
    "--env simple_world_comm --good_algo ppo --adv_algo mbpo",
]
NUM_RUNS_PER_EXP = 2
BATCH_SIZE = 20 
LOG_DIR = "logs"

os.makedirs(LOG_DIR, exist_ok=True)

def get_clean_filename(args_str, run_id):
    """
    Turns '--env simple_push --good_algo mbpo --adv_algo ppo' 
    into 'simple_push_mbpo_vs_ppo_run1.log'
    """
    parts = args_str.split()
    # Extract values after the flags
    env = parts[parts.index("--env") + 1]
    good = parts[parts.index("--good_algo") + 1]
    adv = parts[parts.index("--adv_algo") + 1]
    
    return f"{env}_{good}_vs_{adv}_run{run_id}.log"

def run_batches():
    # Flatten all runs into a single list
    tasks = []
    for exp_args in experiments:
        # for r in range(1, NUM_RUNS_PER_EXP + 1):
        for r in range(3, 6):
            tasks.append((exp_args, r))

    for i in range(0, len(tasks), BATCH_SIZE):
        batch = tasks[i : i + BATCH_SIZE]
        processes = []
        
        print(f"--- Launching Batch {i//BATCH_SIZE + 1} ---")
        
        for exp_args, run_id in batch:
            log_name = get_clean_filename(exp_args, run_id)
            log_path = os.path.join(LOG_DIR, log_name)
            
            # Pass the log path to the agent's internal logger
            full_cmd = f"python main.py {exp_args} --runs 1 --log_file {log_path}"
            
            f = open(log_path, "w")
            p = subprocess.Popen(shlex.split(full_cmd), stdout=f, stderr=f)
            processes.append((p, f))  # remember to close f later

            print(f"  > Logging to: {log_name}")
            time.sleep(2) 
        
        # Wait for this batch to finish before moving to next
        for p, f in processes:
            p.wait()
            f.close()

if __name__ == "__main__":
    run_batches()
    print("All tasks finished.")
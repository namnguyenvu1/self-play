#!/bin/bash

# Configuration
# 1. Get the task name from the first argument (default to simple_spread)
TASK=${1:-simple_spread}
# NUM_AGENTS=3 # comment it out as i could get it from the yaml file directly
EPISODE_LENGTH=25
STEPS=500000
LOG_DIR="log_td_ppo"
HORIZON=${2:-3}

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

echo "Starting experiments for task: $TASK"

for i in {5..10}
do
    SEED=$((10 + i)) # Generates seeds 11, 12, 13, 14, 15
    LOG_FILE="$LOG_DIR/${TASK}_run_${i}_seed_${SEED}_horizon_${HORIZON}.txt"
    
    echo "Running experiment $i/5 (Seed: $SEED)... Logging to $LOG_FILE"
    
    # Run the command and redirect both standard output (1) and error (2) to the log file
    # We use unbuffer (optional, requires expect package) or just python to flush output
    # Using stdbuf -oL to force line buffering so you can see logs populate in real time
    
    HYDRA_FULL_ERROR=1 stdbuf -oL python train.py \
        task=$TASK \
        episode_length=$EPISODE_LENGTH \
        steps=$STEPS \
        enable_wandb=false \
        compile=false \
        seed=$SEED \
        horizon=$HORIZON \
        > "$LOG_FILE" 2>&1
        
    echo "Experiment $i completed."
done

echo "All 5 runs completed."
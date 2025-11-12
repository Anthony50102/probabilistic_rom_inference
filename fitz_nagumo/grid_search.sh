#!/bin/bash

# Grid search for FitzHugh-Nagumo Bayesian Operator Inference
# Tests different values of gamma (operator uncertainty) and gamma2 (ODE constraint)

# Create timestamped output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="mcmc_results_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

echo "=================================================="
echo "Grid Search - Run started at $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "=================================================="

# Parameter ranges
gamma_values=(5e1 1e1 5e0 1e0 5e-1 1e-1)
gamma2_values=(5e1 1e1 5e0 1e0 5e-1 1e-1 5e-2)

# MCMC settings (adjust for speed vs accuracy tradeoff)
NUM_WARMUP=200
NUM_SAMPLES=200
NUM_CHAINS=1

# Number of parallel jobs (adjust based on your CPU cores)
MAX_JOBS=1

# Counter for running jobs
job_count=0

# Log file for errors
log_file="$OUTPUT_DIR/grid_search.log"
echo "Grid search started at $(date)" > "$log_file"
echo "Parameters:" >> "$log_file"
echo "  gamma values: ${gamma_values[*]}" >> "$log_file"
echo "  gamma2 values: ${gamma2_values[*]}" >> "$log_file"
echo "  MCMC: ${NUM_WARMUP} warmup + ${NUM_SAMPLES} samples × ${NUM_CHAINS} chains" >> "$log_file"
echo "  Max parallel jobs: ${MAX_JOBS}" >> "$log_file"
echo "---------------------------------------------------" >> "$log_file"

total_jobs=$((${#gamma_values[@]} * ${#gamma2_values[@]}))
current_job=0

for gamma in "${gamma_values[@]}"; do
    for gamma2 in "${gamma2_values[@]}"; do
        ((current_job++))
        
        # Run in background with error handling
        (
            echo "[$(date)] [$current_job/$total_jobs] Starting: gamma=$gamma, gamma2=$gamma2" >> "$log_file"
            
            if python fitz_eval_time_simp.py \
                --gamma "$gamma" \
                --gamma2 "$gamma2" \
                --num_warmup "$NUM_WARMUP" \
                --num_samples "$NUM_SAMPLES" \
                --num_chains "$NUM_CHAINS" \
                --use_scaled="False" \
                --output_dir "$OUTPUT_DIR" \
                2>&1 | tee -a "$log_file"; then
                echo "[$(date)] [$current_job/$total_jobs] SUCCESS: gamma=$gamma, gamma2=$gamma2" >> "$log_file"
            else
                exit_code=$?
                echo "[$(date)] [$current_job/$total_jobs] FAILED: gamma=$gamma, gamma2=$gamma2 (exit code: $exit_code)" >> "$log_file"
            fi
        ) &
        
        # Increment job counter
        ((job_count++))
        
        # Wait if we've reached max parallel jobs
        if ((job_count >= MAX_JOBS)); then
            wait -n  # Wait for any one job to finish
            ((job_count--))
        fi
    done
done

# Wait for all remaining jobs to complete
wait

echo "---------------------------------------------------" >> "$log_file"
echo "Grid search completed at $(date)" >> "$log_file"
echo ""
echo "=================================================="
echo "Grid search completed!"
echo "=================================================="
echo "Results saved in: $OUTPUT_DIR/"
echo "  - results_g{gamma}_g2{gamma2}.npz (numerical results)"
echo "  - operator_trajectories_g{gamma}_g2{gamma2}.png (trajectory plots)"
echo "  - prior_operator_fit_g{gamma}_g2{gamma2}.png (prior fit plots)"
echo "  - prior_derivative_matching_g{gamma}_g2{gamma2}.png (prior derivative matching)"
echo "  - posterior_derivative_matching_g{gamma}_g2{gamma2}.png (posterior derivative matching)"
echo "Log file: $log_file"
echo "=================================================="
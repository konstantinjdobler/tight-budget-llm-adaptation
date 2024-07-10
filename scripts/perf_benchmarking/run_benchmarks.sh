#!/bin/bash

# Automatically determine the full path to the script's directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
# Navigate two levels up to the root directory, assuming the script is in 'scripts/perf_benchmarking'
ROOT_DIR=$(cd "$BASE_DIR/../.." && pwd)

# Function to handle keyboard interrupt (Ctrl+C)
handle_interrupt() {
    echo "Interrupt received, stopping the script..."
    exit 1  # Exit with a status indicating the script was interrupted
}

# Trap keyboard interrupt (SIGINT)
trap handle_interrupt SIGINT

# Function to check for OOM in log files
check_oom() {
    logfile=$1
    # Count occurrences of "optimizer_step" entries where it follows "Step stats:" on the next line
    steps=$(awk '
        /Step stats:/ {             # When "Step stats:" is found
            getline nextLine        # Read the next line
            if (nextLine ~ /optimizer_step:/) {  # Check if the next line contains "optimizer_step:"
                count++             # Increment count if the pattern is matched
            }
        }
        END {
            print count+0           # Print count, ensure its treated as numeric
        }
    ' "$logfile")
    if [[ $steps -lt 10 ]]; then
        echo 1  # Indicates OOM
    else
        echo 0
    fi
}

# Create the benchmark_logs directory if it does not exist
if [ ! -d "$BASE_DIR/benchmark_logs2" ]; then
    mkdir "$BASE_DIR/benchmark_logs2"
fi

# Define all parameters
mb_sizes=(1 2 4 8)
checkpointing=("true" "false")
sharding_strategies=("FULL_SHARD" "SHARD_GRAD_OP")
precisions=("bf16-true" "bf16-mixed")
device_counts=(8 4 2)  # Ordered to prefer configurations with more GPUs first
no_sync=("false" "true")
paged_adamw=("true" "false")

# Dictionary to keep track of OOM status
declare -A oom_status

should_skip() {
    local mb=$1 cp=$2 shard=$3 precision=$4 devices=$5 no_sync=$6 paged_adamw=$7
    local check_mb check_cp check_shard check_precision check_devices check_no_sync check_paged_adamw

    # Iterate over all mb sizes less than or equal to the current one to check for previous OOMs
    for check_mb in "${mb_sizes[@]}"; do
        if [[ $check_mb -le $mb ]]; then
            # Check all combinations of precision and devices with less or equal memory-friendly settings
            for check_precision in "bf16-true" "$precision"; do  # Less memory first
                for check_devices in $(seq $devices 8); do  # More devices first
                    for check_cp in "true" "$cp"; do  # More memory-friendly first
                        for check_shard in "FULL_SHARD" "$shard"; do  # More memory-friendly first
                            for check_no_sync in "false" "$no_sync"; do  # Less memory first
                                for check_paged_adamw in "true" "$paged_adamw"; do  # Less memory first
                                    # Only check if settings are equal or more memory-friendly than current
                                    if [[ ($check_cp == "true" || $check_cp == $cp) && ($check_shard == "FULL_SHARD" || $check_shard == $shard) && ($check_no_sync == "false" || $check_no_sync == $no_sync) && ($check_paged_adamw == "true" || $check_paged_adamw == $paged_adamw) ]]; then
                                        if [[ ${oom_status[$check_mb,$check_cp,$check_shard,$check_precision,$check_devices,$check_no_sync,$check_paged_adamw]} == "1" ]]; then
                                            oom_status[$mb,$cp,$shard,$precision,$devices,$no_sync,$paged_adamw]=1  # Preemptively mark as OOM
                                            return 0  # Skip this configuration
                                        fi
                                    fi
                                done
                            done
                        done
                    done
                done
            done
        fi
    done

    return 1  # Do not skip this configuration
}

# Loop over configurations
for precision in "${precisions[@]}"; do
    for devices in "${device_counts[@]}"; do
        for mb in "${mb_sizes[@]}"; do
            for cp in "${checkpointing[@]}"; do
                for shard in "${sharding_strategies[@]}"; do
                    for ns in "${no_sync[@]}"; do
                        for pa in "${paged_adamw[@]}"; do
                            cfg="--mb $mb --activation_checkpointing $cp --fsdp_sharding_strategy $shard --precision $precision --gradient_accumulation_no_sync $ns --use_paged_adamw $pa"
                            device_range=$(seq -s, 0 $((devices - 1)))
                            name="run-${precision}-g${devices}-${cfg// /-}"
                            logfile="$BASE_DIR/benchmark_logs2/$name.log"

                            # check if logfile already exists. if so, skip and set oom_status based on check_oom 
                            if [ -f "$logfile" ]; then
                                oom=$(check_oom "$logfile")
                                oom_status[$mb,$cp,$shard,$precision,$devices,$ns,$pa]=$oom
                                if [[ $oom -eq 1 ]]; then
                                    echo "OOM detected for $cfg with -g $device_range."
                                fi
                                continue
                            fi
                            
                            # Check if we should skip this configuration based on earlier results and current parameters
                            if should_skip $mb $cp $shard $precision $devices $ns $pa; then
                                echo "Skipping $cfg with -g $device_range due to prior OOM with more favorable memory settings."
                                # create empty log file to avoid re-running
                                touch "$logfile"
                                # put in some dummy OOM content
                                echo "OOM detected for $cfg with -g $device_range." > "$logfile"
                                continue
                            fi

                            echo "Running configuration: $cfg with devices -g $device_range > $logfile"
                            (cd "$ROOT_DIR" && bash "./scripts/run-docker.sh" -g $device_range python train.py --config_path "cfgs/hindsight.yml" "./cfgs/perf-search2.yml" $cfg -n $name > "$BASE_DIR/benchmark_logs2/$name.log")
                            oom=$(check_oom "$logfile")
                            oom_status[$mb,$cp,$shard,$precision,$devices,$ns,$pa]=$oom
                            if [[ $oom -eq 1 ]]; then
                                echo "OOM detected for $cfg with -g $device_range."
                            fi
                        done
                    done
                done
            done
        done
    done
done

# Call the analysis Python script
python "$BASE_DIR/analyze_logs.py"

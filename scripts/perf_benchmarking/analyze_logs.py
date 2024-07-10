"""
Many sins were committed together with ChatGPT to produce script and `run_benchmarks.sh`.
Do not use this for anything but reproducing the paper's results.
"""

import os
import re


def is_better_than_recorded(result, best_record):
    if result[0] == "OOM":
        return False  # Not a valid result to compare
    if best_record[0] == "OOM":
        return True

    try:
        current_time = float(result[0].split("s")[0])
        recorded_time = float(best_record[0].split("s")[0])
    except ValueError:
        # Handle cases where conversion to float fails
        print("Error: Invalid time format.")
        return False

    return current_time < recorded_time


def generate_full_latex_table(results):
    # Sort results by Total GPU Hours (with OOM going last)
    latex_table = "\\clearpage\n\\onecolumn\n"
    latex_table += "\\begin{longtable}{llcccc}\n"
    latex_table += "\\caption{Performance benchmarking results grouped by (precision, \\# GPUs), sorted by Total H100 80GB GPU Hours. We abbreviate \\texttt{mb := micro-batch size}, \\texttt{ckpt := activation checkpointing}, and \\texttt{sharding := FSDP ZeRO stage sharding}. FSDP \\texttt{grad\_op} shards only gradients and optimizer states, whereas \\texttt{full} additionally shards the model weights.}\n"
    latex_table += "\\label{tab:full-perf-results}\n"
    latex_table += "\\toprule\n"
    latex_table += "Precision & \# GPUs & \\texttt{(mb, ckpt, sharding)} & Max. CUDA RAM &Step time & Total GPU Hours \\\\ \n"
    latex_table += "\\midrule\n"
    latex_table += "\\endfirsthead\\\n"
    latex_table += "\\toprule\n"
    latex_table += "Precision & \# GPUs & \\texttt{(mb, ckpt, sharding)} & Max. CUDA RAM &Step time & Total GPU Hours \\\\ \n"
    latex_table += "\\midrule\n"
    latex_table += "\\endhead\n"
    latex_table += "\\bottomrule\n"
    latex_table += "\\endfoot\n"

    last_prec_num_devices = None

    sorted_results = sorted(results.items(), key=lambda x: (x[0][1], x[0][0]))
    for (precision, num_devices), config_results in sorted_results:
        config_results.sort(key=lambda x: (x[3] == "OOM", x[3]))
        for individual_result in config_results:
            config, max_cuda_ram, runtime, gpu_hours = individual_result

            # parse config name string to retrive --mb --activation_checkpointing and --fsdp_sharding_strategy
            mb = re.search(r"--mb-(\d+)", config).group(1)
            ckpt = re.search(r"--activation_checkpointing-(\w+)", config).group(1)
            sharding = re.search(r"--fsdp_sharding_strategy-(\w+)", config).group(1)
            no_sync = re.search(r"--gradient_accumulation_no_sync-(\w+)", config).group(1)
            paged_adamw = re.search(r"--use_paged_adamw-(\w+)", config).group(1)
            if runtime == "OOM":
                max_cuda_ram = "OOM"
                runtime = "N/A"
            config = (
                "\\texttt{("
                + mb
                + ", "
                + ("yes" if ckpt == "true" else "no")
                + ", "
                + ("full" if sharding == "FULL_SHARD" else "grad\\_op")
                + ", "
                + ("no\_sync" if no_sync == "true" else "sync")
                + ", "
                + ("paged" if paged_adamw == "true" else "no\_paged")
                + ")}"
            )

            precision_str = "pure \\bfp{}" if (precision.strip() == "bf16-true") else "mixed-precision \\bfp{}"
            if last_prec_num_devices != (precision, num_devices):
                last_prec_num_devices = (precision, num_devices)
                latex_table += "\\midrule\n"
            latex_table += f"{precision_str} & {num_devices} & {config} & {max_cuda_ram} & {runtime} & {gpu_hours} \\\\ \n"
    latex_table += "\\end{longtable}\n"
    latex_table += "\\clearpage\n\\twocolumn\n"

    return latex_table


def generate_best_results_table(group_results):
    results = []

    for key, value in group_results.items():
        precision, num_devices = key
        best_result, best_config = value
        average_runtime, total_gpu_hours = best_result[0], best_result[1]
        results.append((precision, num_devices, best_config, average_runtime, total_gpu_hours))

    # Sort results by Total GPU Hours (with OOM going last)
    results.sort(key=lambda x: (x[4] == "OOM", x[4]))

    # Group results by num_devices
    grouped_results = {}
    for result in results:
        precision, num_devices, best_config, average_runtime, total_gpu_hours = result
        if num_devices not in grouped_results:
            grouped_results[num_devices] = []
        grouped_results[num_devices].append(result)

    print(grouped_results)

    # Sort groups by precision (mixed-precision first, pure second)
    sorted_flat = []
    for num_devices_group in sorted(grouped_results.items(), key=lambda x: x[0]):
        # put bf16-mixed first, then bf16-true
        sorted_flat.extend(sorted(num_devices_group[1], key=lambda x: x[0].strip()))

    results = sorted_flat

    latex_table = "\\begin{table}[ht]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{llccc}\n"
    latex_table += "\\toprule\n"
    latex_table += "Precision & \# GPUs & Best Config & GPU Hours & Speedup \\\\ \n"
    print(results)
    last_num_devices = None
    for i, (precision, num_devices, best_config, average_runtime, total_gpu_hours) in enumerate(results):
        precision_str = "pure" if (precision.strip() == "bf16-true") else "mixed"
        print(precision, num_devices, best_config, total_gpu_hours)
        if last_num_devices != num_devices:
            last_num_devices = num_devices
            latex_table += "\\midrule\n"
        if precision_str.strip() == "mixed":
            speedup = "0\%"
        else:
            if results[i - 1][3] in ["OOM", "N/A"]:
                # best_config = "--"
                speedup = "$\\infty$"
            else:
                # print(results[i - 1][4])
                prev_gpu_ours = results[i - 1][4]
                prev_gpu_ours = float(prev_gpu_ours.split(" ")[0])

                cur_gpu_hours = float(total_gpu_hours.split(" ")[0])
                speedup = f"{float(((prev_gpu_ours / cur_gpu_hours) - 1) * 100):.1f}\%"

        mb = re.search(r"--mb-(\d+)", best_config).group(1)
        ckpt = re.search(r"--activation_checkpointing-(\w+)", best_config).group(1)
        sharding = re.search(r"--fsdp_sharding_strategy-(\w+)", best_config).group(1)
        no_sync = re.search(r"--gradient_accumulation_no_sync-(\w+)", best_config).group(1)
        paged_adamw = re.search(r"--use_paged_adamw-(\w+)", best_config).group(1)
        if average_runtime == "OOM":
            average_runtime = "N/A"
            total_gpu_hours = "--"
        else:
            total_gpu_hours = f"{float(total_gpu_hours.split(' ')[0]):.1f}"

        # else:
        best_config = (
            "\\texttt{("
            + mb
            + ", "
            + ("yes" if ckpt == "true" else "no")
            + ", "
            + ("full" if sharding == "FULL_SHARD" else "grad\\_op")
            + ", "
            + ("nosync" if no_sync == "true" else "sync")
            + ", "
            + ("paged" if paged_adamw == "true" else "nopaged")
            + ")}"
        )
        latex_table += f"{precision_str} & {num_devices} & {best_config} & {total_gpu_hours.split(' ')[0]} & {speedup} \\\\ \n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Best configuration per (Precision, Num Devices) group, sorted by Total GPU Hours.}\n"
    latex_table += "\\label{tab:best-results}\n"
    latex_table += "\\end{table}\n"
    return latex_table


def parse_log(filename):
    times = []
    num_devices = 1  # Default value if not specified in the logs
    precision = "unknown"  # Default value
    step_number = None
    time_found = False
    max_cuda_ram = 0  # Initialize max CUDA RAM

    with open(filename, "r") as file:
        for line in file:
            if "Step stats:" in line:
                # Reset step number and time found flag for each 'Step stats:' entry
                step_number = None
                time_found = False

            if "optimizer_step:" in line and not step_number:
                # Extract step number when found
                step_match = re.search(r"optimizer_step: (\d+)", line)
                if step_match:
                    step_number = int(step_match.group(1))
                    if not (2 <= step_number <= 11):
                        # Reset if step number is outside of the range we care about
                        step_number = None

            if "global_step_time:" in line and step_number:
                # Extract time when found and valid step number is set
                time_match = re.search(r"global_step_time: ([\d\.]+)s", line)
                if time_match:
                    time = float(time_match.group(1))
                    times.append(time)
                    time_found = True  # Mark that time was found for the current valid step
                else:
                    time_minute_fmt_match = re.search(r"global_step_time: (\d+):([\d\.]+)m", line)
                    if time_minute_fmt_match:
                        time = int(time_minute_fmt_match.group(1)) * 60 + float(time_minute_fmt_match.group(2))
                        times.append(time)
                        time_found = True

            # Extract max_cuda_ram when found
            if "max_cuda_ram:" in line:
                ram_match = re.search(r"max_cuda_ram: ([\d\.]+) GB", line)
                if ram_match:
                    ram = float(ram_match.group(1))
                    max_cuda_ram = max(max_cuda_ram, ram)

            # Additional checks or breaks can be added here if necessary
            if time_found and step_number == 11:
                # Break if the end of the range of steps we care about has been processed
                break

    # Extract device and precision information if stored in the file
    with open(filename, "r") as file:
        content = file.read()
        match = re.search(r"args.num_devices=(\d+)", content)
        if match:
            num_devices = int(match.group(1))
        else:
            # try to extract from filename if not found in the content
            match = re.search(r"-g(\d+)---", filename)
            num_devices = int(match.group(1))
        match = re.search(r"-(bf16-\w+)-g", filename)  # Adjust regex as necessary
        if match:
            precision = match.group(1)
    if len(times) < 10:
        print(f"Warning: Not enough steps found in {filename}")
        print(f"Times: {times}")
        print(f"Num devices: {num_devices}")
        print(f"Precision: {precision}")
        print(f"Max CUDA RAM: {max_cuda_ram}")
        print("-------------")

    return times, num_devices, precision, max_cuda_ram


# Update the printing and handling of results in the main function
def main():
    log_dir = "./benchmark_logs2"
    logs = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith(".log")]
    group_results = {}
    full_results = {}

    print("Config Name, Precision, Num Devices, Average Runtime, Total GPU Hours, Max CUDA RAM")
    for log in logs:
        times, num_devices, precision, max_cuda_ram = parse_log(log)
        config_name = os.path.basename(log)[:-4]
        if len(times) < 10:
            result = ("OOM", "N/A", "N/A")
        else:
            average_time = sum(times) / len(times)
            time_std = (sum((t - average_time) ** 2 for t in times) / len(times)) ** 0.5
            total_steps = 7680
            total_gpu_hours = (average_time * total_steps * int(num_devices)) / 3600
            result = (f"{average_time:.2f}s ± {time_std:.2f}s", f"{total_gpu_hours:.2f} hours", f"{max_cuda_ram} GB")

        # Print results
        print(f"{config_name}, {precision}, {num_devices}, {result[0]}, {result[1]}, {result[2]}")
        if full_results.get((precision, num_devices)) is None:
            full_results[(precision, num_devices)] = [(config_name, result[2], result[0], result[1])]
        else:
            full_results[(precision, num_devices)].append((config_name, result[2], result[0], result[1]))

        # Update the best results per group
        key = (precision, num_devices)
        # Usage within your script
        if key not in group_results or is_better_than_recorded(result, group_results[key][0]):
            # Update the group_results with new best result
            group_results[key] = (result, config_name)

    print(generate_full_latex_table(full_results))
    # Print the best configuration for each group
    print("\nBest configuration per (Precision, Num Devices) group:")
    for key, value in group_results.items():
        precision, num_devices = key
        best_result, best_config = value
        print(
            f"{precision}, {num_devices} devices: {best_config} with average runtime {best_result[0]}, total GPU hours {best_result[1]}, and max CUDA RAM {best_result[2]}"
        )
    print(generate_best_results_table(group_results))


if __name__ == "__main__":
    main()

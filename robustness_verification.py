import os
import subprocess
import yaml
import glob

# Custom YAML representer for lists to enforce flow style
def represent_list_flow_style(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(list, represent_list_flow_style)

# Define constants
ABCROWN_CMD = "python alpha-beta-CROWN/complete_verifier/abcrown.py --config"
RESULTS_FILE = "results/results_robustness_verification.txt"
CONFIG_FOLDER = "configs_robustness"

# Define paths
VNNLIB_PATHS = [
    "vnnlib_files/specifications/safe",
    "vnnlib_files/specifications/perform"
]
ONNX_PATH = "Neural_Networks_Under_Verification"

# Create config folder if needed
os.makedirs(CONFIG_FOLDER, exist_ok=True)

# Define robustness parameters
ROBUSTNESS_TYPES = ["bright_0.2", "bright_0.4", "mb_2", "mb_4"]
ONNX_MODELS = ["Nvidia", "Resnet18"]

# Generate YAML configurations
for vnnlib_path in VNNLIB_PATHS:
    for robustness_type in ROBUSTNESS_TYPES:
        for onnx_model in ONNX_MODELS:
            # Find matching VNNLIB files
            vnnlib_pattern = os.path.join(vnnlib_path, f"*{robustness_type}*.vnnlib")
            vnnlib_files = glob.glob(vnnlib_pattern)
            
            # Find matching ONNX files
            onnx_pattern = os.path.join(ONNX_PATH, f"*{onnx_model}*{robustness_type}*.onnx")
            onnx_files = glob.glob(onnx_pattern)
            
            # Debugging: Print file search results
            print(f"Searching for VNNLIB files with pattern: {vnnlib_pattern}")
            print(f"Found VNNLIB files: {vnnlib_files}")
            print(f"Searching for ONNX files with pattern: {onnx_pattern}")
            print(f"Found ONNX files: {onnx_files}")
            
            if not vnnlib_files:
                print(f"No VNNLIB files found for {robustness_type} in {vnnlib_path}")
            if not onnx_files:
                print(f"No ONNX files found for {robustness_type} and {onnx_model} in {ONNX_PATH}")
            
            if not vnnlib_files or not onnx_files:
                print(f"Skipping {robustness_type} for {onnx_model} - files not found")
                continue

            for vnnlib_file in vnnlib_files:
                for onnx_file in onnx_files:
                    config_data = {
                        "model": {
                            "onnx_path": onnx_file,
                            "input_shape": [-1, 8]  # Will now be formatted as [-1, 8]
                        },
                        "specification": {
                            "vnnlib_path": vnnlib_file
                        },
                        "solver": {
                            "batch_size": 2000,
                            "alpha-crown": {
                                "iteration": 100,
                                "lr_alpha": 0.1
                            }
                        },
                        "bab": {
                            "timeout": 2000,
                            "branching": {
                                "reduceop": "min",
                                "method": "fsb",
                                "candidates": 50
                            }
                        },
                        "attack": {
                            "pgd_steps": 50000,
                            "pgd_restarts": 50
                        }
                    }

                    # Create filename components
                    vnnlib_base = os.path.basename(vnnlib_file).replace(".vnnlib", "")
                    onnx_base = os.path.basename(onnx_file).replace(".onnx", "")
                    config_filename = f"{vnnlib_base}_{onnx_base}.yaml"
                    config_path = os.path.join(CONFIG_FOLDER, config_filename)

                    if not os.path.exists(config_path):
                        print(f"Creating YAML file: {config_filename}")
                        with open(config_path, "w") as f:
                            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                    else:
                        print(f"YAML file already exists: {config_filename}")

# Run verification process
results = []
for config_file in os.listdir(CONFIG_FOLDER):
    if config_file.endswith(".yaml"):
        config_path = os.path.join(CONFIG_FOLDER, config_file)
        cmd = f"{ABCROWN_CMD} {config_path}"
        try:
            output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, timeout=300).decode("utf-8")
            last_two_lines = output.splitlines()[-2:]
            
            # Print to terminal
            print(f"Results for {config_file}:")
            print("\n".join(last_two_lines))
            print("-" * 50)  # Separator for readability
            
            # Save to results list
            results.append(f"{config_file}:\n" + "\n".join(last_two_lines) + "\n\n")
        except Exception as e:
            error_msg = f"{config_file}: ERROR - {str(e)}\n\n"
            print(error_msg)
            results.append(error_msg)

# Sort results alphabetically by config file name
results.sort()

# Write sorted results to the text file
with open(RESULTS_FILE, "w") as results_f:
    for result in results:
        results_f.write(result)

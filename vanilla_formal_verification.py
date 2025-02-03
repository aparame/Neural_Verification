import os
import subprocess
import yaml
import glob

# Custom YAML representer for lists to enforce flow style
def represent_list_flow_style(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(list, represent_list_flow_style)

# Define constants
ABCROWN_CMD = "python /home/adi2440/Desktop/Neurosymbolic/alpha-beta-CROWN/complete_verifier/abcrown.py --config"
RESULTS_FILE = "results/results_vanilla_formal_verification.txt"
CONFIG_FOLDER = "NEW_configs_vanilla"
# CONFIG_FOLDER = "configs_vanilla"

# Define vnnlib and onnx paths
# Define paths
# VNNLIB_PATHS = [
#     "vnnlib_files/specifications/safe",
#     "vnnlib_files/specifications/perform"
# ]
VNNLIB_PATHS = [
    "configs"
]

# ONNX_PATHS = ["Neural_Networks_Under_Verification/formal_Nvidia.onnx",
# 	     "Neural_Networks_Under_Verification/formal_Resnet18.onnx"]


ONNX_PATHS = ["saved_models/combined/NvidiaNet_GMVAE_vanilla.onnx",
	     "saved_models/combined/ResNet18_GMVAE_vanilla.onnx"]


# Create config folder if it doesn't exist
os.makedirs(CONFIG_FOLDER, exist_ok=True)

# Generate yaml files
for vnnlib_path in VNNLIB_PATHS:
    for onnx_path in ONNX_PATHS:
        for vnnlib_file in glob.glob(os.path.join(vnnlib_path, "*formal*.vnnlib")):
            config_data = {
                "model": {
                    "onnx_path": onnx_path,
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

            # Create yaml file
            base_name = f"{os.path.basename(vnnlib_file).split('.')[0]}_{os.path.basename(onnx_path).split('.')[0]}"
            config_path = os.path.join(CONFIG_FOLDER, f"{base_name}.yaml")
            
            if not os.path.exists(config_path):
                with open(config_path, "w") as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

# Run abcrown.py for each yaml file
with open(RESULTS_FILE, "w") as results_f:
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
                
                # Save to results file
                results_f.write(f"{config_file}:\n" + "\n".join(last_two_lines) + "\n\n")
            except Exception as e:
                error_msg = f"{config_file}: ERROR - {str(e)}"
                print(error_msg)
                results_f.write(error_msg + "\n\n")

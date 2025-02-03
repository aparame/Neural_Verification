import pandas as pd
import numpy as np
import glob
import os

latent_dim = 8
folder_path = 'encodings/'  # Folder containing all CSV files
output_folder = 'encodings/processed/'  # Folder to save processed files

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Step 1: Get a list of all CSV files in the folder
file_paths = glob.glob(os.path.join(folder_path, '*.csv'))

# Step 2: Process each file and save results individually
for file_path in file_paths:
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Filter the data based on the specified steering value ranges
    df = df[(df[df.columns[0]] >= -0.4) & (df[df.columns[0]] <= 0.4)]
    
    # Define the steering ranges
    ranges = [(-0.4,0), (0.0,0.4)]
    # ranges = [(-0.4, -0.15), (-0.15, 0.15), (0.15, 0.4)]
    
    # Calculate upper and lower bounds for each steering range
    bounds = []
    for lower, upper in ranges:
        # Select rows within the current range
        group = df[(df[df.columns[0]] >= lower) & (df[df.columns[0]] < upper)]
        
        # Calculate the lower and upper bounds for each latent dimension
        lower_bounds = group.iloc[:, -latent_dim:].min().values
        upper_bounds = group.iloc[:, -latent_dim:].max().values
        
        # Append the results for each range
        bounds.append({
            'steering_range': f"{lower:.2f} to {upper:.2f}",
            **{f'lower_bound_{i+1}': lower_bounds[i] for i in range(latent_dim)},
            **{f'upper_bound_{i+1}': upper_bounds[i] for i in range(latent_dim)}
        })
    
    # Create DataFrame with the calculated bounds
    bounds_df = pd.DataFrame(bounds)
    
    # Save each file’s result to a new CSV file in the output folder
    output_path = os.path.join(output_folder, f'safe_{os.path.basename(file_path)}')
    bounds_df.to_csv(output_path, index=False)

    print(f"Upper and lower bounds for {os.path.basename(file_path)} saved to {output_path}")

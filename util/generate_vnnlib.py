import pandas as pd
import numpy as np
import os
import glob
import csv
import re

# Constants
latent_dim = 8
folder_path = '../encodings/'  # Folder containing all raw CSV files
processed_folder = '../encodings/processed/'  # Folder to save processed files
vnnlib_folder = '../vnnlibs/'  # Folder to save VNNLIB files

# Ensure the output folders exist
os.makedirs(processed_folder, exist_ok=True)
os.makedirs(vnnlib_folder, exist_ok=True)

# Define the steering ranges for both cases
ranges_list = {
    'safe': [(-0.4, 0), (0.0, 0.4)],
    'perform': [(-0.4, -0.15), (-0.15, 0.15), (0.15, 0.4)]
}

def process_csv_files():
    """Process raw CSV files and calculate bounds for safe and perform ranges."""
    # Step 1: Get a list of all CSV files in the folder
    file_paths = glob.glob(os.path.join(folder_path, '*.csv'))

    # Step 2: Process each file for both ranges
    for file_path in file_paths:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Filter the data based on the specified steering value ranges
        df = df[(df[df.columns[0]] >= -0.4) & (df[df.columns[0]] <= 0.4)]
        
        # Process for both ranges
        for suffix, ranges in ranges_list.items():
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
            output_path = os.path.join(processed_folder, f'{suffix}_{os.path.basename(file_path)}')
            bounds_df.to_csv(output_path, index=False)

            print(f"Upper and lower bounds for {os.path.basename(file_path)} ({suffix}) saved to {output_path}")

def generate_vnnlib(row, output_filename):
    """Generate a VNNLIB file from a row of aggregated CSV data."""
    try:
        # Parse Y_0 output constraints (e.g., "-0.40 to -0.15")
        y_range = re.split(r'\s*to\s*', row[0].strip())
        if len(y_range) != 2:
            raise ValueError(f"Invalid Y_0 range: {row[0]}")
        y_lower, y_upper = y_range

        # Declare variables (X_0 to X_7 and Y_0)
        input_decls = '\n'.join([f'(declare-const X_{i} Real)' for i in range(8)])
        output_decl = '(declare-const Y_0 Real)\n'

        # Build input constraints (X_0 to X_7)
        input_constraints = []
        for i in range(8):
            lower = row[1 + i].strip() # Lower bounds (columns 1-8)
            upper = row[9 + i].strip()  # Upper bounds (columns 9-16)

            input_constraints.append(f'(assert (>= X_{i} {lower}))')
            input_constraints.append(f'(assert (<= X_{i} {upper}))')
        input_constraints_str = '\n\n'.join(input_constraints)

        # Build output constraints (Y_0)
        output_constraints = f'(assert (>= Y_0 {y_lower}))\n(assert (<= Y_0 {y_upper}))'

        # Combine into VNNLIB content
        vnnlib_content = f"""; Generated VNNLIB file
{input_decls}
{output_decl}

; Input constraints:

{input_constraints_str}

; Output constraints:

{output_constraints}

; End of constraint set"""

        # Write to file
        with open(output_filename, 'w') as f:
            f.write(vnnlib_content)
        print(f"Generated: {output_filename}")

    except Exception as e:
        print(f"Error generating {output_filename}: {str(e)}")

def process_vnnlib_files():
    """Generate VNNLIB files from aggregated CSV files."""
    for csv_file in glob.glob(os.path.join(processed_folder, '*.csv')):
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row_idx, row in enumerate(reader, 1):
                base_name = os.path.splitext(os.path.basename(csv_file))[0]
                output_file = os.path.join(
                    vnnlib_folder, 
                    f"{base_name}_phi_{row_idx}.vnnlib"
                )
                generate_vnnlib(row, output_file)

def main():
    # Step 1: Process raw CSVs into aggregated bounds
    process_csv_files()
    
    # Step 2: Generate VNNLIB files from aggregated CSVs
    process_vnnlib_files()

if __name__ == "__main__":
    main()
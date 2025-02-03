import pandas as pd
import numpy as np
import glob
import os
import csv
import re

def process_csv_files(input_folder, output_folder, latent_dim=8):
    os.makedirs(output_folder, exist_ok=True)
    file_paths = glob.glob(os.path.join(input_folder, '*.csv'))
    
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        df = df[(df[df.columns[0]] >= -0.4) & (df[df.columns[0]] <= 0.4)]
        ranges = [(-0.4, 0), (0.0, 0.4)]
        
        bounds = []
        for lower, upper in ranges:
            group = df[(df[df.columns[0]] >= lower) & (df[df.columns[0]] < upper)]
            lower_bounds = group.iloc[:, -latent_dim:].min().values
            upper_bounds = group.iloc[:, -latent_dim:].max().values
            bounds.append({
                'steering_range': f"{lower:.2f} to {upper:.2f}",
                **{f'lower_bound_{i+1}': lower_bounds[i] for i in range(latent_dim)},
                **{f'upper_bound_{i+1}': upper_bounds[i] for i in range(latent_dim)}
            })
        
        bounds_df = pd.DataFrame(bounds)
        output_path = os.path.join(output_folder, f'safe_{os.path.basename(file_path)}')
        bounds_df.to_csv(output_path, index=False)
        print(f"Processed: {output_path}")

def generate_vnnlib(row, output_filename):
    y_range = re.split(r'\s*to\s*', row[0].strip())
    if len(y_range) != 2:
        raise ValueError(f"Invalid Y_0 range in row: {row[0]}")
    y_lower, y_upper = y_range
    
    input_declarations = '\n'.join([f'(declare-const X_{i} Real)' for i in range(8)])
    output_declaration = '(declare-const Y_0 Real)\n'
    
    input_constraints = []
    for i in range(8):
        lower = float(row[1 + i].strip()) + 0.5
        upper = float(row[9 + i].strip()) - 0.5
        input_constraints.append(f'(assert (>= X_{i} {lower}))')
        input_constraints.append(f'(assert (<= X_{i} {upper}))')
    input_constraints_str = '\n\n'.join(input_constraints)
    
    output_constraints = f'(assert (>= Y_0 {y_lower}))\n(assert (<= Y_0 {y_upper}))'
    
    vnnlib_content = f"""; Generated VNNLIB file
{input_declarations}
{output_declaration}

; Input constraints:

{input_constraints_str}

; Output constraints:

{output_constraints}

; End of constraint set"""
    
    with open(output_filename, 'w') as f:
        f.write(vnnlib_content)

def process_vnnlib_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            csv_filepath = os.path.join(input_folder, filename)
            with open(csv_filepath, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row_idx, row in enumerate(reader, 1):
                    output_filename = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_row_{row_idx}.vnnlib")
                    try:
                        generate_vnnlib(row, output_filename)
                        print(f"Generated: {output_filename}")
                    except Exception as e:
                        print(f"Error processing row {row_idx} in {csv_filepath}: {e}")

def main():
    input_folder = 'encodings/'
    processed_folder = 'encodings/processed/'
    output_folder = 'configs/'
    
    process_csv_files(input_folder, processed_folder)
    process_vnnlib_files(processed_folder, output_folder)
    
if __name__ == "__main__":
    main()

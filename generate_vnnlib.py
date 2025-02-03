import csv
import re
import os

def generate_vnnlib(row, output_filename):
    # Process Y_0 output constraints (split on "to" with optional whitespace)
    y_range = re.split(r'\s*to\s*', row[0].strip())
    if len(y_range) != 2:
        raise ValueError(f"Invalid Y_0 range in row: {row[0]}")
    y_lower, y_upper = y_range

    # Generate variable declarations
    input_declarations = '\n'.join([f'(declare-const X_{i} Real)' for i in range(8)])
    output_declaration = '(declare-const Y_0 Real)\n'

    # Generate input constraints for X_0 to X_7
    input_constraints = []
    for i in range(8):
        lower = float(row[1 + i].strip()) + 0.5  # Columns 2-9: lower bounds + relaxation
        upper = float(row[9 + i].strip()) - 0.5 # Columns 10-17: upper bounds - relaxation
        lower = str(lower)
        upper = str(upper)
        input_constraints.append(f'(assert (>= X_{i} {lower}))')
        input_constraints.append(f'(assert (<= X_{i} {upper}))')
    input_constraints_str = '\n\n'.join(input_constraints)

    # Generate output constraints for Y_0
    output_constraints = f'(assert (>= Y_0 {y_lower}))\n(assert (<= Y_0 {y_upper}))'

    # Combine all parts into the VNNLIB content
    vnnlib_content = f"""; Generated VNNLIB file
{input_declarations}
{output_declaration}

; Input constraints:

{input_constraints_str}

; Output constraints:

{output_constraints}

; End of constraint set"""

    # Write to the output file
    with open(output_filename, 'w') as f:
        f.write(vnnlib_content)

def process_csv_file(csv_filepath, output_folder):
    with open(csv_filepath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row (e.g., "steering_range")
        for row_idx, row in enumerate(reader, 1):
            # Generate output filename based on CSV filename and row index
            csv_filename = os.path.basename(csv_filepath)
            output_filename = os.path.join(output_folder, f"{os.path.splitext(csv_filename)[0]}_row_{row_idx}.vnnlib")
            try:
                generate_vnnlib(row, output_filename)
                print(f"Generated: {output_filename}")
            except Exception as e:
                print(f"Error processing row {row_idx} in {csv_filepath}: {e}")

def main(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all CSV files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            csv_filepath = os.path.join(input_folder, filename)
            print(f"Processing: {csv_filepath}")
            process_csv_file(csv_filepath, output_folder)

if __name__ == "__main__":
    import sys
    # if len(sys.argv) != 3:
    #     print("Usage: python generate_vnnlib.py <input_folder> <output_folder>")
    #     sys.exit(1)
    input_folder = 'encodings/processed'
    output_folder = 'configs'
    main(input_folder, output_folder)
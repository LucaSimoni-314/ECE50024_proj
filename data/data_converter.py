import csv
import re

def convert_txt_table_to_csv(input_filename, output_filename='output_results.csv'):
    rows = []
    
    try:
        with open(input_filename, 'r', encoding='utf-8') as file:
            for line in file:
                # 1. Skip lines that are just separators (==== or ----)
                if re.match(r'^[=\-\s]+$', line) or not line.strip():
                    continue
                
                # 2. Split by pipe and strip whitespace
                columns = [col.strip() for col in line.split('|')]
                
                # 3. Remove empty strings (caused by leading/trailing pipes)
                columns = [col for col in columns if col]
                
                if columns:
                    rows.append(columns)

        # 4. Write the parsed rows to a CSV file
        if rows:
            with open(output_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            print(f"Success! {len(rows) - 1} data rows saved to {output_filename}")
        else:
            print("No data found in the file.")

    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")

# Usage
convert_txt_table_to_csv('N100_data.txt')
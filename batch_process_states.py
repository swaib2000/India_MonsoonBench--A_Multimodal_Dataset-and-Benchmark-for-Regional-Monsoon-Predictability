#!/usr/bin/env python3
"""
Batch Processing Script for Remaining Indian States

This script generates GEE scripts for all remaining states.
Requires a CSV file with state names and precipitation normals.

CSV format: state,normals
Where normals is comma-separated: June,July,August,September

Usage:
    python batch_process_states.py --csv states_normals.csv
"""

import os
import argparse
import csv
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description='Batch process remaining states')
    parser.add_argument('--csv', required=True, help='CSV file with state,normals')

    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV file not found: {args.csv}")
        sys.exit(1)

    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            state = row['state']
            normals = row['normals']

            print(f"\nProcessing {state} with normals: {normals}")

            # Generate GEE script
            gee_file = f"{state.lower().replace(' ', '_')}_gee.js"
            cmd = f"python generate_gee_script.py --state '{state}' --normals '{normals}'"

            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
                with open(gee_file, 'w') as f:
                    f.write(result.stdout)
                print(f"GEE script saved: {gee_file}")
            except subprocess.CalledProcessError as e:
                print(f"Error generating {gee_file}: {e.stderr}")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == '__main__':
    main()

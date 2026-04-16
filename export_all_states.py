#!/usr/bin/env python3
"""
Export Data for All States using Earth Engine Python API

This script reads the states_normals.csv and exports multimodal data
for all remaining states programmatically.

Usage:
    python export_all_states.py --project "your-gee-project-id"

Note: This will start 176 export tasks (8 per state × 22 states) for complete
12-month data collection (2020-2024).
"""

import os
import argparse
import csv
import subprocess
import sys
import time

def main():
    parser = argparse.ArgumentParser(description='Export data for all states using EE Python API')
    parser.add_argument('--project', required=True, help='Google Earth Engine project ID')

    args = parser.parse_args()

    csv_file = 'states_normals.csv'
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        sys.exit(1)

    print("Starting exports for all states...")
    print("Note: This will start 176 export tasks (8 per state × 22 states)")
    print("Each state will have 65 precipitation bands + 60 bands per other modality (complete 2020-2024 data)")
    print("GEE has limits on concurrent tasks. Consider running in batches.\n")

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            state = row['state']
            normals = row['normals']

            print(f"Starting exports for {state}...")

            # Run the export script for this state
            cmd = f"python export_state_data.py --state '{state}' --normals '{normals}' --project '{args.project}'"
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
                print(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error exporting {state}: {e.stderr}")

            # Small delay to avoid overwhelming GEE
            time.sleep(2)

    print("\nAll export tasks initiated!")
    print("Monitor progress in Google Earth Engine Tasks panel.")
    print("Downloaded TIFFs will appear in your Google Drive 'GEE_Exports' folder.")

if __name__ == '__main__':
    main()

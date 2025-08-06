#!/usr/bin/env python3
"""
Script to filter ETestData.txt and keep only columns that begin with specified prefixes.

This script reads the ETestData.txt file and filters it to keep only columns that start with:
- LOT
- WAFER
- X
- Y
- TEST_END_DATE (note: corrected from TESET_END_DATE)
- WAFER_ID
- OPERATION
- PRODUCT
- PROGRAM
- fill in etest parms below
"""

import pandas as pd
import os

def filter_etest_data(input_file, output_file=None):
    """
    Filter ETestData.txt to keep only columns with specified prefixes.
    
    Args:
        input_file (str): Path to the input ETestData.txt file
        output_file (str): Path to the output filtered file. If None, will use 'filtered_ETestData.txt'
    """
    
    # Define the base column prefixes to keep (excluding etest parameters)
    base_prefixes = [
        'LOT',
        'WAFER', 
        'TEST_END_DATE',  # Note: corrected from TESET_END_DATE in user request
        'WAFER_ID',
        'OPERATION',
        'PRODUCT', 
        'PROGRAM'
    ]
    
    # Define coordinate columns that need exact matching (not prefix matching)
    coordinate_columns = ['X', 'Y']
    
    # Read etest parameters from config file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(script_dir, 'config_etest_parameters.txt')
    
    etest_parameters = []
    if os.path.exists(config_file):
        print(f"Reading etest parameters from: {config_file}")
        try:
            with open(config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments (lines starting with #)
                    if line and not line.startswith('#'):
                        etest_parameters.append(line)
            print(f"Found {len(etest_parameters)} etest parameters: {etest_parameters}")
        except Exception as e:
            print(f"Warning: Error reading config file {config_file}: {e}")
            print("Proceeding with base prefixes only.")
    else:
        print(f"Warning: Config file not found: {config_file}")
        print("Proceeding with base prefixes only.")
    
    # Combine base prefixes with etest parameters
    keep_prefixes = base_prefixes + etest_parameters
    
    print(f"\nUsing the following prefixes to filter columns:")
    for i, prefix in enumerate(keep_prefixes, 1):
        print(f"  {i:2d}. {prefix}")
    
    print(f"\nCoordinate columns (exact match): {coordinate_columns}")
    
    print(f"\nReading data from: {input_file}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Original data shape: {df.shape}")
        print(f"Total columns: {len(df.columns)}")
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Get column names and filter them
    all_columns = df.columns.tolist()
    
    # Find columns that start with any of the specified prefixes
    columns_to_keep = []
    prefix_matches = {}  # Track which prefix matched which columns
    
    # First, add coordinate columns with exact matching
    for col in all_columns:
        if col in coordinate_columns:
            columns_to_keep.append(col)
            if 'Coordinate Columns' not in prefix_matches:
                prefix_matches['Coordinate Columns'] = []
            prefix_matches['Coordinate Columns'].append(col)
    
    # Then, add columns that start with the specified prefixes
    for col in all_columns:
        # Skip if already added as a coordinate column
        if col in coordinate_columns:
            continue
            
        for prefix in keep_prefixes:
            if col.startswith(prefix):
                columns_to_keep.append(col)
                if prefix not in prefix_matches:
                    prefix_matches[prefix] = []
                prefix_matches[prefix].append(col)
                break  # Break to avoid adding the same column multiple times
    
    print(f"\nColumns matched by each category:")
    
    # Show coordinate columns first
    if 'Coordinate Columns' in prefix_matches:
        print(f"  Coordinate Columns (exact match): {len(prefix_matches['Coordinate Columns'])} columns")
        for col in prefix_matches['Coordinate Columns']:
            print(f"    - {col}")
    
    # Show prefix matches
    for prefix in keep_prefixes:
        if prefix in prefix_matches:
            print(f"  {prefix}: {len(prefix_matches[prefix])} columns")
            for col in prefix_matches[prefix]:
                print(f"    - {col}")
        else:
            print(f"  {prefix}: 0 columns (no matches)")
    
    print(f"\nTotal columns to keep: {len(columns_to_keep)}")
    
    # Filter the dataframe
    df_filtered = df[columns_to_keep]
    print(f"\nFiltered data shape: {df_filtered.shape}")
    
    # Clean up column names by removing [PROBE]@ETEST or @ETEST suffixes
    print(f"\nCleaning column names...")
    original_columns = df_filtered.columns.tolist()
    cleaned_columns = []
    
    for col in original_columns:
        # Remove [PROBE]@ETEST suffix first, then @ETEST suffix
        cleaned_col = col
        if cleaned_col.endswith('[PROBE]@ETEST'):
            cleaned_col = cleaned_col[:-len('[PROBE]@ETEST')]
        elif cleaned_col.endswith('@ETEST'):
            cleaned_col = cleaned_col[:-len('@ETEST')]
        cleaned_columns.append(cleaned_col)
    
    # Rename the columns
    df_filtered.columns = cleaned_columns
    
    # Show the column name changes
    print(f"Column name changes:")
    for orig, clean in zip(original_columns, cleaned_columns):
        if orig != clean:
            print(f"  '{orig}' -> '{clean}'")
    
    # Set output filename if not provided
    if output_file is None:
        input_dir = os.path.dirname(input_file)
        output_file = os.path.join(input_dir, 'filtered_ETestData.txt')
    
    # Save the filtered data
    try:
        df_filtered.to_csv(output_file, index=False)
        print(f"\nFiltered data saved to: {output_file}")
        
        # Show wafer_id count
        if 'WAFER_ID' in df_filtered.columns:
            unique_wafer_ids = df_filtered['WAFER_ID'].nunique()
            total_rows = len(df_filtered)
            print(f"\nWafer ID Analysis:")
            print(f"  - Unique WAFER_IDs: {unique_wafer_ids}")
            print(f"  - Total rows: {total_rows:,}")
            print(f"  - Average rows per wafer: {total_rows/unique_wafer_ids:.1f}")
            
            # Show the wafer IDs
            wafer_id_list = sorted(df_filtered['WAFER_ID'].unique())
            print(f"  - Wafer IDs: {wafer_id_list}")
        else:
            print(f"\nWARNING: WAFER_ID column not found in filtered data")
        
        # Show some sample data
        print(f"\nFirst 5 rows of filtered data:")
        print(df_filtered.head())
        
        # Show data types and non-null counts for each column
        print(f"\nColumn information:")
        print(df_filtered.info())
        
    except Exception as e:
        print(f"Error saving file: {e}")

def main():
    """Main function to run the filtering process."""
    
    # Define file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'ETestData.txt')
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print("Please ensure ETestData.txt is in the same directory as this script.")
        return
    
    # Run the filtering
    filter_etest_data(input_file)
    
    print("\nFiltering completed successfully!")

if __name__ == "__main__":
    main()

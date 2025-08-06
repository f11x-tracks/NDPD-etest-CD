# Combine ETestData csv files
import pandas as pd
import glob
import os
from pathlib import Path

def combine_etest_csv_files():
    """
    Combine all ETestData CSV files in the current directory into one ETestData.txt file
    """
    
    # Get the current directory
    current_dir = Path(__file__).parent
    
    # Find all ETestData CSV files
    csv_pattern = str(current_dir / "ETestData*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print("No ETestData CSV files found in the current directory.")
        return
    
    print(f"Found {len(csv_files)} ETestData CSV files:")
    for file in sorted(csv_files):
        file_size = os.path.getsize(file) / (1024*1024)  # Size in MB
        print(f"  - {os.path.basename(file)} ({file_size:.1f} MB)")
    
    # Initialize list to store dataframes
    dataframes = []
    total_rows = 0
    
    # Read each CSV file
    for i, csv_file in enumerate(sorted(csv_files)):
        print(f"\nProcessing file {i+1}/{len(csv_files)}: {os.path.basename(csv_file)}")
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            print(f"  - Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Display column names for the first file
            if i == 0:
                print(f"  - Columns: {list(df.columns)}")
            
            dataframes.append(df)
            total_rows += len(df)
            
        except Exception as e:
            print(f"  - Error reading {csv_file}: {e}")
            continue
    
    if not dataframes:
        print("No valid CSV files could be read.")
        return
    
    # Combine all dataframes
    print(f"\nCombining {len(dataframes)} dataframes...")
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"Combined dataframe: {len(combined_df)} rows, {len(combined_df.columns)} columns")
    
    # Check for duplicates
    duplicates = combined_df.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows")
        # Optionally remove duplicates
        # combined_df = combined_df.drop_duplicates()
        # print(f"After removing duplicates: {len(combined_df)} rows")
    
    # Output file path
    output_file = current_dir / "ETestData.txt"
    
    # Save to TXT file (tab-separated)
    print(f"\nSaving combined data to: {output_file}")
    combined_df.to_csv(output_file, sep='\t', index=False)
    
    # Verify the output file
    output_size = os.path.getsize(output_file) / (1024*1024)  # Size in MB
    print(f"Successfully created ETestData.txt ({output_size:.1f} MB)")
    
    # Display summary statistics
    print(f"\nSummary:")
    print(f"  - Input files: {len(csv_files)}")
    print(f"  - Total rows: {len(combined_df)}")
    print(f"  - Total columns: {len(combined_df.columns)}")
    print(f"  - Output file: ETestData.txt")
    
    # Display first few rows as verification
    print(f"\nFirst 3 rows of combined data:")
    print(combined_df.head(3).to_string())

if __name__ == "__main__":
    print("=== ETestData CSV Combiner ===")
    combine_etest_csv_files()
    print("\nDone!")
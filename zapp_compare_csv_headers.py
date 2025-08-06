# Compare column headers across ETestData CSV files
import pandas as pd
import glob
import os
from pathlib import Path

def compare_csv_headers():
    """
    Compare column headers across all ETestData CSV files to check for consistency
    """
    
    # Get the current directory
    current_dir = Path(__file__).parent
    
    # Find all ETestData CSV files
    csv_pattern = str(current_dir / "ETestData*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print("No ETestData CSV files found in the current directory.")
        return
    
    print(f"=== CSV Header Comparison ===")
    print(f"Found {len(csv_files)} ETestData CSV files:")
    for file in sorted(csv_files):
        file_size = os.path.getsize(file) / (1024*1024)  # Size in MB
        print(f"  - {os.path.basename(file)} ({file_size:.1f} MB)")
    
    # Dictionary to store headers from each file
    file_headers = {}
    all_headers_set = set()
    
    print(f"\n=== Reading Headers ===")
    
    # Read headers from each file
    for i, csv_file in enumerate(sorted(csv_files)):
        file_name = os.path.basename(csv_file)
        print(f"\nProcessing {file_name}...")
        
        try:
            # Read only the first row to get headers
            df = pd.read_csv(csv_file, nrows=0)  # nrows=0 reads only headers
            headers = list(df.columns)
            file_headers[file_name] = headers
            all_headers_set.update(headers)
            
            print(f"  ✓ {len(headers)} columns found")
            print(f"  ✓ First 5 columns: {headers[:5]}")
            print(f"  ✓ Last 5 columns: {headers[-5:]}")
            
        except Exception as e:
            print(f"  ✗ Error reading {file_name}: {e}")
            continue
    
    if not file_headers:
        print("No valid CSV files could be read.")
        return
    
    # Analysis
    print(f"\n=== Header Analysis ===")
    
    # Get all unique headers across all files
    all_headers_list = sorted(list(all_headers_set))
    print(f"Total unique columns across all files: {len(all_headers_list)}")
    
    # Check if all files have the same headers
    file_names = list(file_headers.keys())
    first_file_headers = file_headers[file_names[0]]
    
    headers_match = True
    for file_name in file_names[1:]:
        if file_headers[file_name] != first_file_headers:
            headers_match = False
            break
    
    if headers_match:
        print(f"✓ All files have IDENTICAL column headers ({len(first_file_headers)} columns)")
    else:
        print(f"✗ Files have DIFFERENT column headers")
        
        # Detailed comparison
        print(f"\n=== Detailed Comparison ===")
        
        # Show column count for each file
        print(f"Column counts by file:")
        for file_name, headers in file_headers.items():
            print(f"  {file_name}: {len(headers)} columns")
        
        # Find common headers
        common_headers = set(file_headers[file_names[0]])
        for file_name in file_names[1:]:
            common_headers = common_headers.intersection(set(file_headers[file_name]))
        
        print(f"\nCommon headers across all files: {len(common_headers)}")
        
        # Find unique headers per file
        print(f"\nUnique headers per file:")
        for file_name, headers in file_headers.items():
            other_files_headers = set()
            for other_file, other_headers in file_headers.items():
                if other_file != file_name:
                    other_files_headers.update(other_headers)
            
            unique_to_this_file = set(headers) - other_files_headers
            if unique_to_this_file:
                print(f"  {file_name}: {len(unique_to_this_file)} unique columns")
                # Show first few unique columns
                unique_list = sorted(list(unique_to_this_file))
                print(f"    Examples: {unique_list[:5]}")
            else:
                print(f"  {file_name}: 0 unique columns")
        
        # Find missing headers per file
        print(f"\nMissing headers per file (present in others but not in this file):")
        for file_name, headers in file_headers.items():
            other_files_headers = set()
            for other_file, other_headers in file_headers.items():
                if other_file != file_name:
                    other_files_headers.update(other_headers)
            
            missing_from_this_file = other_files_headers - set(headers)
            if missing_from_this_file:
                print(f"  {file_name}: {len(missing_from_this_file)} missing columns")
                # Show first few missing columns
                missing_list = sorted(list(missing_from_this_file))
                print(f"    Examples: {missing_list[:5]}")
            else:
                print(f"  {file_name}: 0 missing columns")
    
    # Check for common ETest patterns
    print(f"\n=== ETest Parameter Analysis ===")
    
    # Look for columns that end with common ETest patterns
    etest_patterns = ['[PROBE]@ETEST', '_MFW2', '_MF2W2', 'RDSON', 'VTE_', 'PCD5V_']
    
    for pattern in etest_patterns:
        matching_cols = [col for col in all_headers_list if pattern in col]
        if matching_cols:
            print(f"Columns containing '{pattern}': {len(matching_cols)}")
            print(f"  Examples: {matching_cols[:3]}")
    
    # Check for coordinate columns
    coord_cols = [col for col in all_headers_list if col in ['X', 'Y', 'WAFER_ID', 'LOT', 'PRODUCT']]
    print(f"\nCore coordinate/identification columns found: {coord_cols}")
    
    # Save detailed comparison to file
    output_file = current_dir / "csv_header_comparison.txt"
    with open(output_file, 'w') as f:
        f.write("=== CSV Header Comparison Report ===\n\n")
        f.write(f"Files analyzed: {len(csv_files)}\n")
        f.write(f"Headers match: {'Yes' if headers_match else 'No'}\n")
        f.write(f"Total unique columns: {len(all_headers_list)}\n\n")
        
        f.write("=== Column Details by File ===\n")
        for file_name, headers in file_headers.items():
            f.write(f"\n{file_name} ({len(headers)} columns):\n")
            for i, header in enumerate(headers):
                f.write(f"  {i+1:4d}. {header}\n")
        
        f.write(f"\n=== All Unique Columns ===\n")
        for i, header in enumerate(all_headers_list):
            f.write(f"  {i+1:4d}. {header}\n")
    
    print(f"\n✓ Detailed comparison saved to: {output_file}")
    
    return headers_match, file_headers

if __name__ == "__main__":
    print("Starting CSV header comparison...")
    match_result, headers_dict = compare_csv_headers()
    print(f"\nComparison complete. Headers match: {'YES' if match_result else 'NO'}")

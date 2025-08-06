# Analyze ETestData.txt for corruption and data integrity
import pandas as pd
import os
from pathlib import Path
import numpy as np

def analyze_etest_data():
    """
    Analyze ETestData.txt file for corruption and data integrity
    """
    
    # File path
    file_path = Path(__file__).parent / "ETestData.txt"
    
    if not file_path.exists():
        print(f"ERROR: {file_path} does not exist!")
        return False
    
    print("=== ETestData.txt Analysis ===")
    
    # Basic file information
    file_size = os.path.getsize(file_path) / (1024*1024)  # Size in MB
    print(f"File size: {file_size:.1f} MB")
    
    try:
        # Test reading the file
        print("\n1. Testing file readability...")
        df = pd.read_csv(file_path, sep='\t', nrows=10)  # Read first 10 rows to test format
        print(f"✓ File can be read as tab-separated")
        print(f"✓ Detected {len(df.columns)} columns")
        print(f"✓ Column names: {list(df.columns)}")
        
    except Exception as e:
        print(f"✗ ERROR reading file: {e}")
        return False
    
    try:
        # Read full file for comprehensive analysis
        print("\n2. Loading full dataset...")
        df_full = pd.read_csv(file_path, sep='\t')
        total_rows = len(df_full)
        total_cols = len(df_full.columns)
        print(f"✓ Successfully loaded {total_rows:,} rows and {total_cols} columns")
        
    except Exception as e:
        print(f"✗ ERROR loading full dataset: {e}")
        return False
    
    # Data integrity checks
    print("\n3. Data integrity checks...")
    
    # Check for completely empty rows
    empty_rows = df_full.isnull().all(axis=1).sum()
    print(f"Empty rows: {empty_rows}")
    
    # Check for duplicate rows
    duplicate_rows = df_full.duplicated().sum()
    print(f"Duplicate rows: {duplicate_rows}")
    
    # Check data types and missing values
    print(f"\n4. Column analysis:")
    for col in df_full.columns:
        null_count = df_full[col].isnull().sum()
        null_pct = (null_count / total_rows) * 100
        dtype = df_full[col].dtype
        unique_vals = df_full[col].nunique()
        
        print(f"  {col}:")
        print(f"    - Data type: {dtype}")
        print(f"    - Missing values: {null_count:,} ({null_pct:.1f}%)")
        print(f"    - Unique values: {unique_vals:,}")
        
        # Show sample values for first few columns
        if col in df_full.columns[:3]:
            sample_vals = df_full[col].dropna().head(3).tolist()
            print(f"    - Sample values: {sample_vals}")
    
    # Check for expected ETest columns
    print(f"\n5. Expected ETest columns check...")
    expected_cols = ['X', 'Y', 'LOT', 'WAFER', 'WAFER_ID', 'PRODUCT', 'PROGRAM']
    missing_expected = [col for col in expected_cols if col not in df_full.columns]
    present_expected = [col for col in expected_cols if col in df_full.columns]
    
    print(f"✓ Present expected columns: {present_expected}")
    if missing_expected:
        print(f"⚠ Missing expected columns: {missing_expected}")
    
    # Check for ETest parameter columns (should end with measurement values)
    etest_param_cols = [col for col in df_full.columns if col not in expected_cols and col not in ['TEST_END_DATE', 'OPERATION']]
    print(f"✓ ETest parameter columns found: {etest_param_cols}")
    
    # Statistical analysis of ETest parameters
    print(f"\n6. ETest parameter statistics...")
    for col in etest_param_cols:
        if df_full[col].dtype in ['float64', 'int64']:
            valid_data = df_full[col].dropna()
            if len(valid_data) > 0:
                print(f"  {col}:")
                print(f"    - Valid measurements: {len(valid_data):,}")
                print(f"    - Min: {valid_data.min():.3f}")
                print(f"    - Max: {valid_data.max():.3f}")
                print(f"    - Mean: {valid_data.mean():.3f}")
                print(f"    - Std: {valid_data.std():.3f}")
                
                # Check for outliers (values beyond 3 standard deviations)
                mean_val = valid_data.mean()
                std_val = valid_data.std()
                outliers = valid_data[(valid_data < mean_val - 3*std_val) | (valid_data > mean_val + 3*std_val)]
                print(f"    - Outliers (>3σ): {len(outliers)}")
    
    # Check coordinate system
    print(f"\n7. Coordinate system analysis...")
    if 'X' in df_full.columns and 'Y' in df_full.columns:
        x_data = df_full['X'].dropna()
        y_data = df_full['Y'].dropna()
        
        print(f"  X coordinates:")
        print(f"    - Range: {x_data.min():.1f} to {x_data.max():.1f}")
        print(f"    - Unique values: {x_data.nunique()}")
        
        print(f"  Y coordinates:")
        print(f"    - Range: {y_data.min():.1f} to {y_data.max():.1f}")
        print(f"    - Unique values: {y_data.nunique()}")
    
    # Check wafer and lot information
    print(f"\n8. Wafer and lot analysis...")
    if 'WAFER_ID' in df_full.columns:
        unique_wafers = df_full['WAFER_ID'].nunique()
        print(f"  Unique wafers: {unique_wafers}")
        
        # Show wafer distribution
        wafer_counts = df_full['WAFER_ID'].value_counts().head(10)
        print(f"  Top 10 wafers by measurement count:")
        for wafer, count in wafer_counts.items():
            print(f"    - {wafer}: {count:,} measurements")
    
    if 'LOT' in df_full.columns:
        unique_lots = df_full['LOT'].nunique()
        print(f"  Unique lots: {unique_lots}")
    
    if 'PRODUCT' in df_full.columns:
        unique_products = df_full['PRODUCT'].nunique()
        products = df_full['PRODUCT'].unique()
        print(f"  Unique products: {unique_products}")
        print(f"  Products: {list(products)}")
    
    # Final assessment
    print(f"\n9. File integrity assessment...")
    issues = []
    
    if empty_rows > 0:
        issues.append(f"{empty_rows} completely empty rows")
    
    if duplicate_rows > 0:
        issues.append(f"{duplicate_rows} duplicate rows")
    
    if missing_expected:
        issues.append(f"Missing expected columns: {missing_expected}")
    
    # Check if any ETest parameter columns have all NaN values
    all_nan_cols = []
    for col in etest_param_cols:
        if df_full[col].isnull().all():
            all_nan_cols.append(col)
    
    if all_nan_cols:
        issues.append(f"Columns with all NaN values: {all_nan_cols}")
    
    if not issues:
        print("✓ No major data integrity issues detected")
        print("✓ File appears to be valid and not corrupted")
        return True
    else:
        print("⚠ Issues detected:")
        for issue in issues:
            print(f"  - {issue}")
        print("File may need cleaning but is readable")
        return False

if __name__ == "__main__":
    print("Starting ETestData.txt analysis...")
    is_valid = analyze_etest_data()
    print(f"\nAnalysis complete. File is {'VALID' if is_valid else 'HAS ISSUES'}")

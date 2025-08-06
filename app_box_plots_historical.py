import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import io
import base64
from datetime import datetime
import sys

# Print Python version information
print("=== Python Environment Information ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("=" * 50)

# Load the data
df_cd = pd.read_csv('data/CD-data-sql.txt')
df_etest = pd.read_csv('data/filtered_ETestData.txt')

# Convert date columns to datetime immediately after loading
df_cd['ENTITY_DATA_COLLECT_DATE'] = pd.to_datetime(df_cd['ENTITY_DATA_COLLECT_DATE'], errors='coerce')
df_etest['TEST_END_DATE'] = pd.to_datetime(df_etest['TEST_END_DATE'], errors='coerce')

print(f"DEBUG: ETest data loaded and datetime converted:")
print(f"  df_etest['TEST_END_DATE'].dtype: {df_etest['TEST_END_DATE'].dtype}")
print(f"  Sample converted values: {df_etest['TEST_END_DATE'].head(3).tolist()}")

# Load wafer conditions data
try:
    print("Loading config_wafer_conditions.txt...")
    # Manually parse the file to handle space-separated format
    wafer_conditions = {}
    with open('config_wafer_conditions.txt', 'r') as f:
        lines = f.readlines()
        
        for i, line in enumerate(lines):
            if i == 0:  # Skip header
                continue
                
            # Split on whitespace (spaces and tabs) and filter out empty strings
            parts = [p for p in line.strip().split() if p]
            
            if len(parts) >= 2:
                wafer_id = parts[0]
                layer = parts[1]
                
                # If there are 3 or more parts, everything after layer is the condition
                if len(parts) >= 3:
                    condition = ' '.join(parts[2:])  # Join remaining parts with spaces
                else:
                    condition = ''
                
                if condition:  # Only add if condition exists and is not empty
                    wafer_conditions[(wafer_id, layer)] = condition
                        
    print(f"Loaded {len(wafer_conditions)} wafer-layer combinations with conditions")
            
except Exception as e:
    print(f"Warning: Could not load wafer conditions file: {e}")
    wafer_conditions = {}

# Load ETest parameters from config file
try:
    print("Loading config_etest_parameters.txt...")
    etest_parameters = []
    with open('config_etest_parameters.txt', 'r') as f:
        lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                etest_parameters.append(line)
                        
    print(f"Loaded {len(etest_parameters)} ETest parameters: {etest_parameters}")
            
except Exception as e:
    print(f"Warning: Could not load ETest parameters file: {e}")
    # Fallback to default parameters
    etest_parameters = ['RBS_MFW2', 'RBS_MF2W2']

# Separate FCCD and DCCD data
df_fccd = df_cd[df_cd['MEASUREMENT_SET_NAME'].str.contains('FCCD', na=False)].copy()
df_dccd = df_cd[df_cd['MEASUREMENT_SET_NAME'].str.contains('DCCD', na=False)].copy()

def transform_etest_coordinates(df_etest, df_cd):
    """
    Transform ETest coordinates from center-based (0,0 at wafer center) 
    to corner-based (0,0 at lower-left) to match FCCD/DCCD coordinate system.
    
    Calculate transformation separately for each product to handle different coordinate systems.
    """
    df_etest_transformed = df_etest.copy()
    
    # Get common wafers to determine coordinate transformation
    common_wafers = list(set(df_cd['WAFERID']) & set(df_etest['WAFER_ID']))
    
    if not common_wafers:
        print("  No common wafers found for coordinate transformation")
        return df_etest_transformed
    
    print("Analyzing coordinate systems using common wafers by product:")
    
    # Get common data for analysis
    all_cd_data = df_cd[df_cd['WAFERID'].isin(common_wafers)]
    all_etest_data = df_etest[df_etest['WAFER_ID'].isin(common_wafers)]
    
    if all_cd_data.empty or all_etest_data.empty:
        print("  No data found for common wafers")
        return df_etest_transformed
    
    # Group by product to calculate separate transformations
    products = all_cd_data['PRODUCT'].unique()
    
    transformations = {}
    
    for product in products:
        print(f"\n  Product: {product}")
        
        # Get wafers for this product
        product_wafers = all_cd_data[all_cd_data['PRODUCT'] == product]['WAFERID'].unique()
        
        # Filter data for this product
        cd_product_data = all_cd_data[all_cd_data['PRODUCT'] == product]
        etest_product_data = all_etest_data[all_etest_data['WAFER_ID'].isin(product_wafers)]
        
        if cd_product_data.empty or etest_product_data.empty:
            print(f"    No matching data for product {product}")
            continue
        
        # Calculate coordinate ranges for this product
        etest_x_min, etest_x_max = etest_product_data['X'].min(), etest_product_data['X'].max()
        etest_y_min, etest_y_max = etest_product_data['Y'].min(), etest_product_data['Y'].max()
        cd_x_min, cd_x_max = cd_product_data['X'].min(), cd_product_data['X'].max()
        cd_y_min, cd_y_max = cd_product_data['Y'].min(), cd_product_data['Y'].max()
        
        # Calculate transformation for this product
        x_offset = cd_x_min - etest_x_min
        y_offset = cd_y_min - etest_y_min
        
        transformations[product] = {'x_offset': x_offset, 'y_offset': y_offset}
        
        print(f"    ETest original:  X({etest_x_min:.1f} to {etest_x_max:.1f}), Y({etest_y_min:.1f} to {etest_y_max:.1f})")
        print(f"    CD target:       X({cd_x_min:.1f} to {cd_x_max:.1f}), Y({cd_y_min:.1f} to {cd_y_max:.1f})")
        print(f"    Transformation:  X+{x_offset:.1f}, Y+{y_offset:.1f}")
    
    # Apply product-specific transformations to ALL ETest data
    print(f"\nApplying product-specific transformations to {len(df_etest_transformed)} ETest measurement points:")
    
    for product, transform in transformations.items():
        # Find all wafers for this product in the ETest data
        # We need to map ETest WAFER_ID to CD PRODUCT via the common wafer mapping
        product_wafer_ids = all_cd_data[all_cd_data['PRODUCT'] == product]['WAFERID'].unique()
        
        # Apply transformation to ETest data for wafers of this product
        mask = df_etest_transformed['WAFER_ID'].isin(product_wafer_ids)
        count = mask.sum()
        
        if count > 0:
            df_etest_transformed.loc[mask, 'X'] += transform['x_offset']
            df_etest_transformed.loc[mask, 'Y'] += transform['y_offset']
            print(f"  {product}: {count} points transformed by X+{transform['x_offset']:.1f}, Y+{transform['y_offset']:.1f}")
    
    return df_etest_transformed

print(f"Loaded CD data: {len(df_cd)} rows")
print(f"FCCD data: {len(df_fccd)} rows")
print(f"DCCD data: {len(df_dccd)} rows")
print(f"Loaded ETest data: {len(df_etest)} rows")
print(f"CD data columns: {df_cd.columns.tolist()}")
print(f"ETest data columns: {df_etest.columns.tolist()}")

# Transform ETest coordinates to match CD coordinate system BEFORE filtering
print("=== Coordinate System Transformation ===")
df_etest = transform_etest_coordinates(df_etest, df_cd)
print("ETest coordinates transformed to match FCCD/DCCD coordinate system")

# Get unique LAYER and CD_SITE values for dropdowns
unique_layers = sorted(df_cd['LAYER'].unique())
layer_cd_site_map = {}
for layer in unique_layers:
    layer_cd_site_map[layer] = sorted(df_cd[df_cd['LAYER'] == layer]['CD_SITE'].unique())

print(f"Unique LAYER values: {unique_layers}")
print(f"CD_SITE by LAYER: {layer_cd_site_map}")

# Get common WAFER_IDs - matching WAFERID in CD data to WAFER_ID in ETest data
cd_wafer_ids = set(df_cd['WAFERID'].unique())
etest_wafer_ids = set(df_etest['WAFER_ID'].unique())
common_wafer_ids = list(cd_wafer_ids & etest_wafer_ids)
print(f"Found {len(common_wafer_ids)} common WAFER_IDs")

# Also include ETest-only wafers (wafers with ETest data but no CD data)
etest_only_wafer_ids = list(etest_wafer_ids - cd_wafer_ids)
print(f"Found {len(etest_only_wafer_ids)} ETest-only WAFER_IDs")

# Combine all wafers that have ETest data
all_wafer_ids = common_wafer_ids + etest_only_wafer_ids
print(f"Total wafers to analyze: {len(all_wafer_ids)}")

# Filter data to include all wafers with ETest data
df_fccd_common = df_fccd[df_fccd['WAFERID'].isin(all_wafer_ids)]
df_dccd_common = df_dccd[df_dccd['WAFERID'].isin(all_wafer_ids)]
df_etest_common = df_etest[df_etest['WAFER_ID'].isin(all_wafer_ids)]

# Group wafers by PRODUCT, LOT, and LAYER from CD data (using combined data for grouping)
df_cd_common = df_cd[df_cd['WAFERID'].isin(all_wafer_ids)]

# Create a mapping of WAFER_ID to LOT from ETest data for proper grouping
wafer_to_etest_lot = dict(zip(df_etest_common['WAFER_ID'], df_etest_common['LOT']))

# Create a mapping of WAFER_ID to PRODUCT from ETest data for ETest-only wafers
wafer_to_etest_product = dict(zip(df_etest_common['WAFER_ID'], df_etest_common['PRODUCT']))

# Debug: Show unique products in ETest data
unique_etest_products = df_etest_common['PRODUCT'].unique()
print(f"DEBUG: Unique products in ETest data: {unique_etest_products}")
print(f"DEBUG: First 6 characters of products: {[p[:6] for p in unique_etest_products]}")
print(f"DEBUG: Sample ETest-only wafer to product mapping: {dict(list(wafer_to_etest_product.items())[:5])}")

# Group wafers by PRODUCT from CD data and LAYER from CD data, but use ETest LOT
# First get all combinations from wafers that have CD data, then map to ETest LOT
cd_groups = df_cd_common.groupby(['PRODUCT', 'LAYER'])['WAFERID'].unique().to_dict()

# Create new grouping with ETest LOT values for wafers with CD data
wafer_groups = {}
for (product, layer), wafer_ids in cd_groups.items():
    # For each wafer group, get the ETest LOT for proper grouping
    for wafer_id in wafer_ids:
        if wafer_id in wafer_to_etest_lot:
            etest_lot = wafer_to_etest_lot[wafer_id]
            # Use first 6 characters of product for grouping
            product_group = product[:6] if len(product) >= 6 else product
            key = (product_group, etest_lot, layer)
            if key not in wafer_groups:
                wafer_groups[key] = []
            wafer_groups[key].append(wafer_id)

# Handle ETest-only wafers (wafers with ETest data but no CD data)
# Group them by LOT from ETest data, using actual PRODUCT from ETest data
for wafer_id in etest_only_wafer_ids:
    if wafer_id in wafer_to_etest_lot and wafer_id in wafer_to_etest_product:
        etest_lot = wafer_to_etest_lot[wafer_id]
        etest_product = wafer_to_etest_product[wafer_id]
        # Use first 6 characters of product for grouping
        product_group = etest_product[:6] if len(etest_product) >= 6 else etest_product
        # Use actual PRODUCT from ETest data and create entries for each layer
        # We'll use the unique layers from CD data to create entries for ETest-only wafers
        for layer in unique_layers:
            key = (product_group, etest_lot, layer)
            if key not in wafer_groups:
                wafer_groups[key] = []
            wafer_groups[key].append(wafer_id)

# Convert lists back to arrays for consistency
wafer_groups = {k: np.array(v) for k, v in wafer_groups.items()}

print(f"Found {len(wafer_groups)} unique PRODUCT, LOT (from RS), and LAYER combinations")

def create_box_plot(data_dict, y_col, title, y_title, layer, wafer_lot_map=None):
    """Create a box plot with wafer_id and condition on x-axis and specified measurement on y-axis"""
    if not data_dict:
        # Return empty plot if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title=title, height=436)
        return fig
    
    fig = go.Figure()
    
    # Sort wafer IDs by condition first, then by wafer_id
    # Create tuples of (condition, wafer_id) for sorting
    wafer_condition_pairs = []
    for wafer_id in data_dict.keys():
        condition = wafer_conditions.get((wafer_id, layer), 'NA')  # Set to 'NA' if no condition found
        wafer_condition_pairs.append((condition, wafer_id))
    
    # Sort by condition first, then by wafer_id
    wafer_condition_pairs.sort(key=lambda x: (x[0], x[1]))
    sorted_wafer_ids = [pair[1] for pair in wafer_condition_pairs]
    
    # Create color mapping for conditions
    unique_conditions = list(set([wafer_conditions.get((wafer_id, layer), 'NA') for wafer_id in sorted_wafer_ids]))
    unique_conditions.sort()  # Sort for consistent color assignment
    
    # Define a color palette for conditions
    color_palette = [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f',  # Gray
        '#bcbd22',  # Olive
        '#17becf',  # Cyan
        '#aec7e8',  # Light Blue
        '#ffbb78',  # Light Orange
    ]
    
    # Create condition to color mapping
    condition_colors = {}
    for i, condition in enumerate(unique_conditions):
        condition_colors[condition] = color_palette[i % len(color_palette)]
    
    # Create x-axis labels with LOT, shortened wafer ID and condition
    x_labels = []
    for wafer_id in sorted_wafer_ids:
        # Extract 3-character shortened ID (characters 6, 7, 8)
        short_id = str(wafer_id)[5:8] if len(str(wafer_id)) >= 8 else str(wafer_id)
        
        # Get lot info if wafer_lot_map is provided
        lot = wafer_lot_map.get(wafer_id, '') if wafer_lot_map else ''
        
        condition = wafer_conditions.get((wafer_id, layer), 'NA')  # Set to 'NA' if no condition found
        
        # Build the label: LOT, WAFER, Condition (always include all three)
        if lot:
            x_label = f"{lot}\n{short_id}\n{condition}"
        else:
            x_label = f"{short_id}\n{condition}"
        x_labels.append(x_label)
    
    for i, wafer_id in enumerate(sorted_wafer_ids):
        df = data_dict[wafer_id]
        if df.empty or y_col not in df.columns:
            continue
            
        # Remove NaN values
        values = df[y_col].dropna()
        if len(values) == 0:
            continue
            
        # Get condition for hover template and color
        condition = wafer_conditions.get((wafer_id, layer), 'NA')  # Set to 'NA' if no condition found
        box_color = condition_colors.get(condition, '#1f77b4')  # Default to blue if condition not found
        
        # Get X and Y coordinates for hover info
        x_coords = df['X'].values if 'X' in df.columns else ['N/A'] * len(values)
        y_coords = df['Y'].values if 'Y' in df.columns else ['N/A'] * len(values)
        
        # Create custom hover text with X, Y coordinates
        hover_text = []
        lot = wafer_lot_map.get(wafer_id, '') if wafer_lot_map else ''
        for j, (val, x_coord, y_coord) in enumerate(zip(values, x_coords, y_coords)):
            hover_text.append(f'Lot: {lot}<br>Wafer: {wafer_id}<br>Condition: {condition}<br>{y_title}: {val:.3f}<br>X: {x_coord}<br>Y: {y_coord}')
        
        # Add box plot for this wafer with condition-based color
        fig.add_trace(go.Box(
            y=values,
            name=x_labels[i],  # Use x_labels which include LOT, WAFER, Condition
            boxpoints='all',  # Show all data points instead of just outliers
            jitter=0.3,
            pointpos=-1.8,
            showlegend=False,  # Don't show individual box plots in legend
            marker=dict(
                size=3,  # Slightly smaller points when showing all
                opacity=0.6,  # More transparent when showing all points
                color=box_color  # Use condition-based color
            ),
            line=dict(width=2, color=box_color),  # Box outline color
            fillcolor=box_color,  # Box fill color (with some transparency)
            text=hover_text,
            hovertemplate='%{text}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="LOT | Wafer ID | Condition",
        yaxis_title=y_title,
        height=436,
        # Remove fixed width to allow responsive sizing
        margin=dict(l=60, r=20, t=50, b=140),  # More bottom margin for LOT/condition labels
        font=dict(size=12),
        showlegend=False,  # Don't show legend for box plots
        xaxis=dict(
            tickangle=45,  # Rotate labels for better readability
            tickmode='array',
            tickvals=list(range(len(sorted_wafer_ids))),
            ticktext=x_labels,
            tickfont=dict(size=9)  # Smaller font for multi-line text
        )
    )
    
    # Add a legend for condition colors if there are multiple conditions
    if len(unique_conditions) > 1 and unique_conditions != ['']:
        # Create invisible traces for legend
        for condition in unique_conditions:
            if condition:  # Add all conditions including 'NA'
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, color=condition_colors[condition]),
                    name=f'Condition: {condition}',
                    showlegend=True,
                    hoverinfo='skip'
                ))
        
        # Update layout to show legend
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
    
    return fig
    
    return fig

def remove_outliers_iqr(data, column_name):
    """
    Remove outliers using the Interquartile Range (IQR) method.
    Outliers are defined as values outside of Q1 - 1.5*IQR and Q3 + 1.5*IQR
    """
    if data.empty or column_name not in data.columns:
        return data
    
    initial_count = len(data)
    
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out outliers
    filtered_data = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]
    
    final_count = len(filtered_data)
    outliers_removed = initial_count - final_count
    
    if outliers_removed > 0:
        print(f"DEBUG: Removed {outliers_removed} outliers from {initial_count} data points for {column_name} (bounds: {lower_bound:.3f} to {upper_bound:.3f})")
    
    return filtered_data

# Create a function to generate plots based on filter criteria
def calculate_group_ranges(wafer_groups, df_fccd_common, df_dccd_common, df_etest_common, selected_parameter=None):
    """Calculate min/max ranges for selected ETest parameter, FCCD, and DCCD for each product/lot/layer group"""
    group_ranges = {}
    
    if selected_parameter is None:
        selected_parameter = etest_parameters[0] if etest_parameters else 'RBS_MFW2'
    
    for (product, lot, layer), wafer_ids in wafer_groups.items():
        # Skip if layer is NaN
        if pd.isna(layer):
            continue
            
        # Use the selected parameter instead of layer-based selection
        etest_column = selected_parameter
        
        # Collect all values for this group
        all_etest_values = []
        all_fccd_values = []
        all_dccd_values = []
        
        for wafer_id in wafer_ids:
            # ETest data for this wafer
            etest_data = df_etest_common[(df_etest_common['WAFER_ID'] == wafer_id) & 
                                 (df_etest_common[etest_column].notna())]
            if not etest_data.empty:
                all_etest_values.extend(etest_data[etest_column].values)
            
            # FCCD data for this wafer and layer
            fccd_data = df_fccd_common[(df_fccd_common['WAFERID'] == wafer_id) & 
                                     (df_fccd_common['LAYER'] == layer)]
            if not fccd_data.empty:
                all_fccd_values.extend(fccd_data['CD'].values)
            
            # DCCD data for this wafer and layer
            dccd_data = df_dccd_common[(df_dccd_common['WAFERID'] == wafer_id) & 
                                     (df_dccd_common['LAYER'] == layer)]
            if not dccd_data.empty:
                all_dccd_values.extend(dccd_data['CD'].values)
        
        # Calculate ranges for this group
        group_key = (product, lot, layer)
        group_ranges[group_key] = {
            'etest_min': min(all_etest_values) if all_etest_values else None,
            'etest_max': max(all_etest_values) if all_etest_values else None,
            'fccd_min': min(all_fccd_values) if all_fccd_values else None,
            'fccd_max': max(all_fccd_values) if all_fccd_values else None,
            'dccd_min': min(all_dccd_values) if all_dccd_values else None,
            'dccd_max': max(all_dccd_values) if all_dccd_values else None,
            'etest_column': etest_column
        }
    
    return group_ranges

def generate_complete_summary_stats(selected_parameter=None, etest_start_date=None, etest_end_date=None, cd_start_date=None, cd_end_date=None):
    """Generate complete summary statistics for all layers and CD sites for Excel export"""
    all_summary_data = []
    
    if selected_parameter is None:
        selected_parameter = etest_parameters[0] if etest_parameters else 'RBS_MFW2'
    
    # Apply date filtering to the data for this function
    df_etest_filtered = df_etest_common.copy()
    if etest_start_date and etest_end_date:
        df_etest_filtered = df_etest_filtered[
            (df_etest_filtered['TEST_END_DATE'] >= etest_start_date) & 
            (df_etest_filtered['TEST_END_DATE'] <= etest_end_date)
        ]
    
    df_fccd_filtered = df_fccd_common.copy()
    df_dccd_filtered = df_dccd_common.copy()
    if cd_start_date and cd_end_date:
        df_fccd_filtered = df_fccd_filtered[
            (df_fccd_filtered['ENTITY_DATA_COLLECT_DATE'] >= cd_start_date) & 
            (df_fccd_filtered['ENTITY_DATA_COLLECT_DATE'] <= cd_end_date)
        ]
        df_dccd_filtered = df_dccd_filtered[
            (df_dccd_filtered['ENTITY_DATA_COLLECT_DATE'] >= cd_start_date) & 
            (df_dccd_filtered['ENTITY_DATA_COLLECT_DATE'] <= cd_end_date)
        ]
    
    for (product, lot, layer), wafer_ids in wafer_groups.items():
        # Skip if layer is NaN
        if pd.isna(layer):
            continue
            
        # Use the selected parameter instead of layer-based selection
        etest_column = selected_parameter
        etest_title = f'{selected_parameter} (ETest)'
        
        # Get all CD sites for this layer
        layer_cd_sites = layer_cd_site_map.get(layer, [])
        
        for wafer_id in sorted(wafer_ids):
            # Filter ETest data for selected wafer with valid values in the selected ETest column
            etest_data = df_etest_filtered[(df_etest_filtered['WAFER_ID'] == wafer_id) & 
                                 (df_etest_filtered[etest_column].notna())].copy()
            
            if etest_data.empty:
                continue
            
            # Filter FCCD and DCCD data for selected wafer and layer
            fccd_data = df_fccd_filtered[(df_fccd_filtered['WAFERID'] == wafer_id) & 
                                     (df_fccd_filtered['LAYER'] == layer)]
            dccd_data = df_dccd_filtered[(df_dccd_filtered['WAFERID'] == wafer_id) & 
                                     (df_dccd_filtered['LAYER'] == layer)]
            
            # For ETest-only wafers, they won't have FCCD/DCCD data, so we still include them
            # Calculate ETest statistics
            etest_avg = etest_data[etest_column].mean()
            etest_std = etest_data[etest_column].std()
            etest_min = etest_data[etest_column].min()
            etest_max = etest_data[etest_column].max()
            etest_count = len(etest_data)
            
            # Calculate FCCD statistics (all data for the layer)
            fccd_avg = fccd_data['CD'].mean() if not fccd_data.empty else np.nan
            fccd_std = fccd_data['CD'].std() if not fccd_data.empty else np.nan
            fccd_min = fccd_data['CD'].min() if not fccd_data.empty else np.nan
            fccd_max = fccd_data['CD'].max() if not fccd_data.empty else np.nan
            fccd_count = len(fccd_data) if not fccd_data.empty else 0
            
            # Calculate DCCD statistics (all data for the layer)
            dccd_avg = dccd_data['CD'].mean() if not dccd_data.empty else np.nan
            dccd_std = dccd_data['CD'].std() if not dccd_data.empty else np.nan
            dccd_min = dccd_data['CD'].min() if not dccd_data.empty else np.nan
            dccd_max = dccd_data['CD'].max() if not dccd_data.empty else np.nan
            dccd_count = len(dccd_data) if not dccd_data.empty else 0
            
            # For each CD site, calculate scatter statistics
            for cd_site in layer_cd_sites:
                fccd_scatter_data = fccd_data[fccd_data['CD_SITE'] == cd_site] if not fccd_data.empty else pd.DataFrame()
                dccd_scatter_data = dccd_data[dccd_data['CD_SITE'] == cd_site] if not dccd_data.empty else pd.DataFrame()
                
                fccd_scatter_count = len(fccd_scatter_data)
                dccd_scatter_count = len(dccd_scatter_data)
                
                # Calculate scatter-specific statistics
                fccd_scatter_avg = fccd_scatter_data['CD'].mean() if not fccd_scatter_data.empty else np.nan
                fccd_scatter_std = fccd_scatter_data['CD'].std() if not fccd_scatter_data.empty else np.nan
                dccd_scatter_avg = dccd_scatter_data['CD'].mean() if not dccd_scatter_data.empty else np.nan
                dccd_scatter_std = dccd_scatter_data['CD'].std() if not dccd_scatter_data.empty else np.nan
                
                # Get condition for this wafer and layer
                condition = wafer_conditions.get((wafer_id, layer), '')
                
                all_summary_data.append({
                    'Product': product,
                    'Lot': lot,
                    'Layer': layer,
                    'CD_Site': cd_site,
                    'Wafer_ID': wafer_id,
                    'Condition': condition,
                    'ETest_Column': etest_column,
                    'ETest_Count': etest_count,
                    'ETest_Avg': etest_avg if not np.isnan(etest_avg) else None,
                    'ETest_StdDev': etest_std if not np.isnan(etest_std) else None,
                    'ETest_Min': etest_min if not np.isnan(etest_min) else None,
                    'ETest_Max': etest_max if not np.isnan(etest_max) else None,
                    'FCCD_Total_Count': fccd_count,
                    'FCCD_Total_Avg': fccd_avg if not np.isnan(fccd_avg) else None,
                    'FCCD_Total_StdDev': fccd_std if not np.isnan(fccd_std) else None,
                    'FCCD_Total_Min': fccd_min if not np.isnan(fccd_min) else None,
                    'FCCD_Total_Max': fccd_max if not np.isnan(fccd_max) else None,
                    'DCCD_Total_Count': dccd_count,
                    'DCCD_Total_Avg': dccd_avg if not np.isnan(dccd_avg) else None,
                    'DCCD_Total_StdDev': dccd_std if not np.isnan(dccd_std) else None,
                    'DCCD_Total_Min': dccd_min if not np.isnan(dccd_min) else None,
                    'DCCD_Total_Max': dccd_max if not np.isnan(dccd_max) else None,
                    'FCCD_Scatter_Count': fccd_scatter_count,
                    'FCCD_Scatter_Avg': fccd_scatter_avg if not np.isnan(fccd_scatter_avg) else None,
                    'FCCD_Scatter_StdDev': fccd_scatter_std if not np.isnan(fccd_scatter_std) else None,
                    'DCCD_Scatter_Count': dccd_scatter_count,
                    'DCCD_Scatter_Avg': dccd_scatter_avg if not np.isnan(dccd_scatter_avg) else None,
                    'DCCD_Scatter_StdDev': dccd_scatter_std if not np.isnan(dccd_scatter_std) else None
                })
    
    return pd.DataFrame(all_summary_data)

def generate_plots(filter_option, selected_layer=None, selected_cd_site=None, selected_parameter=None, scale_option='auto',
                   etest_start_date=None, etest_end_date=None, cd_start_date=None, cd_end_date=None, outlier_option='include'):
    """Generate box plots based on filter and display options"""
    plots = []
    plot_count = 0
    
    if selected_parameter is None:
        selected_parameter = etest_parameters[0] if etest_parameters else 'RBS_MFW2'
    
    print(f"DEBUG: Starting generate_plots with parameters:")
    print(f"  filter_option: {filter_option}")
    print(f"  selected_layer: {selected_layer}")
    print(f"  selected_cd_site: {selected_cd_site}")
    print(f"  selected_parameter: {selected_parameter}")
    print(f"  etest_start_date: {etest_start_date}")
    print(f"  etest_end_date: {etest_end_date}")
    print(f"  cd_start_date: {cd_start_date}")
    print(f"  cd_end_date: {cd_end_date}")
    
    # Apply date filtering to the data
    # Use local variables instead of modifying globals
    df_etest_filtered = df_etest_common.copy()
    print(f"DEBUG: Before date filtering - df_etest_common['TEST_END_DATE'].dtype: {df_etest_common['TEST_END_DATE'].dtype}")
    if etest_start_date and etest_end_date:
        print(f"DEBUG: Date filtering details:")
        print(f"  etest_start_date type: {type(etest_start_date)}, value: {etest_start_date}")
        print(f"  etest_end_date type: {type(etest_end_date)}, value: {etest_end_date}")
        print(f"  TEST_END_DATE column type: {df_etest_filtered['TEST_END_DATE'].dtype}")
        print(f"  Sample TEST_END_DATE values: {df_etest_filtered['TEST_END_DATE'].head(3).tolist()}")
        
        # Convert date strings to datetime if needed
        if isinstance(etest_start_date, str):
            etest_start_date = pd.to_datetime(etest_start_date)
        if isinstance(etest_end_date, str):
            etest_end_date = pd.to_datetime(etest_end_date)
            
        print(f"  After conversion - start: {etest_start_date}, end: {etest_end_date}")
        
        df_etest_filtered = df_etest_filtered[
            (df_etest_filtered['TEST_END_DATE'] >= etest_start_date) & 
            (df_etest_filtered['TEST_END_DATE'] <= etest_end_date)
        ]
    
    print(f"DEBUG: After ETest date filtering: {len(df_etest_filtered)} rows")
    
    # Filter CD data by date range  
    df_fccd_filtered = df_fccd_common.copy()
    df_dccd_filtered = df_dccd_common.copy()
    if cd_start_date and cd_end_date:
        df_fccd_filtered = df_fccd_filtered[
            (df_fccd_filtered['ENTITY_DATA_COLLECT_DATE'] >= cd_start_date) & 
            (df_fccd_filtered['ENTITY_DATA_COLLECT_DATE'] <= cd_end_date)
        ]
        df_dccd_filtered = df_dccd_filtered[
            (df_dccd_filtered['ENTITY_DATA_COLLECT_DATE'] >= cd_start_date) & 
            (df_dccd_filtered['ENTITY_DATA_COLLECT_DATE'] <= cd_end_date)
        ]
    
    print(f"DEBUG: After CD date filtering: FCCD={len(df_fccd_filtered)} rows, DCCD={len(df_dccd_filtered)} rows")
    
    # Calculate group ranges for normalized scaling
    group_ranges = calculate_group_ranges(wafer_groups, df_fccd_filtered, df_dccd_filtered, df_etest_filtered, selected_parameter)

    # Group wafers by Product (include all LOTs for each Product)
    product_groups = {}
    for (product, lot, layer), wafer_ids in wafer_groups.items():
        if product not in product_groups:
            product_groups[product] = {}
        for wafer_id in wafer_ids:
            if wafer_id in df_etest_filtered['WAFER_ID'].values:
                if wafer_id not in product_groups[product]:
                    product_groups[product][wafer_id] = lot

    print(f"DEBUG: Found {len(product_groups)} products with ETest data")
    for product, wafers in product_groups.items():
        print(f"  Product {product}: {len(wafers)} wafers")

    # Track total wafer count to limit to 100
    total_wafer_count = 0
    max_total_wafers = 100
    # Iterate through each product group
    for product, wafer_lot_map in product_groups.items():
        print(f"DEBUG: Processing product {product} with {len(wafer_lot_map)} wafers")
        
        # Use the selected parameter instead of layer-based selection
        etest_column = selected_parameter
        etest_title = f'{selected_parameter} (ETest)'
        
        # Filter and collect valid wafers for this product
        valid_wafers = []
        etest_data_dict = {}
        fccd_data_dict = {}
        dccd_data_dict = {}
        
        # Sort wafers by LOT then by wafer_id for consistent ordering
        sorted_wafers = sorted(wafer_lot_map.items(), key=lambda x: (x[1], x[0]))  # (wafer_id, lot)
        
        for wafer_id, lot in sorted_wafers:
            # Check if we've reached the total wafer limit
            if total_wafer_count >= max_total_wafers:
                break
                
            # Filter ETest data for selected wafer with valid values in the selected ETest column
            etest_data = df_etest_filtered[(df_etest_filtered['WAFER_ID'] == wafer_id) & 
                                 (df_etest_filtered[etest_column].notna())].copy()
            
            # Skip if no ETest data for this wafer
            if etest_data.empty:
                continue
            
            # Apply outlier removal if selected
            if outlier_option == 'remove':
                etest_data = remove_outliers_iqr(etest_data, etest_column)
                # Skip if no data left after outlier removal
                if etest_data.empty:
                    continue
                
            # Apply filter if selected
            if filter_option == 'filtered':
                if len(etest_data) <= 10:  # Check data points for the specific ETest measurement
                    continue
            
            # Include wafer if it has ETest data
            valid_wafers.append(wafer_id)
            total_wafer_count += 1
            
            # Store ETest data for box plots
            if not etest_data.empty:
                etest_data_dict[wafer_id] = etest_data
            
            # For CD data, filter by selected layer if available
            if selected_layer:
                # Filter FCCD data for selected wafer and layer
                fccd_data = df_fccd_filtered[(df_fccd_filtered['WAFERID'] == wafer_id) & 
                                         (df_fccd_filtered['LAYER'] == selected_layer)]
                
                # Filter DCCD data for selected wafer and layer
                dccd_data = df_dccd_filtered[(df_dccd_filtered['WAFERID'] == wafer_id) & 
                                         (df_dccd_filtered['LAYER'] == selected_layer)]
                
                # Apply additional filtering based on selected CD_SITE
                if selected_cd_site:
                    fccd_data = fccd_data[fccd_data['CD_SITE'] == selected_cd_site] if not fccd_data.empty else pd.DataFrame()
                    dccd_data = dccd_data[dccd_data['CD_SITE'] == selected_cd_site] if not dccd_data.empty else pd.DataFrame()
                
                # Store CD data for box plots
                if not fccd_data.empty:
                    fccd_data_dict[wafer_id] = fccd_data
                if not dccd_data.empty:
                    dccd_data_dict[wafer_id] = dccd_data
        
        # Skip product if no valid wafers
        if not valid_wafers:
            continue
            
        # Filter data dictionaries to match valid wafers
        etest_data_dict = {k: v for k, v in etest_data_dict.items() if k in valid_wafers}
        fccd_data_dict = {k: v for k, v in fccd_data_dict.items() if k in valid_wafers}
        dccd_data_dict = {k: v for k, v in dccd_data_dict.items() if k in valid_wafers}
        
        # Create section header for each product group
        group_header = html.Div([
            html.H2(f"Product: {product} ({len(valid_wafers)} wafers) - Conditions based on Layer: {selected_layer or 'All'}", 
                   style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 20,
                          'backgroundColor': '#e9ecef', 'padding': '10px', 'borderRadius': '5px'})
        ])
        plots.append(group_header)
        
        # Create box plots for this product FIRST
        plot_divs = []
        
        # ETest Box Plot
        if etest_data_dict:
            fig_etest = create_box_plot(etest_data_dict, etest_column, f'{etest_title} Distribution', etest_title, selected_layer, wafer_lot_map)
            plot_divs.append(
                html.Div([
                    dcc.Graph(
                        figure=fig_etest, 
                        style={'height': '436px'},
                        config={'responsive': True}  # Enable responsive resizing
                    )
                ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top'})
            )
        
        # FCCD Box Plot
        if fccd_data_dict:
            fig_fccd = create_box_plot(fccd_data_dict, 'CD', 'FCCD Distribution', 'FCCD (nm)', selected_layer, wafer_lot_map)
            plot_divs.append(
                html.Div([
                    dcc.Graph(
                        figure=fig_fccd, 
                        style={'height': '436px'},
                        config={'responsive': True}  # Enable responsive resizing
                    )
                ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
            )
        
        # DCCD Box Plot
        if dccd_data_dict:
            fig_dccd = create_box_plot(dccd_data_dict, 'CD', 'DCCD Distribution', 'DCCD (nm)', selected_layer, wafer_lot_map)
            plot_divs.append(
                html.Div([
                    dcc.Graph(
                        figure=fig_dccd, 
                        style={'height': '436px'},
                        config={'responsive': True}  # Enable responsive resizing
                    )
                ], style={'width': '32%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
            )
        
        # Add the box plots row BEFORE the summary table
        if plot_divs:
            box_plots_row = html.Div([
                html.H3(f"Box Plots - Conditions based on Layer: {selected_layer or 'All'}, CD Site: {selected_cd_site or 'All'}", 
                       style={'textAlign': 'center', 'margin': '20px 0', 'color': '#333',
                              'backgroundColor': '#e3f2fd', 'padding': '8px', 'borderRadius': '3px'}),
                html.Div(plot_divs, style={'textAlign': 'center'})
            ], style={'marginBottom': '30px', 'border': '1px solid #dee2e6', 'borderRadius': '5px', 'padding': '15px'})
            
            plots.append(box_plots_row)
        
        # Create summary table for this product AFTER the box plots
        summary_data = []
        for wafer_id in valid_wafers:
            etest_data = etest_data_dict.get(wafer_id, pd.DataFrame())
            fccd_data = fccd_data_dict.get(wafer_id, pd.DataFrame())
            dccd_data = dccd_data_dict.get(wafer_id, pd.DataFrame())
            
            # Calculate statistics for each measurement type
            etest_avg = etest_data[etest_column].mean() if not etest_data.empty else np.nan
            etest_std = etest_data[etest_column].std() if not etest_data.empty else np.nan
            
            fccd_avg = fccd_data['CD'].mean() if not fccd_data.empty else np.nan
            fccd_std = fccd_data['CD'].std() if not fccd_data.empty else np.nan
            
            dccd_avg = dccd_data['CD'].mean() if not dccd_data.empty else np.nan
            dccd_std = dccd_data['CD'].std() if not dccd_data.empty else np.nan
            
            # Get condition for this wafer and selected layer
            condition = wafer_conditions.get((wafer_id, selected_layer), '') if selected_layer else ''
            lot = wafer_lot_map.get(wafer_id, '')
            
            summary_data.append({
                'Wafer_ID': wafer_id,
                'Lot': lot,
                'Condition': condition,
                'ETest_Avg': f"{etest_avg:.3f}" if not np.isnan(etest_avg) else "N/A",
                'ETest_StdDev': f"{etest_std:.3f}" if not np.isnan(etest_std) else "N/A",
                'ETest_Count': len(etest_data) if not etest_data.empty else 0,
                'FCCD_Avg': f"{fccd_avg:.3f}" if not np.isnan(fccd_avg) else "N/A",
                'FCCD_StdDev': f"{fccd_std:.3f}" if not np.isnan(fccd_std) else "N/A",
                'FCCD_Count': len(fccd_data) if not fccd_data.empty else 0,
                'DCCD_Avg': f"{dccd_avg:.3f}" if not np.isnan(dccd_avg) else "N/A",
                'DCCD_StdDev': f"{dccd_std:.3f}" if not np.isnan(dccd_std) else "N/A",
                'DCCD_Count': len(dccd_data) if not dccd_data.empty else 0
            })
        
        # Create summary table component
        summary_table = html.Div([
            html.H3("Summary Statistics", style={'textAlign': 'center', 'marginBottom': 15, 'color': '#333'}),
            html.Table([
                html.Thead([
                    html.Tr([
                        html.Th("Wafer ID", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                        html.Th("Lot", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                        html.Th("Condition", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                        html.Th(f"{etest_title} Avg", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                        html.Th(f"{etest_title} StdDev", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                        html.Th(f"{etest_title} Count", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                        html.Th("FCCD Avg", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                        html.Th("FCCD StdDev", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                        html.Th("FCCD Count", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                        html.Th("DCCD Avg", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                        html.Th("DCCD StdDev", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'}),
                        html.Th("DCCD Count", style={'padding': '8px', 'border': '1px solid #ddd', 'backgroundColor': '#f2f2f2'})
                    ])
                ]),
                html.Tbody([
                    html.Tr([
                        html.Td(row['Wafer_ID'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'}),
                        html.Td(row['Lot'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'}),
                        html.Td(row['Condition'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center', 'fontSize': '11px'}),
                        html.Td(row['ETest_Avg'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'}),
                        html.Td(row['ETest_StdDev'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'}),
                        html.Td(row['ETest_Count'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'}),
                        html.Td(row['FCCD_Avg'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'}),
                        html.Td(row['FCCD_StdDev'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'}),
                        html.Td(row['FCCD_Count'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'}),
                        html.Td(row['DCCD_Avg'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'}),
                        html.Td(row['DCCD_StdDev'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'}),
                        html.Td(row['DCCD_Count'], style={'padding': '6px', 'border': '1px solid #ddd', 'textAlign': 'center'})
                    ]) for row in summary_data
                ])
            ], style={'width': '100%', 'borderCollapse': 'collapse', 'margin': '0 auto'})
        ], style={
            'marginBottom': '30px', 
            'padding': '15px', 
            'backgroundColor': '#f8f9fa', 
            'borderRadius': '5px',
            'border': '1px solid #dee2e6'
        })
        
        plots.append(summary_table)
        
        plot_count += len(valid_wafers)
        
        # Break out of the outer loop if we've reached the total wafer limit
        if total_wafer_count >= max_total_wafers:
            # Add a notice that we've hit the limit
            limit_notice = html.Div([
                html.H4(f"Note: Displaying limited to first {max_total_wafers} wafers. Use date filters to narrow results for specific time periods.", 
                       style={'textAlign': 'center', 'marginTop': 20, 'marginBottom': 20,
                              'backgroundColor': '#fff3cd', 'padding': '10px', 'borderRadius': '5px',
                              'border': '1px solid #ffeaa7', 'color': '#856404'})
            ])
            plots.append(limit_notice)
            break
    
    return plots, plot_count

# Initialize the Dash app
app = dash.Dash(__name__)



# Callback to initialize date pickers with data ranges
@app.callback(
    [Output('etest-date-picker', 'start_date'),
     Output('etest-date-picker', 'end_date'),
     Output('etest-date-picker', 'min_date_allowed'),
     Output('etest-date-picker', 'max_date_allowed'),
     Output('cd-date-picker', 'start_date'),
     Output('cd-date-picker', 'end_date'),
     Output('cd-date-picker', 'min_date_allowed'),
     Output('cd-date-picker', 'max_date_allowed')],
    [Input('etest-date-picker', 'id')]  # Dummy input to trigger on page load
)
def initialize_date_pickers(dummy):
    # Get date ranges from data
    etest_min = df_etest['TEST_END_DATE'].min()
    etest_max = df_etest['TEST_END_DATE'].max()
    cd_min = df_cd['ENTITY_DATA_COLLECT_DATE'].min()
    cd_max = df_cd['ENTITY_DATA_COLLECT_DATE'].max()
    
    return etest_min, etest_max, etest_min, etest_max, cd_min, cd_max, cd_min, cd_max

# Add callback for Excel export
@app.callback(
    Output("download-excel", "data"),
    [Input("export-excel-btn", "n_clicks")],
    [Input('parameter-dropdown', 'value'),
     Input('etest-date-picker', 'start_date'),
     Input('etest-date-picker', 'end_date'),
     Input('cd-date-picker', 'start_date'),
     Input('cd-date-picker', 'end_date')],
    prevent_initial_call=True
)
def export_to_excel(n_clicks, selected_parameter, etest_start_date, etest_end_date, cd_start_date, cd_end_date):
    if n_clicks:
        # Generate complete summary statistics with date filtering
        summary_df = generate_complete_summary_stats(selected_parameter, etest_start_date, etest_end_date, cd_start_date, cd_end_date)
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"MD_etest_Rs_CD_Summary_{timestamp}.xlsx"
        
        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Write main summary data
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            # Create a separate sheet with group ranges if needed
            group_ranges = calculate_group_ranges(wafer_groups, df_fccd_common, df_dccd_common, df_etest_common, selected_parameter)
            if group_ranges:
                ranges_data = []
                for (product, lot, layer), ranges in group_ranges.items():
                    ranges_data.append({
                        'Product': product,
                        'Lot': lot,
                        'Layer': layer,
                        'ETest_Column': ranges.get('etest_column'),
                        'ETest_Min': ranges.get('etest_min'),
                        'ETest_Max': ranges.get('etest_max'),
                        'FCCD_Min': ranges.get('fccd_min'),
                        'FCCD_Max': ranges.get('fccd_max'),
                        'DCCD_Min': ranges.get('dccd_min'),
                        'DCCD_Max': ranges.get('dccd_max')
                    })
                ranges_df = pd.DataFrame(ranges_data)
                ranges_df.to_excel(writer, sheet_name='Group_Ranges', index=False)
        
        output.seek(0)
        
        return dcc.send_bytes(output.read(), filename)
    
    return dash.no_update

# Add callback to update CD_SITE dropdown based on LAYER selection
@app.callback(
    Output('cd-site-dropdown', 'options'),
    Output('cd-site-dropdown', 'value'),
    [Input('layer-dropdown', 'value')]
)
def update_cd_site_options(selected_layer):
    if selected_layer is None:
        return [], None
    
    cd_sites = layer_cd_site_map.get(selected_layer, [])
    options = [{'label': site, 'value': site} for site in cd_sites]
    value = cd_sites[0] if cd_sites else None
    
    return options, value

# Add callback for dynamic plot generation
@app.callback(
    Output('plots-container', 'children'),
    [Input('filter-radio', 'value'),
     Input('outlier-radio', 'value'),
     Input('layer-dropdown', 'value'),
     Input('cd-site-dropdown', 'value'),
     Input('parameter-dropdown', 'value'),
     Input('etest-date-picker', 'start_date'),
     Input('etest-date-picker', 'end_date'),
     Input('cd-date-picker', 'start_date'),
     Input('cd-date-picker', 'end_date')]
)
def update_plots(filter_option, outlier_option, selected_layer, selected_cd_site, selected_parameter, 
                 etest_start_date, etest_end_date, cd_start_date, cd_end_date):
    plots, plot_count = generate_plots(filter_option, selected_layer, selected_cd_site, selected_parameter, 'auto',
                                     etest_start_date, etest_end_date, cd_start_date, cd_end_date, outlier_option)
    
    if not plots:
        return html.Div([
            html.H3(f"No wafers match the current filter criteria (Layer: {selected_layer})", 
                   style={'textAlign': 'center', 'marginTop': 50, 'color': 'gray'})
        ])
    
    # Add summary at the top
    summary_header = html.Div([
        html.H3(f"Displaying {plot_count} wafers for Layer: {selected_layer}, CD Site: {selected_cd_site or 'All'}, Parameter: {selected_parameter}", 
               style={'textAlign': 'center', 'marginBottom': 20, 'color': '#333'})
    ])
    
    return [summary_header] + plots

# Generate all plots at startup - this is now replaced by the callback
# print("Generating all plots...")
# all_plots = []
# plot_count = 0

# for (product, lot), wafer_ids in wafer_groups.items():
#     # Create section header for each product/lot group
#     group_header = html.Div([
#         html.H2(f"Product: {product} | Lot: {lot}", 
#                style={'textAlign': 'center', 'marginTop': 40, 'marginBottom': 20,
#                       'backgroundColor': '#e9ecef', 'padding': '10px', 'borderRadius': '5px'})
#     ])
#     all_plots.append(group_header)
#     
#     # Create plots for each wafer in this group
#     for wafer_id in sorted(wafer_ids):
#         # Filter data for selected wafer
#         rs_data = df_rs[df_rs['WAFER_ID'] == wafer_id]
#         cd_data = df_cd[df_cd['WAFER_ID'] == wafer_id]
#         
#         # Skip if no CD data for this wafer
#         if cd_data.empty:
#             continue
#         
#         plot_count += 1
#         print(f"Creating plots for wafer {plot_count}: {wafer_id}")
#             
#         # Create plots with inverse color scales to show correlation
#         fig_rs = create_contour_plot(
#             rs_data, 'X', 'Y', 'RBS_MFW2', 
#             f'RBS_MFW2 (Rs) - {wafer_id}',
#             colorscale='RdBu'  # Red to Blue (reversed for resistivity)
#         )
#         
#         fig_cd = create_contour_plot(
#             cd_data, 'X', 'Y', 'CD', 
#             f'CD (FCCD) - {wafer_id}',
#             colorscale='RdBu_r'  # Blue to Red (normal for dimension)
#         )
#         
#         # Create side-by-side layout for this wafer
#         wafer_plots = html.Div([
#             html.Div([
#                 dcc.Graph(figure=fig_rs, style={'height': '400px'})
#             ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
#             
#             html.Div([
#                 dcc.Graph(figure=fig_cd, style={'height': '400px'})
#             ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
#             
#             # Data summary for this wafer
#             html.Div([
#                 html.H4(f"Data Summary for {wafer_id}", style={'textAlign': 'center'}),
#                 html.Div([
#                     html.Div([
#                         html.P(f"RBS_MFW2 Data: {len(rs_data)} points"),
#                         html.P(f"Range: {rs_data['RBS_MFW2'].min():.3f} - {rs_data['RBS_MFW2'].max():.3f}" if not rs_data.empty else "No data"),
#                     ], style={'width': '45%', 'display': 'inline-block', 'textAlign': 'center'}),
#                     
#                     html.Div([
#                         html.P(f"CD Data: {len(cd_data)} points"),
#                         html.P(f"Range: {cd_data['CD'].min():.3f} - {cd_data['CD'].max():.3f}" if not cd_data.empty else "No data"),
#                     ], style={'width': '45%', 'display': 'inline-block', 'textAlign': 'center', 'marginLeft': '10%'})
#                 ])
#             ], style={'marginTop': 10, 'marginBottom': 20, 'padding': 10, 'backgroundColor': '#f8f9fa', 'borderRadius': 5})
#         ], style={'marginBottom': '30px', 'border': '1px solid #dee2e6', 'borderRadius': '5px', 'padding': '15px'})
#         
#         all_plots.append(wafer_plots)

# print(f"Generated {plot_count} wafer plot pairs")

# Define the layout
app.layout = html.Div([
    html.H1("Wafer Measurement Box Plot Analysis", 
            style={'textAlign': 'center', 'marginBottom': 30}),
    
    # Date filters
    html.Div([
        html.Div([
            html.Label("ETest Date Range (TEST_END_DATE):", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
            dcc.DatePickerRange(
                id='etest-date-picker',
                start_date=None,  # Will be set by callback
                end_date=None,    # Will be set by callback
                display_format='YYYY-MM-DD',
                style={'marginBottom': '10px'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Label("CD Date Range (ENTITY_DATA_COLLECT_DATE):", style={'fontWeight': 'bold', 'marginBottom': '5px', 'display': 'block'}),
            dcc.DatePickerRange(
                id='cd-date-picker',
                start_date=None,  # Will be set by callback
                end_date=None,    # Will be set by callback
                display_format='YYYY-MM-DD',
                style={'marginBottom': '10px'}
            )
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'})
    ], style={'textAlign': 'left', 'marginBottom': 20, 'padding': '15px', 'backgroundColor': '#e9ecef', 'borderRadius': '5px'}),

    html.Div([
        html.Div([
            html.P("Box Plot Analysis: Shows distribution of selected ETest parameters, FCCD, and DCCD measurements across wafers", 
                   style={'fontWeight': 'bold', 'margin': '5px 0'}),
            html.P("Each box shows median, quartiles, and outliers for each wafer. Compare distributions between wafers to identify trends.", 
                   style={'fontStyle': 'italic', 'fontSize': 14, 'margin': '5px 0'}),
            html.P("Select any ETest parameter from the dropdown to analyze. Filter by Layer and CD Site to focus analysis.", 
                   style={'fontStyle': 'italic', 'fontSize': 12, 'margin': '5px 0', 'color': '#0066cc'}),
            html.P("Wafer conditions are shown in the summary table to correlate with measurement distributions.", 
                   style={'fontStyle': 'italic', 'fontSize': 12, 'margin': '5px 0', 'color': '#666'})
        ], style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px', 'marginTop': '10px'})
    ], style={'textAlign': 'center', 'marginBottom': 20}),
    
    # Control options
    html.Div([
        html.Div([
            html.Label("Filter Options:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
            dcc.RadioItems(
                id='filter-radio',
                options=[
                    {'label': 'Show all wafers', 'value': 'all'},
                    {'label': 'Show only wafers with >10 ETest data points', 'value': 'filtered'}
                ],
                value='all',
                style={'marginBottom': '15px'}
            )
        ], style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([
            html.Label("Outlier Removal:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
            dcc.RadioItems(
                id='outlier-radio',
                options=[
                    {'label': 'Include outliers', 'value': 'include'},
                    {'label': 'Remove outliers', 'value': 'remove'}
                ],
                value='include',
                style={'marginBottom': '15px'}
            )
        ], style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '1%'}),
        
        html.Div([
            html.Label("ETest Parameter:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
            dcc.Dropdown(
                id='parameter-dropdown',
                options=[{'label': param, 'value': param} for param in etest_parameters],
                value=etest_parameters[0] if etest_parameters else None,
                style={'marginBottom': '15px'}
            )
        ], style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '1%'}),
        
        html.Div([
            html.Label("Export Options:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
            html.Button(
                "Export Summary to Excel",
                id="export-excel-btn",
                style={
                    'backgroundColor': '#28a745',
                    'color': 'white',
                    'border': 'none',
                    'padding': '8px 16px',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'fontSize': '12px',
                    'fontWeight': 'bold'
                }
            ),
            dcc.Download(id="download-excel")
        ], style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '1%'}),
        
        html.Div([
            html.Label("Analysis Layer:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
            dcc.Dropdown(
                id='layer-dropdown',
                options=[{'label': layer, 'value': layer} for layer in unique_layers],
                value=unique_layers[0] if unique_layers else None,
                style={'marginBottom': '15px'}
            )
        ], style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '1%'}),
        
        html.Div([
            html.Label("CD Site Filter:", style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
            dcc.Dropdown(
                id='cd-site-dropdown',
                options=[],
                value=None,
                style={'marginBottom': '15px'}
            )
        ], style={'width': '15%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '1%'})
    ], style={'textAlign': 'left', 'marginBottom': 20, 'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
    
    # Display plots based on selections
    html.Div(id='plots-container', style={'textAlign': 'center'})
])

if __name__ == '__main__':
    app.run(debug=True, port=8053)
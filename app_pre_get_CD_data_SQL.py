import PyUber
import pandas as pd
import numpy as np
import dash
from dash import Dash, dcc, html, State, callback
from dash import dash_table as dt
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


SQL_DATA = '''
SELECT  
        a1.entity AS entity
        ,To_Char(a1.data_collection_time,'yyyy-mm-dd hh24:mi:ss') AS entity_data_collect_date
        ,a0.operation AS spc_operation
        ,a0.lot AS lot
        ,(SELECT lrc99.last_pass FROM F_LOT_RUN_CARD lrc99 where lrc99.lot =a0.lot AND lrc99.operation = a0.operation AND lrc99.site_prevout_date=a0.prev_moveout_time and rownum<=1) AS last_pass
        ,a4.wafer AS waferID
        ,a4.wafer3 AS raw_wafer3
        ,a4.parameter_name AS CD_NAME
        ,a4.value AS CD
        ,a3.measurement_set_name AS measurement_set_name
        ,a3.valid_flag as valid_flag
        ,a3.standard_flag as standard_flag
        ,a4.native_x_col AS X
        ,a4.native_y_row AS Y
        ,a0.route AS route
        ,a0.product AS product
        ,a0.process_operation AS OPN
FROM 
P_SPC_LOT a0
LEFT JOIN P_SPC_ENTITY a1 ON a1.spcs_id = a0.spcs_id AND a1.entity_sequence=1
INNER JOIN P_SPC_SESSION a2 ON a2.spcs_id = a0.spcs_id AND a2.data_collection_time = a0.data_collection_time
INNER JOIN P_SPC_MEASUREMENT_SET a3 ON a3.spcs_id = a2.spcs_id
LEFT JOIN P_SPC_MEASUREMENT a4 ON a4.spcs_id = a3.spcs_id AND a4.measurement_set_name = a3.measurement_set_name
WHERE
a1.data_collection_time >= TRUNC(SYSDATE) - 70
AND (a3.measurement_set_name = 'CD.FCCD_MEASUREMENTS.5051' or a3.measurement_set_name = 'CD.DCCD_MEASUREMENTS.5051')
AND (a4.parameter_name like '%NDT%'
or a4.parameter_name like '%PDT%')
AND a0.lot LIKE 'W435985%'
'''

try:
    conn = PyUber.connect(datasource='F21_PROD_XEUS')
    df = pd.read_sql(SQL_DATA, conn)
    
    # Parse CD_NAME by splitting on semicolons
    def parse_cd_name(cd_name):
        if pd.isna(cd_name):
            return None, None
        
        parts = str(cd_name).split(';')
        
        # LAYER is between first and second ';' (index 1)
        layer = parts[1] if len(parts) > 1 else None
        
        # CD_SITE is at index 2
        cd_site = parts[2] if len(parts) > 2 else None
        
        return layer, cd_site
    
    # Apply the parsing function and create new columns
    df[['LAYER', 'CD_SITE']] = df['CD_NAME'].apply(
        lambda x: pd.Series(parse_cd_name(x))
    )
    
    # Convert ENTITY_DATA_COLLECT_DATE to datetime for proper sorting
    df['ENTITY_DATA_COLLECT_DATE'] = pd.to_datetime(df['ENTITY_DATA_COLLECT_DATE'])
    
    # Check for duplicates based on LAYER, WAFERID, CD_NAME, X, Y
    print(f'Before filtering for duplicates: {len(df)} rows')
    duplicate_cols = ['LAYER', 'WAFERID', 'CD_NAME', 'X', 'Y']
    
    # Check if there are any duplicates
    duplicates = df.duplicated(subset=duplicate_cols, keep=False)
    if duplicates.any():
        print(f'Found {duplicates.sum()} duplicate rows based on {duplicate_cols}')
        print('Sample duplicates (showing dates):')
        sample_duplicates = df[duplicates][duplicate_cols + ['CD', 'ENTITY_DATA_COLLECT_DATE']].head(10)
        print(sample_duplicates.sort_values(['WAFERID', 'LAYER', 'X', 'Y', 'ENTITY_DATA_COLLECT_DATE']))
        
        # For each duplicate group, keep only the most recent one based on ENTITY_DATA_COLLECT_DATE
        print('\nRemoving duplicates - keeping most recent data by ENTITY_DATA_COLLECT_DATE for each spatial location...')
        
        # Sort by duplicate columns and date, then keep the last (most recent) for each duplicate group
        df_filtered = df.sort_values(duplicate_cols + ['ENTITY_DATA_COLLECT_DATE']).drop_duplicates(
            subset=duplicate_cols, keep='last'
        )
        
        # Show what was removed
        removed_count = len(df) - len(df_filtered)
        print(f'Removed {removed_count} older duplicate measurements')
        
        # Show sample of kept vs removed for verification
        if removed_count > 0:
            duplicate_groups = df[duplicates].groupby(duplicate_cols)
            sample_group = list(duplicate_groups)[0]  # Get first group
            group_key, group_data = sample_group
            if len(group_data) > 1:
                print(f'\nExample duplicate group at location {group_key[:2]} X={group_key[3]} Y={group_key[4]}:')
                group_sorted = group_data.sort_values('ENTITY_DATA_COLLECT_DATE')
                print('  All measurements for this location:')
                for _, row in group_sorted.iterrows():
                    print(f'    Date: {row["ENTITY_DATA_COLLECT_DATE"]}, CD: {row["CD"]:.3f}')
                print(f'  → Kept: {group_sorted.iloc[-1]["ENTITY_DATA_COLLECT_DATE"]} (most recent)')
    else:
        print('No duplicates found')
        df_filtered = df.copy()
        
    print(f'After filtering for duplicates: {len(df_filtered)} rows')
    
    # Sort the data by ENTITY_DATA_COLLECT_DATE and LOT before converting back to string
    df_filtered = df_filtered.sort_values(['ENTITY_DATA_COLLECT_DATE', 'LOT'])
    
    # Convert datetime back to string for output consistency
    df_filtered['ENTITY_DATA_COLLECT_DATE'] = df_filtered['ENTITY_DATA_COLLECT_DATE'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Show the distribution of parsed values
    print(f'LAYER value counts:')
    print(df_filtered['LAYER'].value_counts(dropna=False))
    print(f'\nCD_SITE value counts:')
    print(df_filtered['CD_SITE'].value_counts(dropna=False))
    
    # Show a sample of the parsed data
    print(f'\nSample of parsed data:')
    sample_cols = ['CD_NAME', 'LAYER', 'CD_SITE', 'ENTITY_DATA_COLLECT_DATE', 'LOT']
    print(df_filtered[sample_cols].head(10))
    
    # Output the dataframe to a text file
    output_file = 'CD-data-sql.txt'
    df_filtered.to_csv(output_file, sep=',', index=False)
    print(f'\nData successfully saved to {output_file}')
    print(f'Data shape: {df_filtered.shape}')
    print(f'Columns: {list(df_filtered.columns)}')
    
    # Show summary of date filtering
    print(f'\nDate filtering summary:')
    date_summary = df_filtered.groupby(['WAFERID', 'LAYER'])['ENTITY_DATA_COLLECT_DATE'].first().reset_index()
    print(f'Unique WAFERID/LAYER combinations: {len(date_summary)}')
    print(f'Date range: {df_filtered["ENTITY_DATA_COLLECT_DATE"].min()} to {df_filtered["ENTITY_DATA_COLLECT_DATE"].max()}')
except Exception as e:
    print(f'Cannot run SQL script - Consider connecting to VPN. Error: {e}')


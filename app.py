import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.interpolate import griddata

# Load the data
df_rs = pd.read_csv('data/MD_Rs.csv')
df_cd = pd.read_csv('data/MD_FCCD.csv')

# Get common WAFER_IDs
common_wafer_ids = list(set(df_rs['WAFER_ID'].unique()) & set(df_cd['WAFER_ID'].unique()))
print(f"Found {len(common_wafer_ids)} common WAFER_IDs")

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Wafer Contour Plot Comparison", 
            style={'textAlign': 'center', 'marginBottom': 30}),
    
    html.Div([
        html.Label("Select Wafer ID:", style={'fontSize': 16, 'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='wafer-dropdown',
            options=[{'label': wafer_id, 'value': wafer_id} for wafer_id in sorted(common_wafer_ids)],
            value=sorted(common_wafer_ids)[0] if common_wafer_ids else None,
            style={'width': '400px', 'margin': '10px 0'}
        ),
        html.Div([
            html.P("Color Scale: Red = High CD (large FCCD) / Low RBS_MFW2 (low Rs)", 
                   style={'color': 'red', 'fontWeight': 'bold', 'margin': '5px 0'}),
            html.P("Blue = Low CD (small FCCD) / High RBS_MFW2 (high Rs)", 
                   style={'color': 'blue', 'fontWeight': 'bold', 'margin': '5px 0'}),
            html.P("Correlated areas should show similar colors", 
                   style={'fontStyle': 'italic', 'fontSize': 14, 'margin': '5px 0'})
        ], style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px', 'marginTop': '10px'})
    ], style={'textAlign': 'center', 'marginBottom': 20}),
    
    html.Div(id='contour-plots', style={'textAlign': 'center'})
])

def create_contour_plot(df, x_col, y_col, z_col, title, colorscale='Viridis'):
    """Create a contour plot from scattered data"""
    if df.empty:
        # Return empty plot if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title=title, height=400)
        return fig
    
    # Get the data points
    x = df[x_col].values
    y = df[y_col].values
    z = df[z_col].values
    
    # Remove any NaN values
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[mask], y[mask], z[mask]
    
    if len(x) == 0:
        # Return empty plot if no valid data
        fig = go.Figure()
        fig.add_annotation(
            text="No valid data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(title=title, height=400)
        return fig
    
    # Create a grid for interpolation
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Add some padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.1
    x_max += x_range * 0.1
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    
    # Create grid
    grid_x, grid_y = np.mgrid[x_min:x_max:50j, y_min:y_max:50j]
    
    # Interpolate data to grid
    try:
        grid_z = griddata((x, y), z, (grid_x, grid_y), method='linear')
        
        # Create contour plot
        fig = go.Figure()
        
        # Add contour
        fig.add_trace(go.Contour(
            x=grid_x[:, 0],
            y=grid_y[0, :],
            z=grid_z.T,  # Transpose the grid to fix orientation
            colorscale=colorscale,
            showscale=True,
            line=dict(width=0.5),
            contours=dict(coloring='fill'),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Z: %{z:.3f}<extra></extra>'
        ))
        
        # Add scatter points to show actual data locations
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(
                size=4,
                color='white',
                line=dict(width=1, color='black')
            ),
            name='Data Points',
            hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{customdata:.3f}<extra></extra>',
            customdata=z
        ))
        
    except Exception as e:
        # If interpolation fails, create scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(
                size=8,
                color=z,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title="Value")
            ),
            hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{marker.color:.3f}<extra></extra>'
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="X",
        yaxis_title="Y",
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

@app.callback(
    Output('contour-plots', 'children'),
    Input('wafer-dropdown', 'value')
)
def update_plots(selected_wafer_id):
    if not selected_wafer_id:
        return html.Div("Please select a wafer ID")
    
    # Filter data for selected wafer
    rs_data = df_rs[df_rs['WAFER_ID'] == selected_wafer_id]
    cd_data = df_cd[df_cd['WAFER_ID'] == selected_wafer_id]
    
    # Create plots with inverse color scales to show correlation
    # Higher RBS_MFW2 (Rs) = blue, Lower RBS_MFW2 = red
    # Higher CD (FCCD) = red, Lower CD = blue
    # This way, correlated areas (small CD = high Rs) will have similar colors
    fig_rs = create_contour_plot(
        rs_data, 'X', 'Y', 'RBS_MFW2', 
        f'RBS_MFW2 (Rs) - {selected_wafer_id}',
        colorscale='RdBu'  # Red to Blue (reversed for resistivity)
    )
    
    fig_cd = create_contour_plot(
        cd_data, 'X', 'Y', 'CD', 
        f'CD (FCCD) - {selected_wafer_id}',
        colorscale='RdBu_r'  # Blue to Red (normal for dimension)
    )
    
    # Create side-by-side layout
    return html.Div([
        html.Div([
            dcc.Graph(figure=fig_rs)
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        html.Div([
            dcc.Graph(figure=fig_cd)
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '4%'}),
        
        html.Div([
            html.H3(f"Data Summary for {selected_wafer_id}", style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    html.H4("RBS_MFW2 Data"),
                    html.P(f"Data points: {len(rs_data)}"),
                    html.P(f"RBS_MFW2 range: {rs_data['RBS_MFW2'].min():.3f} - {rs_data['RBS_MFW2'].max():.3f}" if not rs_data.empty else "No data"),
                    html.P(f"X range: {rs_data['X'].min()} - {rs_data['X'].max()}" if not rs_data.empty else "No data"),
                    html.P(f"Y range: {rs_data['Y'].min()} - {rs_data['Y'].max()}" if not rs_data.empty else "No data")
                ], style={'width': '45%', 'display': 'inline-block', 'textAlign': 'center'}),
                
                html.Div([
                    html.H4("CD Data"),
                    html.P(f"Data points: {len(cd_data)}"),
                    html.P(f"CD range: {cd_data['CD'].min():.3f} - {cd_data['CD'].max():.3f}" if not cd_data.empty else "No data"),
                    html.P(f"X range: {cd_data['X'].min()} - {cd_data['X'].max()}" if not cd_data.empty else "No data"),
                    html.P(f"Y range: {cd_data['Y'].min()} - {cd_data['Y'].max()}" if not cd_data.empty else "No data")
                ], style={'width': '45%', 'display': 'inline-block', 'textAlign': 'center', 'marginLeft': '10%'})
            ])
        ], style={'marginTop': 30, 'padding': 20, 'backgroundColor': '#f8f9fa', 'borderRadius': 5})
    ])

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
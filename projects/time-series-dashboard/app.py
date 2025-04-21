import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime, timedelta
from ts_forecaster import TimeSeriesForecaster
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Time Series Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper Functions
def get_sample_datasets():
    """Return a dictionary of sample datasets"""
    return {
        "Retail Sales": "https://raw.githubusercontent.com/stevenelliottjr/time-series-dashboard/main/data/retail_sales.csv",
        "Web Traffic": "https://raw.githubusercontent.com/stevenelliottjr/time-series-dashboard/main/data/web_traffic.csv",
        "Energy Consumption": "https://raw.githubusercontent.com/stevenelliottjr/time-series-dashboard/main/data/energy_consumption.csv"
    }

def create_download_link(df, filename):
    """Generate a download link for a DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

@st.cache_data
def load_dataset(dataset_name):
    """Load a sample dataset from the repository"""
    datasets = get_sample_datasets()
    if dataset_name in datasets:
        df = pd.read_csv(datasets[dataset_name])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    return None

def plot_forecast_plotly(forecaster, model_name, include_history=True):
    """Create a Plotly forecast plot with confidence intervals"""
    fig = go.Figure()
    
    # Plot historical data if requested
    if include_history:
        fig.add_trace(go.Scatter(
            x=forecaster.df[forecaster.date_column],
            y=forecaster.df[forecaster.target_column],
            mode='lines+markers',
            name='Historical Data',
            line=dict(color='black', width=1.5),
            marker=dict(size=4)
        ))
    
    # Get the forecast
    forecast = forecaster.forecasts[model_name]
    forecast_col = f'{forecaster.target_column}_{model_name}'
    
    # Plot forecast
    fig.add_trace(go.Scatter(
        x=forecast[forecaster.date_column],
        y=forecast[forecast_col],
        mode='lines',
        name=f'{model_name.capitalize()} Forecast',
        line=dict(color='royalblue', width=2.5)
    ))
    
    # Add confidence intervals if available
    lower_col = f'{forecast_col}_lower'
    upper_col = f'{forecast_col}_upper'
    
    if lower_col in forecast.columns and upper_col in forecast.columns:
        fig.add_trace(go.Scatter(
            x=forecast[forecaster.date_column],
            y=forecast[upper_col],
            mode='lines',
            name='Upper Bound (95%)',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast[forecaster.date_column],
            y=forecast[lower_col],
            mode='lines',
            name='Lower Bound (95%)',
            line=dict(width=0),
            fillcolor='rgba(65, 105, 225, 0.2)',
            fill='tonexty',
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title=f'{model_name.capitalize()} Forecast',
        xaxis_title='Date',
        yaxis_title=forecaster.target_column,
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)'),
        height=500
    )
    
    return fig

def plot_models_comparison_plotly(forecaster, models):
    """Create a Plotly plot comparing multiple models"""
    fig = go.Figure()
    
    # Plot historical data
    fig.add_trace(go.Scatter(
        x=forecaster.df[forecaster.date_column],
        y=forecaster.df[forecaster.target_column],
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='black', width=1.5),
        marker=dict(size=4)
    ))
    
    # Colors for different models
    colors = ['royalblue', 'green', 'red', 'purple', 'orange']
    
    # Plot each model's forecast
    for i, model_name in enumerate(models):
        if model_name not in forecaster.forecasts:
            continue
            
        forecast = forecaster.forecasts[model_name]
        forecast_col = f'{forecaster.target_column}_{model_name}'
        
        fig.add_trace(go.Scatter(
            x=forecast[forecaster.date_column],
            y=forecast[forecast_col],
            mode='lines',
            name=f'{model_name.capitalize()} Forecast',
            line=dict(color=colors[i % len(colors)], width=2)
        ))
    
    # Update layout
    fig.update_layout(
        title='Model Comparison',
        xaxis_title='Date',
        yaxis_title=forecaster.target_column,
        hovermode='x unified',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)'),
        height=500
    )
    
    return fig

def plot_metrics_plotly(metrics):
    """Create a Plotly bar chart of model metrics"""
    models = list(metrics.keys())
    metrics_df = pd.DataFrame({
        'Model': models,
        'MAPE (%)': [metrics[m]['mape'] for m in models],
        'RMSE': [metrics[m]['rmse'] for m in models],
        'MAE': [metrics[m]['mae'] for m in models]
    })
    
    # Create a separate plot for each metric
    fig1 = px.bar(
        metrics_df, x='Model', y='MAPE (%)', 
        title='Mean Absolute Percentage Error (lower is better)',
        color='Model', height=400
    )
    
    fig2 = px.bar(
        metrics_df, x='Model', y='RMSE', 
        title='Root Mean Square Error (lower is better)',
        color='Model', height=400
    )
    
    fig3 = px.bar(
        metrics_df, x='Model', y='MAE', 
        title='Mean Absolute Error (lower is better)',
        color='Model', height=400
    )
    
    return fig1, fig2, fig3

def generate_prophet_components_plotly(forecaster):
    """Create Plotly visualizations of Prophet components"""
    if 'prophet' not in forecaster.forecasts:
        st.warning("Prophet model forecast not available")
        return None, None, None
    
    forecast = forecaster.forecasts['prophet']
    target = forecaster.target_column
    date_col = forecaster.date_column
    
    # Check if components are available
    required_cols = ['trend', 'yearly', 'weekly']
    if not all(col in forecast.columns for col in required_cols):
        st.warning("Component data not available. Rerun forecast with return_components=True")
        return None, None, None
    
    # Trend component
    trend_fig = px.line(
        forecast, x=date_col, y='trend', 
        title='Trend Component',
        labels={'trend': f'{target} (Trend)', date_col: 'Date'},
        height=350
    )
    
    # Yearly seasonality
    yearly_data = forecast.copy()
    yearly_data['ds'] = pd.to_datetime(yearly_data[date_col])
    yearly_data['day_of_year'] = yearly_data['ds'].dt.dayofyear
    yearly_avg = yearly_data.groupby('day_of_year')['yearly'].mean().reset_index()
    
    yearly_fig = px.line(
        yearly_avg, x='day_of_year', y='yearly',
        title='Yearly Seasonality',
        labels={'yearly': f'{target} (Yearly Effect)', 'day_of_year': 'Day of Year'},
        height=350
    )
    
    # Weekly seasonality
    weekly_data = forecast.copy()
    weekly_data['ds'] = pd.to_datetime(weekly_data[date_col])
    weekly_data['day_of_week'] = weekly_data['ds'].dt.dayofweek
    weekly_avg = weekly_data.groupby('day_of_week')['weekly'].mean().reset_index()
    weekly_avg['day_name'] = weekly_avg['day_of_week'].map({
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
        3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    })
    
    weekly_fig = px.line(
        weekly_avg, x='day_name', y='weekly',
        title='Weekly Seasonality',
        labels={'weekly': f'{target} (Weekly Effect)', 'day_name': 'Day of Week'},
        height=350
    )
    
    return trend_fig, yearly_fig, weekly_fig

def main():
    st.sidebar.image("https://raw.githubusercontent.com/stevenelliottjr/time-series-dashboard/main/images/logo.png", width=200)
    st.sidebar.title("⚙️ Dashboard Controls")
    
    # Main sections of the app
    app_mode = st.sidebar.radio(
        "Select Section",
        ["📊 Dashboard", "📈 Forecasting", "🔍 Analysis", "ℹ️ About"]
    )
    
    # Initialize session state
    if 'forecaster' not in st.session_state:
        st.session_state.forecaster = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'models_fitted' not in st.session_state:
        st.session_state.models_fitted = False
    if 'forecasts_generated' not in st.session_state:
        st.session_state.forecasts_generated = False
    
    # Dashboard Section
    if app_mode == "📊 Dashboard":
        st.title("📊 Time Series Forecasting Dashboard")
        
        st.markdown("""
        Welcome to the Time Series Forecasting Dashboard! This interactive tool allows you to:
        
        * Upload your own time series data or use sample datasets
        * Fit multiple forecasting models (Prophet, ARIMA, ETS)
        * Generate and visualize forecasts with confidence intervals
        * Analyze seasonality and trend components
        * Compare model performance with evaluation metrics
        
        Let's get started by selecting or uploading data in the sidebar.
        """)
        
        # Data Selection
        st.sidebar.header("Data Selection")
        data_source = st.sidebar.radio(
            "Choose data source",
            ["Sample Datasets", "Upload CSV"]
        )
        
        if data_source == "Sample Datasets":
            dataset_name = st.sidebar.selectbox(
                "Select a dataset",
                list(get_sample_datasets().keys())
            )
            
            if st.sidebar.button("Load Dataset"):
                with st.spinner(f"Loading {dataset_name} dataset..."):
                    data = load_dataset(dataset_name)
                    st.session_state.data = data
                    st.session_state.forecaster = None
                    st.session_state.models_fitted = False
                    st.session_state.forecasts_generated = False
        
        else:  # Upload CSV
            uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.session_state.data = data
                    st.session_state.forecaster = None
                    st.session_state.models_fitted = False
                    st.session_state.forecasts_generated = False
                except Exception as e:
                    st.sidebar.error(f"Error loading file: {str(e)}")
        
        # Display the data if available
        if st.session_state.data is not None:
            st.subheader("Dataset Preview")
            st.write(st.session_state.data.head())
            
            # Data information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Observations", len(st.session_state.data))
            with col2:
                if 'date' in st.session_state.data.columns:
                    date_range = f"{st.session_state.data['date'].min().strftime('%Y-%m-%d')} to {st.session_state.data['date'].max().strftime('%Y-%m-%d')}"
                    st.metric("Date Range", date_range)
            with col3:
                if 'value' in st.session_state.data.columns:
                    st.metric("Average Value", f"{st.session_state.data['value'].mean():.2f}")
            
            # Basic time series plot
            if 'date' in st.session_state.data.columns and 'value' in st.session_state.data.columns:
                st.subheader("Time Series Visualization")
                
                fig = px.line(
                    st.session_state.data, x='date', y='value',
                    title="Time Series Data",
                    labels={'value': 'Value', 'date': 'Date'},
                    height=400
                )
                
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Value",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Configure forecasting
            st.sidebar.header("Configure Forecasting")
            
            date_col = st.sidebar.selectbox(
                "Select date column",
                st.session_state.data.columns,
                index=list(st.session_state.data.columns).index('date') if 'date' in st.session_state.data.columns else 0
            )
            
            target_col = st.sidebar.selectbox(
                "Select target column",
                st.session_state.data.columns,
                index=list(st.session_state.data.columns).index('value') if 'value' in st.session_state.data.columns else 0
            )
            
            # Models selection
            st.sidebar.header("Models")
            use_prophet = st.sidebar.checkbox("Prophet", value=True)
            use_arima = st.sidebar.checkbox("ARIMA", value=True)
            use_ets = st.sidebar.checkbox("ETS", value=True)
            
            selected_models = []
            if use_prophet:
                selected_models.append("prophet")
            if use_arima:
                selected_models.append("arima")
            if use_ets:
                selected_models.append("ets")
            
            if len(selected_models) == 0:
                st.sidebar.warning("Please select at least one model!")
            
            # Forecast horizon
            forecast_horizon = st.sidebar.slider(
                "Forecast Horizon (days)",
                min_value=7,
                max_value=365,
                value=30,
                step=1
            )
            
            # Train/test split
            train_ratio = st.sidebar.slider(
                "Training Data Percentage",
                min_value=50,
                max_value=95,
                value=80,
                step=5
            ) / 100
            
            # Run forecasting button
            if st.sidebar.button("Run Forecast") and len(selected_models) > 0:
                with st.spinner("Fitting models and generating forecasts..."):
                    try:
                        # Initialize forecaster
                        forecaster = TimeSeriesForecaster(
                            st.session_state.data, 
                            date_column=date_col, 
                            target_column=target_col
                        )
                        
                        # Fit selected models
                        forecaster.fit(models=selected_models, train_ratio=train_ratio)
                        st.session_state.models_fitted = True
                        
                        # Generate forecasts
                        forecaster.predict(horizon=forecast_horizon, return_components=True)
                        st.session_state.forecasts_generated = True
                        
                        # Evaluate models
                        metrics = forecaster.evaluate()
                        
                        # Store forecaster in session state
                        st.session_state.forecaster = forecaster
                        
                        st.success("Forecasting completed successfully!")
                    except Exception as e:
                        st.error(f"Error during forecasting: {str(e)}")
            
            # Quick results if forecasts are available
            if st.session_state.forecasts_generated and st.session_state.forecaster is not None:
                st.subheader("Forecast Results")
                
                # Show the best model's forecast
                best_model = st.session_state.forecaster.best_model
                if best_model:
                    st.write(f"**Best Model:** {best_model.capitalize()} (based on MAPE)")
                    
                    forecast_fig = plot_forecast_plotly(
                        st.session_state.forecaster, 
                        best_model, 
                        include_history=True
                    )
                    
                    st.plotly_chart(forecast_fig, use_container_width=True)
                    
                    st.info("For detailed analysis and model comparison, go to the Forecasting and Analysis sections!")
                
        else:
            st.info("Please select a sample dataset or upload your own CSV file to get started.")
    
    # Forecasting Section
    elif app_mode == "📈 Forecasting":
        st.title("📈 Detailed Forecasting")
        
        if not st.session_state.forecasts_generated or st.session_state.forecaster is None:
            st.warning("No forecasts available! Please go to the Dashboard section to load data and run a forecast first.")
            return
        
        # Get the forecaster
        forecaster = st.session_state.forecaster
        
        # Forecast visualizations
        st.header("Forecast Visualizations")
        
        # Model selection
        model_names = list(forecaster.forecasts.keys())
        selected_model = st.selectbox(
            "Select forecast model to display",
            model_names,
            index=model_names.index(forecaster.best_model) if forecaster.best_model in model_names else 0
        )
        
        # Display the selected model's forecast
        forecast_fig = plot_forecast_plotly(forecaster, selected_model, include_history=True)
        st.plotly_chart(forecast_fig, use_container_width=True)
        
        # Model comparison
        st.header("Model Comparison")
        
        comparison_fig = plot_models_comparison_plotly(forecaster, model_names)
        st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Forecast details in table format
        st.header("Forecast Details")
        
        forecast_days = st.slider(
            "Number of days to display",
            min_value=7,
            max_value=min(90, len(forecaster.forecasts[selected_model])),
            value=30
        )
        
        # Combine all forecasts into one DataFrame
        all_forecasts = pd.DataFrame({
            'Date': forecaster.forecasts[model_names[0]][forecaster.date_column]
        })
        
        for model in model_names:
            forecast_col = f'{forecaster.target_column}_{model}'
            all_forecasts[f'{model.capitalize()} Forecast'] = forecaster.forecasts[model][forecast_col]
        
        # Show forecast table
        st.dataframe(all_forecasts.tail(forecast_days).reset_index(drop=True), height=400)
        
        # Download forecast data
        st.download_button(
            label="Download Forecast Data",
            data=all_forecasts.to_csv(index=False).encode('utf-8'),
            file_name='time_series_forecast.csv',
            mime='text/csv'
        )
    
    # Analysis Section
    elif app_mode == "🔍 Analysis":
        st.title("🔍 Forecast Analysis")
        
        if not st.session_state.forecasts_generated or st.session_state.forecaster is None:
            st.warning("No forecasts available! Please go to the Dashboard section to load data and run a forecast first.")
            return
        
        # Get the forecaster
        forecaster = st.session_state.forecaster
        
        # Metrics comparison
        st.header("Model Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        metrics_figs = plot_metrics_plotly(forecaster.metrics)
        
        with col1:
            st.plotly_chart(metrics_figs[0], use_container_width=True)
        
        with col2:
            st.plotly_chart(metrics_figs[1], use_container_width=True)
            
        with col3:
            st.plotly_chart(metrics_figs[2], use_container_width=True)
        
        # Metrics table
        metrics_df = pd.DataFrame({
            'Model': list(forecaster.metrics.keys()),
            'MAPE (%)': [forecaster.metrics[m]['mape'] for m in forecaster.metrics.keys()],
            'RMSE': [forecaster.metrics[m]['rmse'] for m in forecaster.metrics.keys()],
            'MAE': [forecaster.metrics[m]['mae'] for m in forecaster.metrics.keys()]
        })
        
        st.dataframe(metrics_df, height=200)
        
        # Seasonality components (Prophet only)
        st.header("Seasonality Analysis (Prophet)")
        
        if 'prophet' in forecaster.forecasts:
            trend_fig, yearly_fig, weekly_fig = generate_prophet_components_plotly(forecaster)
            
            if trend_fig is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(trend_fig, use_container_width=True)
                    st.plotly_chart(weekly_fig, use_container_width=True)
                    
                with col2:
                    st.plotly_chart(yearly_fig, use_container_width=True)
        else:
            st.info("Prophet model not available. Run the forecast with Prophet model to see seasonality decomposition.")
    
    # About Section
    else:
        st.title("ℹ️ About This Dashboard")
        
        st.markdown("""
        ## Time Series Forecasting Dashboard
        
        This interactive tool provides comprehensive time series forecasting capabilities using multiple models:
        
        * **Prophet**: Facebook's forecasting tool that handles seasonality and holidays
        * **ARIMA**: Statistical method for time series forecasting
        * **ETS**: Exponential smoothing state space model
        
        ### Features
        
        * Upload your own data or use sample datasets
        * Configure and compare multiple forecasting models
        * Visualize forecasts with confidence intervals
        * Analyze seasonality and trend components
        * Evaluate model performance with accuracy metrics
        * Export results for further analysis
        
        ### Usage Guide
        
        1. Start in the **Dashboard** section to load data and run initial forecasts
        2. Visit the **Forecasting** section for detailed visualizations and comparisons
        3. Use the **Analysis** section to evaluate model performance and examine seasonality
        
        ### About the Developer
        
        This dashboard was developed by [Steven Elliott Jr.](https://www.linkedin.com/in/steven-elliott-jr), a data scientist and machine learning engineer specializing in time series analysis and forecasting.
        
        ### Resources
        
        * [GitHub Repository](https://github.com/stevenelliottjr/time-series-dashboard)
        * [Documentation](https://github.com/stevenelliottjr/time-series-dashboard/docs)
        * [Personal Website](https://stevenelliottjr.github.io)
        
        ### License
        
        This project is licensed under the MIT License.
        """)

if __name__ == "__main__":
    main()
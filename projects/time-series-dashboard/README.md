# Time Series Forecasting Dashboard

An interactive Streamlit dashboard for time series forecasting using multiple machine learning models including Prophet, ARIMA, and ETS.

## Features

- 📊 **Multiple Forecasting Models**: Prophet, ARIMA, and Exponential Smoothing (ETS)
- 📈 **Interactive Visualizations**: Plotly-based charts with confidence intervals
- 🔍 **Model Comparison**: Side-by-side comparison of different forecasting approaches
- 📉 **Seasonality Analysis**: Decomposition of trend, yearly, and weekly patterns
- 📋 **Sample Datasets**: Pre-loaded datasets for quick exploration
- 💾 **CSV Upload**: Bring your own time series data
- 📥 **Export Results**: Download forecasts as CSV files

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone this repository or navigate to the project directory:
```bash
cd projects/time-series-dashboard
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Dashboard

Start the Streamlit app:
```bash
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

### Using Sample Data

The dashboard includes three pre-configured sample datasets:
- **Retail Sales**: Daily sales data showing seasonal trends
- **Web Traffic**: Website visitor counts with weekly patterns
- **Energy Consumption**: Energy usage data with daily and seasonal variations

### Uploading Your Own Data

1. Prepare a CSV file with at least two columns:
   - A date column (any standard date format)
   - A numeric value column

2. In the sidebar, select "Upload CSV" and choose your file

3. Configure the forecasting parameters:
   - Select which models to use (Prophet, ARIMA, ETS)
   - Set the forecast horizon (days to predict)
   - Adjust the train/test split ratio

4. Click "Run Forecast" to generate predictions

### Programmatic API

```python
from ts_forecaster import TimeSeriesForecaster

# Initialize with data
forecaster = TimeSeriesForecaster(df, date_column="date", target_column="value")

# Fit multiple models
forecaster.fit(models=["prophet", "arima", "ets"])

# Generate forecasts
forecast_df = forecaster.predict(horizon=30, return_components=True)

# Evaluate models
metrics = forecaster.evaluate()
best_model = forecaster.get_best_model(metric="mape")
```

## Dashboard Sections

### 📊 Dashboard
Main view for loading data and running forecasts. Shows:
- Data preview and statistics
- Quick forecast visualization
- Configuration options

### 📈 Forecasting
Detailed forecast visualizations including:
- Individual model forecasts with confidence intervals
- Model comparison charts
- Forecast data table
- Download options

### 🔍 Analysis
Model performance evaluation:
- Accuracy metrics (MAPE, RMSE, MAE)
- Metric comparison charts
- Seasonality decomposition (Prophet only)
- Trend and pattern analysis

### ℹ️ About
Information about the dashboard, models, and usage guide

## Models

### Prophet
- Facebook's forecasting tool
- Handles seasonality and holidays automatically
- Best for: Business metrics with strong seasonal patterns

### ARIMA
- Statistical time series method
- Auto-regressive integrated moving average
- Best for: Short-term forecasts with clear trends

### ETS
- Exponential smoothing state space model
- Simple and robust
- Best for: Stable time series with consistent patterns

## Performance Metrics

The dashboard evaluates models using:
- **MAPE** (Mean Absolute Percentage Error): Overall accuracy percentage
- **RMSE** (Root Mean Square Error): Penalty for large errors
- **MAE** (Mean Absolute Error): Average prediction error

## Project Structure

```
time-series-dashboard/
├── app.py                 # Main Streamlit application
├── ts_forecaster.py       # Forecasting model wrapper class
├── requirements.txt       # Python dependencies
├── data/                  # Sample datasets
│   ├── retail_sales.csv
│   ├── web_traffic.csv
│   └── energy_consumption.csv
├── images/                # Screenshots and logos
└── README.md             # This file
```

## Dependencies

Main libraries used:
- `streamlit`: Web application framework
- `prophet`: Facebook's forecasting library
- `statsmodels`: ARIMA and ETS models
- `plotly`: Interactive visualizations
- `pandas`, `numpy`: Data manipulation
- `scikit-learn`: Evaluation metrics

## Troubleshooting

### Prophet Installation Issues

If you encounter issues installing Prophet, try:
```bash
# On macOS
brew install cmake
pip install prophet

# On Windows
conda install -c conda-forge prophet

# On Linux
pip install pystan==2.19.1.1
pip install prophet
```

### Memory Issues with Large Datasets

For datasets with >10,000 rows:
- Reduce the training data percentage
- Use fewer models simultaneously
- Consider aggregating data to weekly/monthly intervals

## Demo

A live demo is available at:
[Time Series Forecasting Dashboard](https://time-series-dashboard.streamlit.app)

## Author

**Steven Elliott Jr.**
- Portfolio: [stevenelliottjr.github.io](https://stevenelliottjr.github.io)
- LinkedIn: [linkedin.com/in/steven-elliott-jr](https://www.linkedin.com/in/steven-elliott-jr)
- GitHub: [@stevenelliottjr](https://github.com/stevenelliottjr)

## License

MIT License - feel free to use this project for learning and development purposes.

## Acknowledgments

- Facebook Prophet team for the excellent forecasting library
- Statsmodels team for ARIMA implementation
- Streamlit team for the amazing app framework
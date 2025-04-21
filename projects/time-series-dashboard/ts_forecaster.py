import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
import logging
from datetime import datetime, timedelta

class TimeSeriesForecaster:
    """
    A unified interface for time series forecasting with multiple models
    and evaluation capabilities.
    """
    
    def __init__(self, df=None, date_column=None, target_column=None):
        """
        Initialize the forecaster with optional data.
        
        Args:
            df: Pandas DataFrame containing the time series data
            date_column: Name of the date column
            target_column: Name of the target variable column
        """
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        self.best_model = None
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger('ts_forecaster')
        
        # Set data if provided
        if df is not None and date_column is not None and target_column is not None:
            self.set_data(df, date_column, target_column)
    
    def set_data(self, df, date_column, target_column):
        """
        Set or update the time series data.
        
        Args:
            df: Pandas DataFrame containing the time series data
            date_column: Name of the date column
            target_column: Name of the target variable column
        """
        # Validate inputs
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        
        if date_column not in df.columns:
            raise ValueError(f"date_column '{date_column}' not found in DataFrame")
            
        if target_column not in df.columns:
            raise ValueError(f"target_column '{target_column}' not found in DataFrame")
        
        # Store original data
        self.df = df.copy()
        self.date_column = date_column
        self.target_column = target_column
        
        # Ensure date column is datetime type
        self.df[date_column] = pd.to_datetime(self.df[date_column])
        
        # Sort by date
        self.df = self.df.sort_values(date_column).reset_index(drop=True)
        
        # Create Prophet-compatible dataframe
        self.prophet_df = self.df[[date_column, target_column]].rename(
            columns={date_column: 'ds', target_column: 'y'}
        )
        
        self.logger.info(f"Data set with {len(df)} observations from "
                        f"{self.df[date_column].min()} to {self.df[date_column].max()}")
    
    def fit(self, models=None, train_ratio=0.8, **kwargs):
        """
        Fit multiple forecasting models to the data.
        
        Args:
            models: List of model names to fit ['prophet', 'arima', 'ets']
            train_ratio: Ratio of data to use for training (rest for validation)
            **kwargs: Additional model-specific parameters
        
        Returns:
            self: The fitted forecaster instance
        """
        if self.df is None:
            raise ValueError("No data set. Call set_data() first.")
        
        if models is None:
            models = ['prophet']  # Default to Prophet only
        
        # Split data for training and validation
        train_size = int(len(self.df) * train_ratio)
        train_df = self.df.iloc[:train_size].copy()
        valid_df = self.df.iloc[train_size:].copy()
        
        self.train_df = train_df
        self.valid_df = valid_df
        
        # Fit each requested model
        for model_name in models:
            self.logger.info(f"Fitting {model_name} model...")
            
            if model_name.lower() == 'prophet':
                self._fit_prophet(**kwargs)
            elif model_name.lower() == 'arima':
                self._fit_arima(**kwargs)
            elif model_name.lower() == 'ets':
                self._fit_ets(**kwargs)
            else:
                self.logger.warning(f"Unknown model: {model_name}")
        
        return self
    
    def _fit_prophet(self, changepoint_prior_scale=0.05, seasonality_mode='additive', 
                    yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                    holidays=None, **kwargs):
        """Fit Facebook Prophet model"""
        train_prophet = self.prophet_df.iloc[:len(self.train_df)].copy()
        
        # Initialize and fit Prophet model
        model = Prophet(
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        
        # Add holidays if provided
        if holidays is not None:
            model.add_country_holidays(country_name=holidays)
        
        # Add additional regressors if in the original dataframe
        for col in self.df.columns:
            if col not in [self.date_column, self.target_column] and col in kwargs.get('regressors', []):
                self.logger.info(f"Adding regressor: {col}")
                model.add_regressor(col)
                train_prophet[col] = self.df.iloc[:len(self.train_df)][col].values
        
        # Suppress convergence warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train_prophet)
        
        self.models['prophet'] = model
        self.logger.info("Prophet model fitted successfully")
    
    def _fit_arima(self, order=(5,1,0), seasonal_order=(1,1,0,12), trend='c', **kwargs):
        """Fit ARIMA or SARIMA model"""
        # Prepare data for ARIMA (just the target variable)
        train_series = self.train_df[self.target_column]
        
        # Determine if we should use seasonal ARIMA
        if max(seasonal_order) > 0:
            use_seasonal = True
        else:
            use_seasonal = False
        
        # Initialize and fit ARIMA model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                model = ARIMA(
                    train_series,
                    order=order,
                    seasonal_order=seasonal_order if use_seasonal else None,
                    trend=trend
                )
                
                fitted_model = model.fit()
                self.models['arima'] = fitted_model
                self.logger.info("ARIMA model fitted successfully")
                
            except Exception as e:
                self.logger.error(f"Error fitting ARIMA model: {str(e)}")
                self.models['arima'] = None
    
    def _fit_ets(self, trend=None, damped_trend=False, seasonal=None, 
                seasonal_periods=12, **kwargs):
        """Fit ETS (Exponential Smoothing) model"""
        # Prepare data for ETS (just the target variable)
        train_series = self.train_df[self.target_column]
        
        # Initialize and fit ETS model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                model = ExponentialSmoothing(
                    train_series,
                    trend=trend,
                    damped_trend=damped_trend,
                    seasonal=seasonal,
                    seasonal_periods=seasonal_periods if seasonal else None
                )
                
                fitted_model = model.fit()
                self.models['ets'] = fitted_model
                self.logger.info("ETS model fitted successfully")
                
            except Exception as e:
                self.logger.error(f"Error fitting ETS model: {str(e)}")
                self.models['ets'] = None
    
    def predict(self, horizon=30, model_name=None, return_components=False, 
                prediction_interval=0.95, **kwargs):
        """
        Generate forecasts using the fitted models.
        
        Args:
            horizon: Number of periods to forecast
            model_name: Specific model to use (if None, use all fitted models)
            return_components: Whether to return seasonality components (Prophet only)
            prediction_interval: Width of prediction intervals
            **kwargs: Additional model-specific parameters
        
        Returns:
            DataFrame containing forecasts
        """
        if not self.models:
            raise ValueError("No fitted models available. Call fit() first.")
        
        forecast_dfs = {}
        
        # Determine which models to use for prediction
        models_to_use = [model_name] if model_name else self.models.keys()
        
        for name in models_to_use:
            if name not in self.models or self.models[name] is None:
                self.logger.warning(f"Model {name} not available for prediction")
                continue
                
            self.logger.info(f"Generating {horizon} step forecast with {name} model")
            
            if name == 'prophet':
                forecast = self._predict_prophet(horizon, return_components, prediction_interval)
            elif name == 'arima':
                forecast = self._predict_arima(horizon, prediction_interval)
            elif name == 'ets':
                forecast = self._predict_ets(horizon, prediction_interval)
            else:
                continue
                
            forecast_dfs[name] = forecast
            self.forecasts[name] = forecast
        
        if not forecast_dfs:
            raise ValueError("No forecasts generated")
            
        # If only one model requested, return its forecast directly
        if model_name and model_name in forecast_dfs:
            return forecast_dfs[model_name]
            
        return forecast_dfs
    
    def _predict_prophet(self, horizon, return_components, prediction_interval):
        """Generate forecast with Prophet model"""
        model = self.models['prophet']
        
        # Create future dataframe
        future = model.make_future_dataframe(periods=horizon, freq='D')
        
        # Add additional regressors if they were used in fitting
        for regressor in model.extra_regressors:
            if 'future_' + regressor['name'] in self.df.columns:
                future[regressor['name']] = self.df['future_' + regressor['name']]
        
        # Make prediction
        forecast = model.predict(future)
        
        # Add prediction intervals
        if prediction_interval:
            interval = (1 - prediction_interval) / 2
            forecast['yhat_lower'] = forecast['yhat'].values - 1.96 * forecast['yhat'].values.std()
            forecast['yhat_upper'] = forecast['yhat'].values + 1.96 * forecast['yhat'].values.std()
            
        # Include only the requested components
        if not return_components:
            keep_cols = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
            forecast = forecast[keep_cols]
        
        # Restore original column names
        forecast = forecast.rename(columns={
            'ds': self.date_column,
            'yhat': f'{self.target_column}_prophet',
            'yhat_lower': f'{self.target_column}_prophet_lower',
            'yhat_upper': f'{self.target_column}_prophet_upper'
        })
        
        return forecast
    
    def _predict_arima(self, horizon, prediction_interval):
        """Generate forecast with ARIMA model"""
        model = self.models['arima']
        
        # Generate forecast
        forecast_result = model.get_forecast(steps=horizon)
        
        # Create forecast dataframe
        last_date = self.df[self.date_column].max()
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon)
        
        forecast = pd.DataFrame({
            self.date_column: forecast_dates,
            f'{self.target_column}_arima': forecast_result.predicted_mean
        })
        
        # Add prediction intervals
        if prediction_interval:
            conf_int = forecast_result.conf_int(alpha=(1-prediction_interval))
            forecast[f'{self.target_column}_arima_lower'] = conf_int.iloc[:, 0].values
            forecast[f'{self.target_column}_arima_upper'] = conf_int.iloc[:, 1].values
        
        return forecast
    
    def _predict_ets(self, horizon, prediction_interval):
        """Generate forecast with ETS model"""
        model = self.models['ets']
        
        # Generate forecast
        forecast_result = model.forecast(horizon)
        
        # Create forecast dataframe
        last_date = self.df[self.date_column].max()
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon)
        
        forecast = pd.DataFrame({
            self.date_column: forecast_dates,
            f'{self.target_column}_ets': forecast_result.values
        })
        
        # Add simple prediction intervals (ETS doesn't provide them directly)
        if prediction_interval:
            # Estimate prediction std dev from historical fit
            residuals_std = np.std(model.resid)
            z_score = 1.96  # ~95% interval
            
            forecast[f'{self.target_column}_ets_lower'] = forecast[f'{self.target_column}_ets'] - z_score * residuals_std
            forecast[f'{self.target_column}_ets_upper'] = forecast[f'{self.target_column}_ets'] + z_score * residuals_std
        
        return forecast
    
    def evaluate(self, metric='mape'):
        """
        Evaluate models on the validation set.
        
        Args:
            metric: Metric to use for evaluation ('mape', 'rmse', 'mae')
            
        Returns:
            Dictionary of evaluation metrics for each model
        """
        if not self.models:
            raise ValueError("No fitted models available. Call fit() first.")
            
        if not hasattr(self, 'valid_df') or self.valid_df is None:
            raise ValueError("No validation data available.")
        
        results = {}
        valid_y = self.valid_df[self.target_column].values
        
        # Generate in-sample predictions for the validation period
        horizon = len(self.valid_df)
        forecasts = self.predict(horizon=horizon)
        
        # Compute metrics for each model
        for model_name, forecast_df in forecasts.items():
            # Extract predictions corresponding to validation period
            if model_name == 'prophet':
                preds = forecast_df[f'{self.target_column}_{model_name}'].iloc[-horizon:].values
            else:
                # For other models, we need to match dates
                merged = pd.merge(
                    self.valid_df[[self.date_column]],
                    forecast_df,
                    on=self.date_column,
                    how='left'
                )
                preds = merged[f'{self.target_column}_{model_name}'].values
            
            # Calculate metrics
            mape = mean_absolute_percentage_error(valid_y, preds) * 100
            rmse = np.sqrt(mean_squared_error(valid_y, preds))
            mae = mean_absolute_error(valid_y, preds)
            
            results[model_name] = {
                'mape': mape,
                'rmse': rmse,
                'mae': mae
            }
            
            self.logger.info(f"{model_name} - MAPE: {mape:.2f}%, RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        
        # Store metrics and determine best model
        self.metrics = results
        self.best_model = self.get_best_model(metric)
        
        return results
    
    def get_best_model(self, metric='mape'):
        """
        Determine the best performing model based on the given metric.
        
        Args:
            metric: Metric to use for comparison ('mape', 'rmse', 'mae')
            
        Returns:
            Name of the best performing model
        """
        if not self.metrics:
            if hasattr(self, 'valid_df') and self.valid_df is not None:
                self.evaluate(metric)
            else:
                return list(self.models.keys())[0] if self.models else None
        
        # Find model with best metric
        if metric == 'mape' or metric == 'rmse' or metric == 'mae':
            best_model = min(self.metrics.items(), key=lambda x: x[1][metric])[0]
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return best_model
    
    def plot_forecast(self, model_name=None, figsize=(12, 6), include_history=True, 
                     ax=None, title=None, **kwargs):
        """
        Plot the forecast with confidence intervals.
        
        Args:
            model_name: Which model's forecast to plot (if None, use best model)
            figsize: Figure size as (width, height) tuple
            include_history: Whether to include historical data in the plot
            ax: Matplotlib axis to plot on
            title: Plot title
            **kwargs: Additional keyword arguments for plotting
            
        Returns:
            Matplotlib figure and axes
        """
        if not self.forecasts:
            raise ValueError("No forecasts available. Call predict() first.")
        
        # Determine which model to plot
        if model_name is None:
            if self.best_model is not None:
                model_name = self.best_model
            else:
                model_name = list(self.forecasts.keys())[0]
        
        if model_name not in self.forecasts:
            raise ValueError(f"No forecast available for model: {model_name}")
        
        # Get the forecast
        forecast = self.forecasts[model_name]
        
        # Create figure if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # Plot historical data if requested
        if include_history:
            ax.plot(
                self.df[self.date_column], 
                self.df[self.target_column],
                'k.-', 
                alpha=0.7, 
                label='Historical Data'
            )
        
        # Plot forecasted values
        forecast_col = f'{self.target_column}_{model_name}'
        ax.plot(
            forecast[self.date_column],
            forecast[forecast_col],
            'b-', 
            linewidth=2,
            label=f'{model_name.capitalize()} Forecast'
        )
        
        # Plot confidence intervals if available
        lower_col = f'{forecast_col}_lower'
        upper_col = f'{forecast_col}_upper'
        
        if lower_col in forecast.columns and upper_col in forecast.columns:
            ax.fill_between(
                forecast[self.date_column],
                forecast[lower_col],
                forecast[upper_col],
                color='b', 
                alpha=0.2,
                label='95% Confidence Interval'
            )
        
        # Add title and labels
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'{model_name.capitalize()} Forecast', fontsize=14)
            
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(self.target_column, fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        return fig, ax
    
    def plot_components(self, model_name='prophet', figsize=(12, 10)):
        """
        Plot the decomposition of the forecast into trend and seasonality components.
        Currently only supported for Prophet model.
        
        Args:
            model_name: Model to use (only 'prophet' supported currently)
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure
        """
        if model_name != 'prophet':
            raise ValueError("Component plots currently only supported for Prophet model")
            
        if 'prophet' not in self.models:
            raise ValueError("Prophet model not fitted. Call fit(['prophet']) first.")
        
        model = self.models['prophet']
        
        # Generate forecast with components if not already done
        if 'prophet' not in self.forecasts or 'trend' not in self.forecasts['prophet'].columns:
            self.predict(model_name='prophet', return_components=True)
        
        # Use Prophet's built-in component plotting
        fig = model.plot_components(self.forecasts['prophet'])
        fig.set_size_inches(figsize)
        
        return fig
    
    def plot_models_comparison(self, figsize=(12, 6), models=None):
        """
        Plot forecasts from multiple models for comparison.
        
        Args:
            figsize: Figure size as (width, height) tuple
            models: List of model names to include (if None, use all available)
            
        Returns:
            Matplotlib figure and axes
        """
        if not self.forecasts:
            raise ValueError("No forecasts available. Call predict() first.")
        
        # Determine which models to plot
        if models is None:
            models = list(self.forecasts.keys())
        else:
            models = [m for m in models if m in self.forecasts]
            
        if not models:
            raise ValueError("No specified models have forecasts available")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot historical data
        ax.plot(
            self.df[self.date_column], 
            self.df[self.target_column],
            'k.-', 
            alpha=0.7, 
            label='Historical Data'
        )
        
        # Plot each model's forecast
        colors = ['b', 'g', 'r', 'c', 'm']
        for i, model_name in enumerate(models):
            forecast = self.forecasts[model_name]
            forecast_col = f'{self.target_column}_{model_name}'
            
            ax.plot(
                forecast[self.date_column],
                forecast[forecast_col],
                f'{colors[i % len(colors)]}-', 
                linewidth=2,
                label=f'{model_name.capitalize()} Forecast'
            )
        
        # Add title and labels
        ax.set_title('Model Comparison', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(self.target_column, fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        return fig, ax
    
    def plot_metrics(self, figsize=(10, 6)):
        """
        Plot comparison of evaluation metrics across models.
        
        Args:
            figsize: Figure size as (width, height) tuple
            
        Returns:
            Matplotlib figure and axes
        """
        if not self.metrics:
            raise ValueError("No evaluation metrics available. Call evaluate() first.")
        
        # Convert metrics to DataFrame for easier plotting
        metrics_df = pd.DataFrame({
            model: {
                'MAPE (%)': metrics['mape'],
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae']
            }
            for model, metrics in self.metrics.items()
        }).T
        
        # Create figure with three subplots (one per metric)
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot each metric
        metrics_df['MAPE (%)'].plot(kind='bar', ax=axes[0], color='skyblue')
        metrics_df['RMSE'].plot(kind='bar', ax=axes[1], color='salmon')
        metrics_df['MAE'].plot(kind='bar', ax=axes[2], color='lightgreen')
        
        # Customize subplots
        for i, metric in enumerate(['MAPE (%)', 'RMSE', 'MAE']):
            axes[i].set_title(metric)
            axes[i].grid(axis='y', alpha=0.3)
            axes[i].set_ylabel(metric)
            
            # Add value labels on top of bars
            for j, v in enumerate(metrics_df[metric]):
                axes[i].text(j, v + (metrics_df[metric].max() * 0.03), 
                           f'{v:.2f}', ha='center')
        
        fig.suptitle('Model Performance Comparison', fontsize=14)
        fig.tight_layout()
        
        return fig, axes
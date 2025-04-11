"""
Module for prediction analysis of Radio Base Stations (RBS) deployment.
Contains functions to predict future trends in RBS deployment based on historical data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime, timedelta

# Import from other modules
from src.advanced_temporal_analysis import preprocess_temporal_data

def preprocess_prediction_data(gdf_rbs):
    """
    Preprocesses the RBS data for time series prediction.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        
    Returns:
        DataFrame: Preprocessed data for prediction analysis
    """
    # Start with temporal preprocessing
    df = preprocess_temporal_data(gdf_rbs)
    
    # Ensure we have installation dates
    if 'installation_date' not in df.columns:
        print("Error: No installation dates available for prediction.")
        return None
    
    # Create time series of installations by day/month/year
    df_time = df.copy()
    
    # Create counts by different time periods
    daily_counts = df_time.groupby(df_time['installation_date'].dt.date).size()
    weekly_counts = df_time.groupby(pd.Grouper(key='installation_date', freq='W')).size()
    monthly_counts = df_time.groupby(pd.Grouper(key='installation_date', freq='M')).size()
    quarterly_counts = df_time.groupby(pd.Grouper(key='installation_date', freq='Q')).size()
    yearly_counts = df_time.groupby(pd.Grouper(key='installation_date', freq='Y')).size()
    
    # Convert to DataFrames for easier handling
    daily_df = daily_counts.reset_index()
    daily_df.columns = ['date', 'installations']
    
    weekly_df = weekly_counts.reset_index()
    weekly_df.columns = ['date', 'installations']
    
    monthly_df = monthly_counts.reset_index()
    monthly_df.columns = ['date', 'installations']
    
    quarterly_df = quarterly_counts.reset_index()
    quarterly_df.columns = ['date', 'installations']
    
    yearly_df = yearly_counts.reset_index()
    yearly_df.columns = ['date', 'installations']
    
    # Return all time series data
    return {
        'daily': daily_df,
        'weekly': weekly_df,
        'monthly': monthly_df,
        'quarterly': quarterly_df,
        'yearly': yearly_df,
        'original': df_time
    }

def test_stationarity(time_series, output_path=None):
    """
    Tests for stationarity in a time series using the Augmented Dickey-Fuller test.
    
    Args:
        time_series: Time series data to test
        output_path: Path to save the visualization
        
    Returns:
        dict: Results of the ADF test
    """
    # Perform ADF test
    result = adfuller(time_series.dropna())
    
    # Extract and format results
    adf_result = {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4]
    }
    
    # Interpret result
    is_stationary = result[1] < 0.05
    
    # Create visualization
    if output_path:
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(time_series)
        plt.title('Original Time Series', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        
        plt.subplot(2, 1, 2)
        plt.plot(time_series.diff().dropna())
        plt.title('First Order Differenced Series', fontsize=14)
        plt.xlabel('Time', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        
        plt.suptitle(
            f'Stationarity Test (ADF): {"Stationary" if is_stationary else "Non-Stationary"}\n'
            f'ADF Statistic: {result[0]:.4f}, p-value: {result[1]:.4f}',
            fontsize=16
        )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return adf_result, is_stationary

def fit_arima_model(time_series, output_path):
    """
    Fits an ARIMA model to the time series data and forecasts future values.
    
    Args:
        time_series: DataFrame with date and value columns
        output_path: Path to save the visualization
        
    Returns:
        DataFrame: Forecast results
    """
    print("Fitting ARIMA model...")
    
    # Check if we have enough data
    if len(time_series) < 10:
        print("Error: Not enough data points for ARIMA modeling.")
        return None
    
    # Set the date as index
    ts = time_series.set_index('date')['installations']
    
    # Test for stationarity
    adf_result, is_stationary = test_stationarity(ts)
    
    # Determine if we need differencing
    d = 0 if is_stationary else 1
    
    # Fit ARIMA model (p, d, q) - using auto-determined values
    try:
        import pmdarima as pm
        
        # Use auto_arima to find optimal parameters
        auto_model = pm.auto_arima(
            ts,
            start_p=0, start_q=0,
            max_p=3, max_q=3, max_d=2,
            seasonal=False,
            trace=True,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
        
        best_order = auto_model.order
        print(f"Best ARIMA order: {best_order}")
        
        model = ARIMA(ts, order=best_order)
    except ImportError:
        # Fallback if pmdarima is not available
        print("Using default ARIMA parameters (1,d,1)")
        model = ARIMA(ts, order=(1, d, 1))
    
    # Fit the model
    fitted_model = model.fit()
    
    # Forecast future values (next 12 periods)
    forecast_steps = 12
    forecast = fitted_model.forecast(steps=forecast_steps)
    
    # Create forecast DataFrame
    last_date = ts.index[-1]
    
    # Generate future dates based on frequency
    if isinstance(last_date, pd.Timestamp):
        # Determine the frequency
        if len(ts) < 20:  # Likely yearly or quarterly
            freq = 'Y'
        elif len(ts) < 50:  # Likely monthly
            freq = 'M'
        elif len(ts) < 200:  # Likely weekly
            freq = 'W'
        else:  # Likely daily
            freq = 'D'
        
        # Create future dates
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                    periods=forecast_steps, 
                                    freq=freq)
    else:
        # If not timestamp, just create numeric future indexes
        future_dates = range(len(ts), len(ts) + forecast_steps)
    
    # Create forecast DataFrame
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'forecast': forecast.values
    })
    
    # Plot the results
    plt.figure(figsize=(15, 8))
    
    # Plot historical data
    plt.plot(ts.index, ts.values, label='Historical Data', color='blue')
    
    # Plot forecast
    plt.plot(forecast_df['date'], forecast_df['forecast'], 
             label='ARIMA Forecast', color='red', linestyle='--')
    
    # Calculate confidence intervals (assume normal distribution)
    from scipy import stats
    
    forecast_err = np.sqrt(fitted_model.params['sigma2'])
    conf_level = 0.95
    z_score = stats.norm.ppf(1 - (1 - conf_level) / 2)
    
    # Calculate upper and lower bounds
    forecast_df['lower_bound'] = forecast_df['forecast'] - z_score * forecast_err
    forecast_df['upper_bound'] = forecast_df['forecast'] + z_score * forecast_err
    
    # Plot confidence intervals
    plt.fill_between(forecast_df['date'],
                     forecast_df['lower_bound'],
                     forecast_df['upper_bound'],
                     color='red', alpha=0.2, label=f'{conf_level*100}% Confidence Interval')
    
    # Styling
    plt.title('ARIMA Time Series Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Number of Installations', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create interactive plotly version
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=ts.index, 
        y=ts.values,
        mode='lines+markers',
        name='Historical Data',
        line=dict(color='blue')
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['forecast'],
        mode='lines+markers',
        name='ARIMA Forecast',
        line=dict(color='red', dash='dash')
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
        y=forecast_df['upper_bound'].tolist() + forecast_df['lower_bound'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255, 0, 0, 0)'),
        hoverinfo='skip',
        showlegend=True,
        name=f'{conf_level*100}% Confidence Interval'
    ))
    
    fig.update_layout(
        title='ARIMA Time Series Forecast',
        xaxis_title='Date',
        yaxis_title='Number of Installations',
        template='plotly_white',
        hovermode='x unified'
    )
    
    fig.write_html(output_path.replace('.png', '.html'))
    
    print(f"ARIMA forecast saved to {output_path}")
    return forecast_df

def fit_prophet_model(time_series, output_path):
    """
    Fits a Prophet model for time series forecasting.
    
    Args:
        time_series: DataFrame with date and value columns
        output_path: Path to save the visualization
        
    Returns:
        DataFrame: Forecast results
    """
    print("Fitting Prophet model...")
    
    # Check if we have enough data
    if len(time_series) < 10:
        print("Error: Not enough data points for Prophet modeling.")
        return None
    
    # Convert data to Prophet format
    prophet_df = time_series.rename(columns={'date': 'ds', 'installations': 'y'})
    
    # Create and fit the model
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        interval_width=0.95
    )
    
    m.fit(prophet_df)
    
    # Make future dataframe for prediction
    # Predict for the next 24 months or 2 years
    future = m.make_future_dataframe(periods=24, freq='M')
    
    # Forecast
    forecast = m.predict(future)
    
    # Plot using Prophet's built-in plotting
    fig1 = m.plot(forecast)
    plt.title('Prophet Forecast of RBS Installations', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Number of Installations', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Plot components (seasonality, trend)
    fig2 = m.plot_components(forecast)
    plt.savefig(output_path.replace('.png', '_components.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Create interactive plotly version
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=prophet_df['ds'],
        y=prophet_df['y'],
        mode='markers+lines',
        name='Historical',
        line=dict(color='blue')
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
        y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(255, 0, 0, 0.2)',
        line=dict(color='rgba(255, 0, 0, 0)'),
        hoverinfo='skip',
        showlegend=True,
        name='95% Confidence Interval'
    ))
    
    fig.update_layout(
        title='Prophet Forecast of RBS Installations',
        xaxis_title='Date',
        yaxis_title='Number of Installations',
        template='plotly_white',
        hovermode='x unified'
    )
    
    fig.write_html(output_path.replace('.png', '.html'))
    
    print(f"Prophet forecast saved to {output_path}")
    return forecast

def predict_technology_trends(gdf_rbs, output_path):
    """
    Predicts future trends in technology adoption.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        output_path: Path to save the visualization
    """
    print("Predicting technology trends...")
    
    # Start with temporal preprocessing
    df = preprocess_temporal_data(gdf_rbs)
    
    # Check if we have the necessary columns
    if 'installation_date' not in df.columns or 'Tecnologia' not in df.columns:
        print("Error: Missing required columns for technology trend prediction.")
        return None
    
    # Group by date and technology
    tech_time_series = df.groupby([pd.Grouper(key='installation_date', freq='Q'), 'Tecnologia']).size().unstack(fill_value=0)
    
    # Get list of technologies
    technologies = tech_time_series.columns
    
    # Create forecast for each technology
    tech_forecasts = {}
    forecast_periods = 8  # 2 years of quarterly forecasts
    
    for tech in technologies:
        # Extract time series for this technology
        tech_ts = tech_time_series[tech]
        
        # Fit ARIMA model if we have enough data
        if len(tech_ts) > 8:
            try:
                # Convert to DataFrame for ARIMA function
                tech_df = pd.DataFrame({
                    'date': tech_ts.index,
                    'installations': tech_ts.values
                })
                
                # Simple ARIMA model (1,1,1)
                model = ARIMA(tech_ts.values, order=(1, 1, 1))
                fitted_model = model.fit()
                
                # Forecast
                forecast = fitted_model.forecast(steps=forecast_periods)
                
                # Store result
                last_date = tech_ts.index[-1]
                future_dates = pd.date_range(start=last_date + pd.DateOffset(months=3), 
                                            periods=forecast_periods, 
                                            freq='Q')
                
                tech_forecasts[tech] = pd.Series(forecast, index=future_dates)
            except Exception as e:
                print(f"Error forecasting {tech}: {e}")
                # Use simple trend continuation as fallback
                last_values = tech_ts.values[-3:]
                avg_growth = np.mean(np.diff(last_values))
                
                future_values = [tech_ts.values[-1]]
                for _ in range(forecast_periods - 1):
                    future_values.append(future_values[-1] + avg_growth)
                
                last_date = tech_ts.index[-1]
                future_dates = pd.date_range(start=last_date + pd.DateOffset(months=3), 
                                            periods=forecast_periods, 
                                            freq='Q')
                
                tech_forecasts[tech] = pd.Series(future_values, index=future_dates)
    
    # Combine historical and forecast data
    result_df = tech_time_series.copy()
    
    for tech, forecast in tech_forecasts.items():
        # Add forecast data to result
        for date, value in forecast.items():
            result_df.loc[date, tech] = max(0, value)  # Ensure non-negative values
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    
    # Plot historical data with solid lines
    for tech in technologies:
        historical = tech_time_series[tech]
        plt.plot(historical.index, historical.values, '-', linewidth=2, label=f'{tech} (Historical)')
    
    # Plot forecast data with dashed lines
    for tech, forecast in tech_forecasts.items():
        plt.plot(forecast.index, forecast.values, '--', linewidth=2, label=f'{tech} (Forecast)')
    
    # Add vertical line to separate historical and forecast
    last_historical_date = tech_time_series.index[-1]
    plt.axvline(x=last_historical_date, color='black', linestyle=':', label='Forecast Start')
    
    # Styling
    plt.title('Technology Adoption Forecast', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Number of Installations', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create interactive plotly version
    fig = go.Figure()
    
    # Add historical data
    for tech in technologies:
        historical = tech_time_series[tech]
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical.values,
            mode='lines',
            name=f'{tech} (Historical)',
            line=dict(width=2)
        ))
    
    # Add forecast data
    for tech, forecast in tech_forecasts.items():
        fig.add_trace(go.Scatter(
            x=forecast.index,
            y=forecast.values,
            mode='lines',
            name=f'{tech} (Forecast)',
            line=dict(width=2, dash='dash')
        ))
    
    # Add vertical line
    fig.add_vline(
        x=last_historical_date,
        line_width=2,
        line_dash="dash",
        line_color="black",
        annotation_text="Forecast Start",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title='Technology Adoption Forecast',
        xaxis_title='Date',
        yaxis_title='Number of Installations',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    fig.write_html(output_path.replace('.png', '.html'))
    
    print(f"Technology trend forecast saved to {output_path}")
    return result_df

def run_prediction_analysis(gdf_rbs, results_dir):
    """
    Runs all prediction analyses.
    
    Args:
        gdf_rbs: GeoDataFrame with the RBS data
        results_dir: Directory to save results
    """
    # Create output directory if it doesn't exist
    prediction_dir = os.path.join(results_dir, 'prediction_analysis')
    os.makedirs(prediction_dir, exist_ok=True)
    
    # Preprocess data
    time_series_data = preprocess_prediction_data(gdf_rbs)
    
    if time_series_data is None:
        print("Error: Could not preprocess data for prediction.")
        return
    
    # Fit ARIMA model on monthly data
    if len(time_series_data['monthly']) > 12:  # Need at least a year of data
        fit_arima_model(
            time_series_data['monthly'], 
            os.path.join(prediction_dir, 'arima_forecast_monthly.png')
        )
    else:
        print("Not enough monthly data for ARIMA forecasting.")
    
    # Fit Prophet model
    try:
        if len(time_series_data['monthly']) > 12:
            fit_prophet_model(
                time_series_data['monthly'],
                os.path.join(prediction_dir, 'prophet_forecast_monthly.png')
            )
        else:
            print("Not enough monthly data for Prophet forecasting.")
    except Exception as e:
        print(f"Error during Prophet modeling: {e}")
        print("Prophet may not be installed. Install with: pip install prophet")
    
    # Predict technology trends
    predict_technology_trends(
        gdf_rbs,
        os.path.join(prediction_dir, 'technology_trends_forecast.png')
    )
    
    print(f"All prediction analyses completed and saved to {prediction_dir}") 
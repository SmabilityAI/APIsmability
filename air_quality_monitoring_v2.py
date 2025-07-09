import requests
import json
import datetime
import matplotlib.pyplot as plt
from datetime import datetime as dt, timedelta
import collections
import time
import numpy as np
from urllib.parse import quote
from sklearn.linear_model import LinearRegression
import mplcursors  # Import mplcursors for hover functionality
import folium
import pandas as pd
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors


# IoT API function to fetch air quality at sample rate (5min or 1 min)
def get_air_quality_data(sensor_id, token, timeDeltaArgKey, timeDeltaArgValue, timezone_offset_hours=-6):
    # Get the current UTC time
    current_time = dt.utcnow() + timedelta(hours=timezone_offset_hours)  # Adjust for local time zone
    
    # Define the time window for the last minute
    dtEnd = current_time.strftime('%Y-%m-%d %H:%M:%S')  # Current local time   
    dtStart = (current_time - timedelta(**{timeDeltaArgKey: timeDeltaArgValue})).strftime('%Y-%m-%d %H:%M:%S')  # One minute, 1h,8h,12h,24,1day,7day,earlier
  
    # URL encode the start and end times
    dtEnd_encoded = dtEnd.replace(' ', '%20')
    dtStart_encoded = dtStart.replace(' ', '%20')
    
    # Construct the API URL with HTTPS
    api_url = (f'https://smability.sidtecmx.com/SmabilityAPI/GetData?token={token}'
               f'&idSensor={sensor_id}&dtStart={dtStart_encoded}&dtEnd={dtEnd_encoded}')
    
    print(f"API URL: {api_url}")  # Debug: Print the URL being used
    
    # Create a session with custom headers (Approach 1)
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache'
    })
    
    try:
        print("Using Approach 1: HTTPS with SSL verification")
        
        response = session.get(
            url=api_url, 
            timeout=30,
            verify=True  # SSL verification enabled
        )
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            # Return the parsed JSON data if the request is successful
            return response.json()
        else:
            return f"Error: Unable to fetch data (status code {response.status_code})"
            
    except requests.exceptions.SSLError as e:
        return f"SSL Error: {str(e)}"
    except requests.exceptions.ConnectionError as e:
        return f"Connection Error: {str(e)}"
    except requests.exceptions.Timeout as e:
        return f"Timeout Error: {str(e)}"
    except Exception as e:
        return f"Unexpected Error: {str(e)}"


def get_hourly_air_quality(sensor_id, token, hours, timezone_offset_hours=-6):
    """
    Fetch air quality data and compute hourly averages.

    Parameters:
        sensor_id (int): ID of the sensor.
        token (str): API token for authentication.
        hours (int): Number of past complete hours to average.
        timezone_offset_hours (int): Offset for local time zone (default -6).

    Returns:
        dict: A dictionary with hour intervals and their respective average values.
    """
    # Get the current UTC time adjusted for the local time zone
    current_time = dt.utcnow() + timedelta(hours=timezone_offset_hours)
    end_time = current_time.replace(minute=0, second=0, microsecond=0)  # Round to the start of the current hour
    start_time = end_time - timedelta(hours=hours)  # Start time for the interval

    # URL encode the start and end times
    dtEnd_encoded = end_time.strftime('%Y-%m-%d %H:%M:%S').replace(' ', '%20')
    dtStart_encoded = start_time.strftime('%Y-%m-%d %H:%M:%S').replace(' ', '%20')

    # Construct the API URL with HTTPS
    api_url = (f'https://smability.sidtecmx.com/SmabilityAPI/GetData?token={token}'
               f'&idSensor={sensor_id}&dtStart={dtStart_encoded}&dtEnd={dtEnd_encoded}')

    print(f"Hourly API URL: {api_url}")  # Debug: Print the URL being used

    # Create a session with custom headers (Approach 1)
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Cache-Control': 'no-cache'
    })

    try:
        print("Using Hourly Approach 1: HTTPS with SSL verification")
        
        response = session.get(
            url=api_url, 
            timeout=30,
            verify=True  # SSL verification enabled
        )
        
        print(f"Hourly status code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                return {"error": "Invalid JSON format from API", "details": str(e)}
        else:
            return {"error": f"Unable to fetch data (status code {response.status_code})"}
            
    except requests.exceptions.SSLError as e:
        return {"error": "SSL Error", "details": str(e)}
    except requests.exceptions.ConnectionError as e:
        return {"error": "Connection Error", "details": str(e)}
    except requests.exceptions.Timeout as e:
        return {"error": "Timeout Error", "details": str(e)}
    except Exception as e:
        return {"error": "Unexpected Error", "details": str(e)}

    # Debug: Check the data structure
    print("Hourly API Response length:", len(data) if data else 0)
    
    # Ensure data is a list and contains expected fields
    if not data or not isinstance(data, list):
        return {"error": "No data available for the specified time range"}

    # Process and group data by hour
    hourly_data = {}
    for entry in data:
        # Parse the timestamp and extract the value
        try:
            timestamp = dt.strptime(entry['TimeStamp'], '%Y-%m-%dT%H:%M:%S')
            value = float(entry['Data'])
        except (ValueError, KeyError) as e:
            return {"error": "Invalid data format", "details": str(e)}

        # Round down to the start of the hour
        hour_start = timestamp.replace(minute=0, second=0, microsecond=0)
        if hour_start not in hourly_data:
            hourly_data[hour_start] = []
        hourly_data[hour_start].append(value)

    # Compute averages for each hour
    hourly_averages = {}
    for hour, values in hourly_data.items():
        if len(values) > 0:  # Avoid division by zero
            hourly_averages[hour.strftime('%Y-%m-%d %H:%M:%S')] = round(sum(values) / len(values), 2)

    # Fill missing hours with None
    result = {}
    for i in range(hours):
        target_hour = end_time - timedelta(hours=i + 1)
        result[target_hour.strftime('%Y-%m-%d %H:%M:%S')] = hourly_averages.get(
            target_hour.strftime('%Y-%m-%d %H:%M:%S'), None)

    return result


# Plot air quality data
def plot_air_quality_data(air_quality_data):
    # Validate input data
    if not isinstance(air_quality_data, list):
        print(f"Error: Expected list, got {type(air_quality_data)}")
        return
    
    if len(air_quality_data) == 0:
        print("Error: No data to plot")
        return
    
    # Check if first item has expected structure
    if not isinstance(air_quality_data[0], dict) or 'Data' not in air_quality_data[0]:
        print("Error: Invalid data structure")
        print("Expected dict with 'Data' key, got:", air_quality_data[0])
        return
    
    try:
        o3_values = [float(d['Data']) for d in air_quality_data]  
        timestamps = [datetime.datetime.strptime(d['TimeStamp'], '%Y-%m-%dT%H:%M:%S') for d in air_quality_data]    
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, o3_values, label='O3')
        plt.grid()
        plt.xlabel('Time')
        plt.ylabel('O3 Concentration (ppb)')
        plt.title('O3 Levels Over Time')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting data: {e}")
        print("Sample data structure:", air_quality_data[0] if air_quality_data else "No data")


# Real-time monitoring function
def real_time_monitoring(sensor_id, interval=60):
    while True:
        air_quality_data = get_air_quality_data(sensor_id)
        print(f"Real-time Air Quality Data: {air_quality_data}")
        time.sleep(interval)


# Forecast air quality
def forecast_o3(air_quality_data):
    timestamps = np.array([i for i in range(len(air_quality_data))]).reshape(-1, 1)
    o3_values = np.array([d['Data'] for d in air_quality_data]).reshape(-1, 1)
    model = LinearRegression()
    model.fit(timestamps, o3_values)
    future_timestamps = np.array([len(air_quality_data) + i for i in range(5)]).reshape(-1, 1)
    predicted_values = model.predict(future_timestamps)
    print("O3 Forecast for the next 5 minutes:", predicted_values)


def plot_hourly_ozone_data(hourly_data):
    """
    Plot hourly ozone concentration data with hover functionality for x and y values.

    Args:
        hourly_data (dict): A dictionary where keys are timestamps (str) and values are ozone concentrations (float).
    """
    # Sort the data by timestamp
    sorted_hourly_data = dict(sorted(hourly_data.items()))
    
    # Extract timestamps and ozone values
    timestamps = list(sorted_hourly_data.keys())
    ozone_values = list(sorted_hourly_data.values())
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    line, = plt.plot(timestamps, ozone_values, marker='.', linestyle='-', color='blue', label='Ozone Concentration')
    
    # X-axis formatting
    plt.xticks(
        ticks=range(0, len(timestamps), max(1, len(timestamps) // 12)),  # Show ticks at regular intervals
        labels=[timestamps[i] for i in range(0, len(timestamps), max(1, len(timestamps) // 12))],  # Corresponding labels
        rotation=45,  # Rotate for readability
        fontsize=10
    )
    
    # Labels and title
    plt.xlabel('Time (Hours)', fontsize=12)
    plt.ylabel('Ozone Concentration (ppb)', fontsize=12)
    plt.title('Hourly Ozone Concentration', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add hover functionality
    cursor = mplcursors.cursor(line, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(
        f"Time: {timestamps[int(sel.index)]}\nOzone: {ozone_values[int(sel.index)]:.2f} ppb"))
    
    # Show the plot
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()


# Function parameter initialization
sensor_id = '7'  # O3 sensor, device Hipodromo
token = '1c5e12e8f00c9f2cbb4c7c8f07c9d398'  # Your working token
hours = 48  # Fetch the last 48 complete hours

# Fetch air quality data and compute hourly averages.
averages = get_hourly_air_quality(sensor_id, token, hours)

sensor_id = '7'  # O3 sensor, device Hipodromo
timeDeltaArgKey = "minutes"
timeDeltaArgValue = 2880  # change this number to 60, at 5min sample rate, will result in a list of 12 values
# 60 samples--> 1hr (will plot the total samples reported in one 1hr); 2880 samples->48hrs
# 2880 samples in 48hr at 5min sample rate is 576 samples in 48hrs                        
kwargs = {timeDeltaArgKey: timeDeltaArgValue}


def main():
    # Print air quality data every 5 minute (sampling rate)
    o3_rawdata = get_air_quality_data(sensor_id, token, timeDeltaArgKey, timeDeltaArgValue)
    
    # Debug: Check what we actually received
    print("Type of o3_rawdata:", type(o3_rawdata))
    print("o3_rawdata content:", str(o3_rawdata)[:200] + "..." if len(str(o3_rawdata)) > 200 else str(o3_rawdata))
    
    # Check if we got an error string instead of data
    if isinstance(o3_rawdata, str):
        print("Error occurred:", o3_rawdata)
        return
    
    # Check if we got a dictionary with error key
    if isinstance(o3_rawdata, dict) and 'error' in o3_rawdata:
        print("Error occurred:", o3_rawdata)
        return
    
    # Check if we got valid data
    if not isinstance(o3_rawdata, list) or len(o3_rawdata) == 0:
        print("No valid data received")
        return
    
    length = len(o3_rawdata)
    print("O3 sample size:", length)
    
    # Print first few entries to verify structure
    print("First entry structure:", o3_rawdata[0] if o3_rawdata else "No data")

    # Plot at original sampling rate of 5 min
    plot_air_quality_data(o3_rawdata)

    o3_hrdata = get_hourly_air_quality(sensor_id, token, hours)
    
    # Debug hourly data too
    print("Type of o3_hrdata:", type(o3_hrdata))
    if isinstance(o3_hrdata, dict) and 'error' in o3_hrdata:
        print("Hourly data error:", o3_hrdata)
        return
    
    length = len(o3_hrdata)
    print("O3 hrs:", length)

    # Plot hourly averages
    plot_hourly_ozone_data(o3_hrdata)


if __name__ == "__main__":
    main()

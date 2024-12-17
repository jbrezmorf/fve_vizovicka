import requests
import openmeteo_requests as omr
import requests_cache
from retry_requests import retry
import cfg

import pandas as pd

def pvgis_data(latitude, longitude, years):
    # API Endpoint for PVGIS
    #url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    url= "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"
    start_year, end_year = years
    # API Request Parameters
    params = {
        "lat": latitude,
        "lon": longitude,
        "outputformat": "json",
        "usehorizon": 0,  # Include horizon shading
        "startyear": start_year,
        "endyear": end_year,
        "userhorizon": None,
        "pvcalculation": 0,
        "fixed": 1,
        "angle": 0,
        "aspect": 0,
        "components": 1
    }

    # Send Request
    response = requests.get(url, params=params)
    data = response.json()
    print(data.keys())
    # Parse Results
    df = pd.DataFrame(data["outputs"]["hourly"])
    df_data = {
        'date_time':  pd.to_datetime(df['time'], format='%Y%m%d:%H%M'),
        'Ibeam': df['Gb(i)'],
        'Idiff': df['Gd(i)'],
        #'ground_reflected': df['Gr(i)'],
        #'integrated': df['Int'],
        'sun_elevation': df['H_sun'],
        'temp_out': df['T2m'],
        'wind_10m': df['WS10m']
    }
    df = pd.DataFrame(df_data)
    df.set_index('date_time', inplace=True)
    return df

def open_meteo(latitude, longitude, start, end):
    """
    Study API doc: https://open-meteo.com/en/docs
    :param latitude:
    :param longitude:
    :param years:

    have:
    dni direct noremal irradiance (with respect to sun)

    direct_radiation
    direct_normal_irradiance 	Preceding 15 minutes mean 	W/m² 	x 	x
    global_tilted_irradiance
    global_tilted_irradiance_instant 	Preceding 15 minutes mean 	W/m² 	x 	x
    diffuse_radiation

    temperature_2m
    """
    # Setup the Open-Meteo API client with cache and retry on error
    #cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    #retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    #openmeteo = omr.Client(session=retry_session)



    # Request weather data from Open-Meteo
    url = "https://api.open-meteo.com/v1/forecast"
    params={
        "latitude": latitude,
        "longitude": longitude,
        "timeformat": "unixtime",
        "tilt": 0,
        "azimuth": 0,
        "hourly": ["temperature_2m", "wind_speed_10m", "direct_radiation", "diffuse_radiation", "global_tilted_irradiance"],
        "start_date": f"{start}-12-11",
        "end_date": f"{end}-12-11",
    }
    response = requests.get(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = response.json()
    print(response)
    return
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(1).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    print(hourly_dataframe)




if __name__ == '__main__':
    latitude = 50.0
    longitude = 20.0
    #data = open_meteo(latitude, longitude, start=2022, end=2024)
    data = pvgis_data(latitude, longitude, [2022, 2023])
    print(data)
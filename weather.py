import requests
import pandas as pd

def pvgis_data(latitude, longitude, years):
    # API Endpoint for PVGIS
    url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    start_year, end_year = years
    # API Request Parameters
    params = {
        "lat": latitude,
        "lon": longitude,
        "outputformat": "json",
        "usehorizon": 1,  # Include horizon shading
        "startyear": start_year,
        "endyear": end_year,
        "optimalangles": 1,
        "userhorizon": None
    }

    # Send Request
    response = requests.get(url, params=params)
    data = response.json()

    # Parse Results
    irradiance_data = pd.DataFrame(data["outputs"]["hourly"])
    irradiance_data["time"] = pd.to_datetime(irradiance_data["time"])
    irradiance_data.set_index("time", inplace=True)
    irradiance_data.to_csv("irradiance_data.csv")

    # Extract GHI and DHI
    ghi = irradiance_data["G(h)"]  # Global Horizontal Irradiance
    dhi = irradiance_data["H(h)"]  # Diffuse Horizontal Irradiance

    return ghi, dhi

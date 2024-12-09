import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
import pytz
from pvlib.location import Location
from pvlib.irradiance import get_total_irradiance
from pvlib.pvsystem import PVSystem
from pvlib.iotools import get_pvgis_tmy
from pvlib import solarposition

def spot_price():
    """
    Read spot price
    :return:
    """
    spot_price = pd.read_csv('spot_price.csv', index_col=0, parse_dates=True)
    return spot_price


def make_horizon():
    # Use hard-coded horizon profile data from location object above.
    profile_dict = {
        0:0.34,
        4:0.34,
        9:1.01,
        25:1.20,
        37:2.05,
        65:1.86,
        97:0.63,
        159:1.96,
        189:6.60,
        206:6.51,
        218:9.35,
        248:9.54,
        267:6.51,
        292:6.03,
        318:1.01,
        360:0.72
    }
    # Extract keys and values from the dictionary as numpy arrays
    x = np.array(list(profile_dict.keys()))
    y = np.array(list(profile_dict.values()))
    interp_func = interpolate.interp1d(x, y, kind='linear', fill_value="extrapolate")



    azimuth = np.arange(0, 361, 1.0)
    horizon_profile = pd.Series(interp_func(azimuth), index=azimuth)

    # Create a matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(8, 4))

    horizon_profile.plot(ax=ax, xlim=(0, 360), ylim=(0, None))
    ax.set_title('Horizon profile')
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel('Azimuth [°]')
    ax.set_ylabel('Horizon angle [°]')
    plt.savefig('horizon_profile.pdf', bbox_inches='tight')
    return interp_func

def compute_production(df, latitude, longitude, panel_config):
    cet_timezone = pytz.timezone('CET')
    times_cet = pd.to_datetime(df.date_time).dt.tz_localize(cet_timezone)
    times_utc = times_cet.dt.tz_convert('UTC')
    times_utc = times_utc + pd.Timedelta(hours=-0.2)
    # Ensure the index is a DatetimeIndex
    times_utc_index = times_utc if isinstance(times_utc, pd.DatetimeIndex) \
        else pd.DatetimeIndex(times_utc)
    df = df.set_index(times_utc_index)

    # Create location object
    location = Location(latitude, longitude)
    solar_position = location.get_solarposition(times_utc_index)

    solar_azimuth = solar_position.azimuth
    solar_zenith = solar_position.apparent_zenith
    print(f"Zenith difference: {np.max(np.abs(solar_zenith - df['Zenith']))}, {np.mean(np.abs(solar_zenith - df['Zenith']))}")
    solar_elevation = solar_position.apparent_elevation
    dni = df['DNI']
    ghi = df['GHI']
    dhi = df['DHI']

    horizon_func = make_horizon()
    # Map solar_azimuth to horizon values
    horizon_for_date_time = horizon_func(solar_azimuth)

    # Adjust DNI based on data - note this is returned as numpy array
    dni_adjusted = np.where(solar_elevation > horizon_for_date_time, dni, 0)

    # Adjust GHI and set it to DHI for time-periods where 'dni_adjusted' is 0.
    # Note this is returned as numpy array
    ghi_adjusted = np.where(dni_adjusted == 0, dhi, ghi)

    # Create PV system with specified configuration
    system = PVSystem(
        surface_tilt=panel_config.inclination,
        surface_azimuth=panel_config.azimuth
    )

    # Compute total irradiance (plane-of-array)
    poa_irradiance = get_total_irradiance(
        surface_tilt=panel_config.inclination,
        surface_azimuth=panel_config.azimuth,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        dni=dni_adjusted,
        ghi=ghi_adjusted,
        dhi=dhi
    )

    # Calculate AOI and Reflection Loss
    aoi = pvlib.irradiance.aoi(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth']
    )
    reflection_loss = aoi_loss(aoi, surface_type='glass')  # Reflection loss

    # Spectral Correction
    airmass = pvlib.atmosphere.get_relative_airmass(solar_position['apparent_zenith'])
    spectral_correction = spectral_factor(airmass, poa_irradiance['poa_global'], module_parameters['a_ref'])

    # Apply Losses
    poa_effective = poa_irradiance['poa_global'] * spectral_correction * (1 - reflection_loss)

    # Temperature Effects
    temperature_cell = pvlib.temperature.sapm_cell(
        poa_irradiance['poa_global'], clearsky['temp_air'], clearsky['wind_speed'], temperature_model_parameters
    )

    # Module Power Using SAPM
    power = pvlib.pvsystem.sapm(
        effective_irradiance=poa_effective,
        temp_cell=temperature_cell,
        parameters=module_parameters
    )
    # Estimate production per panel
    effective_irradiance = poa_irradiance['poa_global']
    production_per_panel = effective_irradiance  # Assuming 1 W/m²/panel efficiency

    # Total production
    total_production = panel_config.n * production_per_panel

    return total_production.to_numpy()

def compute_clear_sky_production(df_column, latitude, longitude, panel_config):
    """
    Computes clear sky production of PV panels.

    Parameters:
    - df_column: pd.Series of datetime objects in CET timezone.
    - latitude: float, latitude of the installation site.
    - longitude: float, longitude of the installation site.
    - panel_config: PanelConfig, configuration of the panels.

    Returns:
    - np.ndarray: Clear sky production in Watts.
    """
    # Convert datetime to UTC for calculations
    cet_timezone = pytz.timezone('CET')
    times_cet = pd.to_datetime(df_column).dt.tz_localize(cet_timezone)
    times_utc = times_cet.dt.tz_convert('UTC')
    # Ensure the index is a DatetimeIndex
    times_utc_index = times_utc if isinstance(times_utc, pd.DatetimeIndex) \
        else pd.DatetimeIndex(times_utc)

    # Create location object
    location = Location(latitude, longitude)
    solar_position = location.get_solarposition(times_utc_index)

    # get_pvgis_tmy returns two additional values besides df and metadata
    df, _, _, metadata = get_pvgis_tmy(latitude, longitude, map_variables=True)

    # Get clear-sky data (GHI, DNI, DHI)
    clearsky = location.get_clearsky(times_utc_index)

    solar_azimuth = solar_position.azimuth
    solar_zenith = solar_position.apparent_zenith
    solar_elevation = solar_position.apparent_elevation
    dni = clearsky.dni
    ghi = clearsky.ghi
    dhi = clearsky.dhi

    horizon_func = make_horizon()
    # Map solar_azimuth to horizon values
    horizon_for_date_time = horizon_func(solar_azimuth)

    # Adjust DNI based on data - note this is returned as numpy array
    dni_adjusted = np.where(solar_elevation > horizon_for_date_time, dni, 0)

    # Adjust GHI and set it to DHI for time-periods where 'dni_adjusted' is 0.
    # Note this is returned as numpy array
    ghi_adjusted = np.where(dni_adjusted == 0, dhi, ghi)

    # Create PV system with specified configuration
    system = PVSystem(
        surface_tilt=panel_config.inclination,
        surface_azimuth=panel_config.azimuth
    )

    # Calculate solar position
    solar_position = location.get_solarposition(times_utc_index)

    # Compute total irradiance (plane-of-array)
    poa_irradiance = get_total_irradiance(
        surface_tilt=panel_config.inclination,
        surface_azimuth=panel_config.azimuth,
        solar_zenith=solar_zenith,
        solar_azimuth=solar_azimuth,
        dni=dni_adjusted,
        ghi=ghi_adjusted,
        dhi=dhi
    )


    # Estimate production per panel
    effective_irradiance = poa_irradiance['poa_global']
    production_per_panel = effective_irradiance  # Assuming 1 W/m²/panel efficiency

    # Total production
    total_production = panel_config.n * production_per_panel

    return total_production.to_numpy()

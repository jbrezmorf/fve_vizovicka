from typing import Dict
import attrs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fit_models import estimate_time_shift
from plot_days import fve_plots
import scipy.interpolate as interpolate
import pytz
import pvlib

import cfg
import weather


def spot_price():
    """
    Read spot price
    :return:
    """
    spot_price = pd.read_csv('spot_price.csv', index_col=0, parse_dates=True)
    return spot_price



#######################

@attrs.define
class PanelConfig:
    n: int
    azimuth: int
    inclination: int



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



    # azimuth = np.arange(0, 361, 1.0)
    # horizon_profile = pd.Series(interp_func(azimuth), index=azimuth)

    # Create a matplotlib figure and axis
    # fig, ax = plt.subplots(figsize=(8, 4))
    #
    # horizon_profile.plot(ax=ax, xlim=(0, 360), ylim=(0, None))
    # ax.set_title('Horizon profile')
    # ax.set_xticks([0, 90, 180, 270, 360])
    # ax.set_xlabel('Azimuth [°]')
    # ax.set_ylabel('Horizon angle [°]')
    # plt.savefig('horizon_profile.pdf', bbox_inches='tight')
    return interp_func




# def weather_from_input_df(df: pd.DataFrame):
#     """
#     See obligatory and optional fields:
#     https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.modelchain.ModelChain.run_model.html#pvlib.modelchain.ModelChain.run_model
#     :param df:
#     :return:
#     """
#     return pd.DataFrame({
#         'ghi': df['GHI'],
#         'dhi': df['DHI'],
#         'dni': df['DNI'],
#         'temp_air': df['temp_out'],
#         'Zenith': 90 - df['sun_elevation'],
#         # module temperature imply temperature_model must be 'sapm'
#         'module_temperature': df['temp_mod_w']
#     })

@cfg.mem.cache
def get_weather(date_time):
    unique_years = set(date_time.map(lambda x: x.year))
    unique_years = list(sorted(unique_years))
    df = weather.pvgis_data(cfg.location['latitude'], cfg.location['longitude'], [unique_years[0], unique_years[-1]])
    #df = pvlib.iotools.get_pvgis_hourly(lat=cfg.location['latitude'], lon=cfg.location['longitude'], start=unique_years[0], end=unique_years[-1])
    #df = weather.open_meteo(cfg.location['latitude'], cfg.location['longitude'], start=unique_years[0], end=unique_years[-1])
    zenith = 90 - df['sun_elevation']
    cs = np.cos(np.radians(zenith))
    dni = df['Ibeam'] / cs
    dni[cs < 0.04] = 0.0
    df['DNI'] = dni

    #plt.scatter(dni, df['Ibeam'])
    #plt.xlim(0, 100)
    #plt.ylim(0, 0.2)
    #plt.show()
    return df


def pv_model(pannels:Dict[str,PanelConfig], df:pd.DataFrame, horizon=False):
    """
    Compute the production of a PV system given the configuration of the panels, the weather data and the location.

    Parameters:
    - pannels: PanelConfig, configuration of the panels.
    - df: pd.DataFrame, weather data.
    - latitude: float, latitude of the installation site.
    - longitude: float, longitude of the installation site.

    Returns:
    - np.ndarray: Production in Watts.
    """

    #cet_timezone  = pytz.FixedOffset(60)  # 60 minutes = 1 hours, CET winter is UTC+1
    #pytz.timezone('CET')
    #times_cet = pd.to_datetime(df.date_time).dt.tz_localize(cet_timezone)
    #times_utc = times_cet.dt.tz_convert('UTC')
    #times_utc = times_utc + pd.Timedelta(hours=-0.2)
    # Ensure the index is a DatetimeIndex
    #times_utc_index = times_utc if isinstance(times_utc, pd.DatetimeIndex) \
    #    else pd.DatetimeIndex(times_utc)
    #df = df.set_index(times_utc_index)
    df = df.set_index('date_time')
    df_zenith = 90 - df['sun_elevation']

    # Create location object
    location = pvlib.location.Location(**cfg.location)
    solar_position = location.get_solarposition(df.index)
    # solar_azimuth = solar_position.azimuth
    solar_zenith = solar_position.apparent_zenith
    # solar_elevation = solar_position.apparent_elevation

    # Time correction to best Zenith fit
    dt = estimate_time_shift(np.minimum(90.0, solar_zenith), df_zenith)
    # dt > 0 means df_zenith is delayed by dt after solar_zenith
    print("Time shift:", dt)
    # Shift time and thus the computed solar zenith
    times_shifted= df.index + pd.Timedelta(hours=-dt)

    solar_position = location.get_solarposition(pd.DatetimeIndex(times_shifted))
    solar_azimuth = np.array(solar_position.azimuth)
    solar_zenith = np.array(solar_position.apparent_zenith)
    solar_elevation = np.array(solar_position.apparent_elevation)

    df['solar_zenith'] = np.minimum(90, np.array(solar_zenith))
    df['Zenith'] = df_zenith
    fve_plots(cfg.workdir, df, {}, "sun_", ['Zenith', 'solar_zenith'])

    dni = np.cos(np.radians(pannels['East'].inclination)) * df['DNI']
    #dni=0.0 * df['DNI']
    ghi = df['GHI']
    dhi = df['DHI']

    if horizon:
        horizon_func = make_horizon()
        # # Map solar_azimuth to horizon values
        horizon_for_date_time = horizon_func(solar_azimuth)
    #
        # # Adjust DNI based on data - note this is returned as numpy array
        dni = np.where(solar_elevation > horizon_for_date_time, dni, 0)
        ghi = np.where(dni == 0, dhi, ghi)
    #
    # # Adjust GHI and set it to DHI for time-periods where 'dni_adjusted' is 0.
    # # Note this is returned as numpy array
    weather = pd.DataFrame({
        'ghi': ghi,
        'dhi': dhi,
        'dni': dni,
        'temp_air': df['temp_out'],
        # module temperature imply temperature_model must be 'sapm'
        #'module_temperature': df['temp_mod_w']
    })

    #
    # # Create PV system with specified configuration
    # system = pvlib.pvsystem.PVSystem(
    #     surface_tilt=panel_config.inclination,
    #     surface_azimuth=panel_config.azimuth
    # )
    #
    def irrad_func(name, pannel:PanelConfig):
        print("pannel: ", pannel.inclination, pannel.azimuth)
        module_dict = dict(
             surface_tilt=pannel.inclination,
             surface_azimuth=pannel.azimuth,
             solar_zenith=solar_zenith,
             solar_azimuth=solar_azimuth,
             #model='isotropic'
         )
        module_dict.update(weather)
        del module_dict['temp_air']
        # # Compute total irradiance (plane-of-array)
        res_df = pvlib.irradiance.get_total_irradiance(**module_dict)
        return res_df.rename(columns={k: f'{k}_{name}' for k in res_df.columns})
    dfs = [irrad_func(k, v) for k, v in pannels.items()]
    irrad_df = pd.concat([weather, *dfs], axis=1)

    #
    # # Calculate AOI and Reflection Loss
    # aoi = pvlib.irradiance.aoi(**module_dict)
    # reflection_loss = pvlib.aoi_loss(aoi, surface_type='glass')  # Reflection loss
    #
    # # Spectral Correction
    # airmass = pvlib.atmospherepd.Series(poa.index).get_relative_airmass(solar_position['apparent_zenith'])
    # spectral_correction = spectral_factor(airmass, poa_irradiance['poa_global'], module_parameters['a_ref'])
    #
    # # Apply Losses
    # poa_effective = poa_irradiance['poa_global'] * spectral_correction * (1 - reflection_loss)
    #
    # # Temperature Effects
    # temperature_cell = pvlib.temperature.sapm_cell(
    #     poa_irradiance['poa_global'], clearsky['temp_air'], clearsky['wind_speed'], temperature_model_parameters
    # )
    #
    # # Module Power Using SAPM
    # power = pvlib.pvsystem.sapm(
    #     effective_irradiance=poa_effective,
    #     temp_cell=temperature_cell,
    #     parameters=module_parameters
    # )
    # # Estimate production per panel
    # effective_irradiance = poa_irradiance['poa_global']
    # production_per_panel = effective_irradiance  # Assuming 1 W/m²/panel efficiency
    #
    # # Total production
    # total_production = panel_config.n * production_per_panel

    cec_modules = pvlib.pvsystem.retrieve_sam('CECMod')
    #sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

    def pv_array(name, pannel:PanelConfig):
        return pvlib.pvsystem.Array(
            pvlib.pvsystem.FixedMount(
                pannel.inclination,
                pannel.azimuth,
                **cfg.mount),
            name=name,
            modules_per_string=18,
            strings=1,
            module_parameters=cfg.aiko_neostar_2s_a500,
            temperature_model_parameters=cfg.temperature_model,
            array_losses_parameters=dict(dc_ohmic_percent=0.5)
        )
    arrays = [pv_array(k, v) for k, v in pannels.items()]
    idx_arr_dict = {arr.name: i for i, arr in enumerate(arrays)}
    system = pvlib.pvsystem.PVSystem(arrays=arrays, inverter_parameters=cfg.inverter)
    mc = pvlib.modelchain.ModelChain(
        system, location, aoi_model='physical',
        spectral_model='no_loss')

    mc.run_model(weather)

    new_cols = pd.DataFrame({
        #'dni': weather['dni'],
        #'dhi': weather['dhi'],
        #'ghi': weather['ghi'],
        'irr_eff_e': irrad_df['poa_global_East'],
        'irr_eff_w': irrad_df['poa_global_West'],
        'irr_global_pv': irrad_df['poa_global_East'] + irrad_df['poa_global_West'],
        #'irr_eff_e': mc.results.effective_irradiance[idx_arr_dict['East']],
        #'irr_eff_w': mc.results.effective_irradiance[idx_arr_dict['West']],
        #'irr_global_pv': (mc.results.total_irrad[0].poa_global + mc.results.total_irrad[1].poa_global),
        'pv_energy_DC': (mc.results.dc[0].p_mp + mc.results.dc[1].p_mp) * 3600 / 1000,
        'calc_zenith': mc.results.solar_position.apparent_zenith
    })
    res_df = pd.concat([df, new_cols, irrad_df], axis=1)
    return res_df

#
# def pv_model_simple(pannels:Dict[str,PanelConfig], df:pd.DataFrame, horizon=False):
#     """
#     Compute the production of a PV system given the configuration of the panels, the weather data and the location.
#
#     Parameters:
#     - pannels: PanelConfig, configuration of the panels.
#     - df: pd.DataFrame, weather data.
#     - latitude: float, latitude of the installation site.
#     - longitude: float, longitude of the installation site.
#
#     Returns:
#     - np.ndarray: Production in Watts.
#     """
#
#     #cet_timezone  = pytz.FixedOffset(60)  # 60 minutes = 1 hours, CET winter is UTC+1
#     #pytz.timezone('CET')
#     #times_cet = pd.to_datetime(df.date_time).dt.tz_localize(cet_timezone)
#     #times_utc = times_cet.dt.tz_convert('UTC')
#     #times_utc = times_utc + pd.Timedelta(hours=-0.2)
#     # Ensure the index is a DatetimeIndex
#     #times_utc_index = times_utc if isinstance(times_utc, pd.DatetimeIndex) \
#     #    else pd.DatetimeIndex(times_utc)
#     #df = df.set_index(times_utc_index)
#     df = df.set_index('date_time')
#     df_zenith = 90 - df['sun_elevation']
#
#     # Create location object
#     location = pvlib.location.Location(**cfg.location)
#     solar_position = location.get_solarposition(df.index)
#     # solar_azimuth = solar_position.azimuth
#     solar_zenith = solar_position.apparent_zenith
#     # solar_elevation = solar_position.apparent_elevation
#
#     # Time correction to best Zenith fit
#     dt = estimate_time_shift(np.minimum(90.0, solar_zenith), df_zenith)
#     # dt > 0 means df_zenith is delayed by dt after solar_zenith
#     print("Time shift:", dt)
#     # Shift time and thus the computed solar zenith
#     times_shifted= df.index + pd.Timedelta(hours=-dt)
#
#     solar_position = location.get_solarposition(pd.DatetimeIndex(times_shifted))
#     solar_azimuth = np.array(solar_position.azimuth)
#     solar_zenith = np.array(solar_position.apparent_zenith)
#     solar_elevation = np.array(solar_position.apparent_elevation)
#
#     df['solar_zenith'] = np.minimum(90, np.array(solar_zenith))
#     df['Zenith'] = df_zenith
#     fve_plots(cfg.workdir, df, {}, "sun_", ['Zenith', 'solar_zenith'])
#
#     if 'DNI' not in df.columns:
#         Ibeam = df['GHI'] - df['DHI']
#         dni =  Ibeam / np.cos(df_zenith)   # Direct Normal Irradiance
#     else:
#         dni = df['DNI']
#     ghi = df['GHI']
#     dhi = df['DHI']
#     if horizon:
#         horizon_func = make_horizon()
#         # # Map solar_azimuth to horizon values
#         horizon_for_date_time = horizon_func(solar_azimuth)
#     #
#         # # Adjust DNI based on data - note this is returned as numpy array
#         dni = np.where(solar_elevation > horizon_for_date_time, dni, 0)
#         ghi = np.where(dni == 0, dhi, ghi)
#     #
#     # # Adjust GHI and set it to DHI for time-periods where 'dni_adjusted' is 0.
#     # # Note this is returned as numpy array
#     weather = pd.DataFrame({
#         'ghi': ghi,
#         'dhi': dhi,
#         'dni': dni,
#         'temp_air': df['temp_out'],
#         # module temperature imply temperature_model must be 'sapm'
#         #'module_temperature': df['temp_mod_w']
#     })
#
#     pannel = pannels['East']
#     print("pannel: ", pannel.inclination, pannel.azimuth)
#     module_dict = dict(
#          surface_tilt=pannel.inclination,
#          surface_azimuth=pannel.azimuth,
#          solar_zenith=solar_zenith,
#          solar_azimuth=solar_azimuth,
#          #model='isotropic'
#      )
#     module_dict.update(weather)
#     del module_dict['temp_air']
#     # # Compute total irradiance (plane-of-array)
#     poa = pvlib.irradiance.get_total_irradiance(**module_dict)
#     poa = pd.concat([weather, poa], axis=1)
#     dt = pd.Series(poa.index)
#     poa.reset_index(inplace=True)
#     poa['date_time'] = dt
#     return poa
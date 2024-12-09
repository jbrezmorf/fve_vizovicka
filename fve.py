import attrs
import pandas as pd
import numpy as np
import seaborn as sns
import math
from datetime import datetime, timedelta
#from pysolar.solar import get_azimuth, get_altitude
from pvlib_model import *
from spot import *

import pandas as pd
import matplotlib.pyplot as plt

import pathlib
script_dir = pathlib.Path(__file__).parent
workdir = script_dir / "workdir"


def compute_sun_normal_angle_with_pvlib(df, lat, lon, panel_azimuth, panel_inclination):
    """
    Computes the cosine of the angle between the sun vector and a given normal vector
    for all values in the DataFrame using pvlib to calculate the sun's position.

    Parameters:
        df (pd.DataFrame): The DataFrame containing date_time information.
        date_time_col (str): The column name with date-time values.
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        normal_azimuth (float): Azimuth of the normal vector (degrees).
        normal_inclination (float): Inclination of the normal vector (degrees).

    Returns:
        pd.Series: A Pandas Series with the cosine of the angle for each row.
    """
    # Convert normal vector angles from degrees to radians
    normal_azimuth_rad = np.radians(panel_azimuth)
    normal_inclination_rad = np.radians(90 - panel_inclination)

    # Convert normal vector to 3D unit vector
    normal_x = np.cos(normal_inclination_rad) * np.sin(normal_azimuth_rad)
    normal_y = np.cos(normal_inclination_rad) * np.cos(normal_azimuth_rad)
    normal_z = np.sin(normal_inclination_rad)
    normal_vector = np.array([normal_x, normal_y, normal_z])

    # Calculate solar position using pvlib
    solar_pos = solarposition.get_solarposition(
        time=pd.to_datetime(df['time'], format='%d.%m. %H:%M'),
        latitude=lat,
        longitude=lon
    )
    sun_altitude = solar_pos['apparent_elevation']
    sun_azimuth = solar_pos['azimuth']

    # Convert sun position to radians
    sun_azimuth_rad = np.radians(sun_azimuth)
    sun_inclination_rad = np.radians(sun_altitude)

    # Convert sun position to a 3D unit vector
    sun_x = np.cos(sun_inclination_rad) * np.sin(sun_azimuth_rad)
    sun_y = np.cos(sun_inclination_rad) * np.cos(sun_azimuth_rad)
    sun_z = np.sin(sun_inclination_rad)
    sun_vectors = np.stack([sun_x, sun_y, sun_z], axis=1)

    # Compute dot product between sun and normal vectors
    dot_products = np.dot(sun_vectors, normal_vector)

    # Ensure dot_products are within valid range for cosines ([-1, 1])
    dot_products = np.clip(dot_products, -1, 1)

    # Return the cosine of the angle
    return pd.Series(np.degrees(sun_inclination_rad), index=df.index)


def read_csv_to_dataframe(file_path):
    """
    Reads a CSV file into a Pandas DataFrame.

    Parameters:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    try:
        # Read the CSV file into a Pandas DataFrame
        df = pd.read_csv(file_path, delimiter=";", skiprows=1, header=0, dtype='str', encoding="utf-8")
        units = df.iloc[0]  # Extract units from the first row
        df = df.iloc[17:]  # Start data from the 20th row (index=19 in the original file)
        df.reset_index(drop=True, inplace=True)  # Reset index after slicing
        print(f"Data successfully loaded. Shape: {df.shape}")

        # read units

        return df, units
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


def create_shortcuts_and_rename(df, units):
    """
    Creates shortcuts for all provided column names and renames the DataFrame accordingly.

    Parameters:
        df (pd.DataFrame): The DataFrame to rename columns for.

    Returns:
        pd.DataFrame: The DataFrame with renamed columns.
        dict: A dictionary mapping shortcuts to full column descriptions.
    """
    # Dictionary for all columns with shortcuts
    shortcuts = {
        "time": "Doba",
        "GHI": "Intenzita záření na horizontálu",
        "DHI": "Difúzní záření na vodorovné rovině",
        "temp_out": "Venkovní teplota",
        "sun_alt_w": "Budovy 02-Oblast modulu Západ: Výška slunce",
        "irr_tilt_w": "Budovy 02-Oblast modulu Západ: Intenzita záření na skloněnou plochu",
        "temp_mod_w": "Budovy 02-Oblast modulu Západ: Teplota modulu",
        "sun_alt_e": "Budovy 02-Oblast modulu Východ: Výška slunce",
        "irr_tilt_e": "Budovy 02-Oblast modulu Východ: Intenzita záření na skloněnou plochu",
        "temp_mod_e": "Budovy 02-Oblast modulu Východ: Teplota modulu",
        "irr_eff_w": "Budovy 02-Oblast modulu Západ: Globální záření na modul",
        "irr_eff_e": "Budovy 02-Oblast modulu Východ: Globální záření na modul",

        "irr_global_horiz_w": "Budovy 02-Oblast modulu Západ: Globální záření - horizontální",
        "spec_loss_w": "Budovy 02-Oblast modulu Západ: Ztráty kvůli Odchylka od standardního spektra",
        "albedo_loss_w": "Budovy 02-Oblast modulu Západ: Ztráty kvůli Odraz od země (Albedo)",
        "module_yield_w": "Budovy 02-Oblast modulu Západ: Výnosy Vyrovnání a sklon úrovně modulu",
        "shade_loss_w": "Budovy 02-Oblast modulu Západ: Ztráty kvůli Odstínění, zaclonění",
        "module_refl_w": "Budovy 02-Oblast modulu Západ: Odraz na povrchu modulu",
        "irr_back_w": "Budovy 02-Oblast modulu Západ: Intenzita záření na zadní části modulu",

        "irr_global_horiz_e": "Budovy 02-Oblast modulu Východ: Globální záření - horizontální",
        "spec_loss_e": "Budovy 02-Oblast modulu Východ: Ztráty kvůli Odchylka od standardního spektra",
        "albedo_loss_e": "Budovy 02-Oblast modulu Východ: Ztráty kvůli Odraz od země (Albedo)",
        "module_yield_e": "Budovy 02-Oblast modulu Východ: Výnosy Vyrovnání a sklon úrovně modulu",
        "shade_loss_e": "Budovy 02-Oblast modulu Východ: Ztráty kvůli Odstínění, zaclonění",
        "module_refl_e": "Budovy 02-Oblast modulu Východ: Odraz na povrchu modulu",
        "irr_back_e": "Budovy 02-Oblast modulu Východ: Intenzita záření na zadní části modulu",

        "pv_energy_DC": "Střídač 1 do  Budovy 02-Oblast modulu Východ & Budovy 02-Oblast modulu Západ: Energie na vstupu měniče",
        "pv_net_energy": "Dodávky energie do sítě",  # Skutečná produkce FV, zráty převodu DC/AC, kabelů

        # %Rízené toky energie
        "grid_supply": "Dodávka do sítě",
        "grid_energy": "Energie ze sítě",
        "consumption": "Spotřeba",
        "feed_limit": "Omezení přetoků do sítě",
        "self_cons": "Vlastní spotřeba",
        "charge_loss": "Ztráty nabíjením/vybíjením",
        "battery_soc": "Stav nabití baterií (ve vztahu k C10)",
        "load_cycle": "Cyklické zatížení",

        "irr_global_horiz": "Globální záření - horizontální",
        "spectrum_dev": "Odchylka od standardního spektra",
        "albedo": "Odraz od země (Albedo)",
        "module_tilt_loss": "Vyrovnání a sklon úrovně modulu",
        "shade_loss": "Odstínění, zaclonění",
        "module_refl": "Odraz na povrchu modulu",
        "irr_back": "Intenzita záření na zadní části modulu",
        "irr_global_mod": "Globální záření na modul",
        "irr_global_pv": "FV globální záření",
        "bifaciality": "Bifacilita (Oboustrannost)",
        "soiling": "Znečistění",
        "stc_conv": "STC konverze (jmenovitá účinnost modulu)",
        "rated_pv_energy": "FV jmenovitá energie",
        "low_light_loss": "Chování za nízké intenzity světla",
        "module_shade_loss": "Specifické dílčí stínění modulu",
        "temp_dev_loss": "Odchylka od jmenovité teploty modulu",
        "diode_loss": "Diody",
        "mismatch_loss_prod": "Nesrovnalost/Nesoulad (údaje výrobce)",
        "mismatch_loss_shade": "Nesrovnalost/Nesoulad (zapojení/stínění)",
        "optimizer_loss": "Výkonový optimizér (přemena DC/deregulace)",
        "pv_dc_energy": "FV energie (DC)",
        "dc_power_loss": "Pokles pod výchozí výkon DC",
        "mpp_voltage_reg_loss": "Sestupná regulace z důvodu napěťového rozsahu MPP",
        "mpp_dc_current_loss": "Sestupná regulace z důvodu max. DC proudu",
        "mpp_dc_power_loss": "Sestupná regulace z důvodu max. DC výkonu",
        "mpp_ac_power_loss": "Sestupná regulace z důvodu max. AC výkonu/cos phi",
        "mpp_adj_loss": "Přizpůsobení MPP",
        "pv_ac_energy": "FV energie (AC)",
        "converter_loss": "Ztráty kvůli Převod DC/AC",
        "cable_loss": "Ztráty v kabelech celkem",
        "standby_loss": "Vlastní spotřeba (pohotovostní režim nebo noc)",
    }

    # explicitely convert columns from 1 to float
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c].str.replace(',', '.'), errors='raise')

    # Reverse mapping to rename the DataFrame columns
    reverse_mapping = {v: k for k, v in shortcuts.items()}

    # Rename the columns in the DataFrame
    df.columns = [c.strip(' ') for c in df.columns]
    df.rename(columns=reverse_mapping, inplace=True)
    # Ensure the 'Date' column is parsed into a datetime object for easier manipulation
    date_time = pd.to_datetime(
                    df['time'].str.replace(' ', ''),  # Remove spaces for cleaner parsing
                    format='%d.%m.%H:%M')
    zenith = 90 - df["sun_alt_w"]
    new_cols = pd.DataFrame({
        'date_time': date_time, # Extract hour and month from the parsed datetime
        'hour': date_time.dt.hour,
        'month': date_time.dt.month,
        "Zenith": zenith,
        'DNI': (df["GHI"] - df["DHI"]) / np.cos(np.radians(zenith)),
    })
    df = pd.concat([df, new_cols], axis=1)

    # rename units keys
    units_dict = {k.strip(' '): v for k, v in units.to_dict().items()}
    units_dict = {reverse_mapping[k]:v for k,v in units_dict.items() if k in reverse_mapping}
    return df, units_dict




def plot_monthly_hourly_averages(df, q_col, unit, axes=None):
    """
    Plots monthly hourly averages of a given quantity 'Q' on the provided axis.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        date_col (str): The name of the column with date information in the format 'day.month. hour:00'.
        q_col (str): The column name for the quantity 'Q' to be plotted.
        color (str): The color to use for the plot lines.
        ax (matplotlib.axes.Axes, optional): The axis to add the plot to. If None, creates a new figure.

    Returns:
        matplotlib.axes.Axes: The axis with the plot added.
    """

    # Prepare the plot
    #fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    ax1, ax2 = axes

    plotted_columns = ['hour', 'month', q_col]  # Replace with the relevant column names
    df = df[plotted_columns].copy()

    # Aggregate data: averages and maxima
    avg_df = df.groupby(['hour', 'month'], as_index=False).mean()
    max_df = df.groupby(['hour', 'month'], as_index=False).max()

    color_list = [
        "red",  # January
        "orange",  # February
        "gold",  # March
        "limegreen",  # April
        "green",  # May
        "teal",  # June
        "blue",  # July
        "navy",  # August
        "purple",  # September
        "magenta",  # October
        "brown",  # November
        "pink"  # December
    ]

    # Create a custom palette from the color list
    custom_palette = sns.color_palette(color_list)

    # Plot averages on ax1
    sns.lineplot(
        data=avg_df,
        x='hour',
        y=q_col,
        hue='month',
        ax=ax1,
        palette=custom_palette,
    )
    ax1.set_title("Average")

    # Plot maxima on ax2
    sns.lineplot(
        data=max_df,
        x='hour',
        y=q_col,
        hue='month',
        ax=ax2,
        palette=custom_palette,
    )
    ax2.set_title("Maximum")

    for ax in axes:
        # Customize the plot
        ax.set_xlabel("Hour of Day", fontsize=12)
        ax.set_ylabel(f"{q_col} [{unit}]", fontsize=12)
        ax.legend(title="Month", loc='upper left', fontsize=10)
        ax.grid(True)

    return axes

@attrs.define
class PanelConfig:
    n: int
    azimuth: int
    inclination: int



# Example usage:
# df['datetime'] = pd.date_range(start='2023-01-01 00:00', end='2023-01-02 00:00', freq='1H')
# config = PanelConfig(n=10, azimuth=180, inclination=30)
# production = compute_clear_sky_production(df['datetime'], 48.8566, 2.3522, config)  # Paris
# print(production)


# Example usage:
# fig, ax = plt.subplots()
# plot_monthly_hourly_averages(df, 'Date', 'Q', 'blue', ax=ax)
# plt.show()

# Example usage:
# df = read_csv_to_dataframe("your_file.csv")
# print(df.head())
# print(column_shortcuts)


def fve_plots(df, units, file_prefix):
    for col in ['irr_tilt', 'irr_eff', 'irr_global_pv', 'pv_energy_DC', 'pv_net_energy']:
        col_e = col + '_e'
        col_w = col + '_w'
        if col_e in df.columns:
            unit_e = units.get(col_e, "-")
            unit_w = units.get(col_w, "-")
            assert unit_e == unit_w, f"Units for {col} are different: {unit_e} vs {unit_w}"
            fig, axes = plt.subplots(2, 2, figsize=(10, 6))
            plot_monthly_hourly_averages(df, col_e, unit_e, axes=axes[0])
            plot_monthly_hourly_averages(df, col_w, unit_w, axes=axes[1])
            fig.savefig(workdir / (file_prefix + col + ".pdf"))
        else:
            unit = units.get(col, "-")
            fig, axes = plt.subplots(1, 2, figsize=(10, 6))
            plot_monthly_hourly_averages(df, col, unit, axes=axes)
            fig.savefig(workdir / (file_prefix + col + ".pdf"))


def main():
    #df_spot = get_spot_price()

    fname = "FVE_Kadlec_strisky_SIM_241115.csv"
    lat, lon = 50.7663, 15.0543  # Liberec

    # vertical panels
    panel_groups = {
        # inclination of panel, i.e. normal is 90 - inclination
        'East': PanelConfig(n=10, azimuth=120, inclination=90),
        'West': PanelConfig(n=10, azimuth=210, inclination=90),
    }

    # vertical panels
    panel_groups = {
        # inclination of panel, i.e. normal is 90 - inclination
        'East': PanelConfig(n=18, azimuth=120, inclination=5),
        'West': PanelConfig(n=18, azimuth=210, inclination=5),
    }

    df, units = read_csv_to_dataframe(fname)
    df, units = create_shortcuts_and_rename(df, units)
    fve_plots(df, units, "Kadlec_")
    #
    # horizon = make_horizon()
    #
    # fig, axes = plt.subplots(len(panel_groups), 2, figsize=(10, 10))
    # for (name, cfg), ax in zip(panel_groups.items(), axes):
    #     col_name = name + '_group_cos'
    #     #df[name+'_panel_cos'] = compute_sun_normal_angle_with_pvlib(df, lat, lon, cfg.azimuth, cfg.inclination)
    #     #df[col_name] = compute_clear_sky_production(df.date_time, lat, lon, cfg)
    #     df[col_name] = compute_production(df, lat, lon, cfg)
    #     plot_monthly_hourly_averages(df, 'date_time', col_name, 'blue', axes=ax)
    #
    # print(df.columns)
    # print(df.head())
    # fig.savefig(workdir / "cleansky_horizon.pdf")
    # #plt.show()

if __name__ == "__main__":
    main()
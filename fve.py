import attrs
import pandas as pd
import numpy as np
import seaborn as sns
import math
from datetime import datetime, timedelta
from pysolar.solar import get_azimuth, get_altitude
import pvlib

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
    solar_pos = pvlib.solarposition.get_solarposition(
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
        df = df.iloc[17:]  # Start data from the 20th row (index=19 in the original file)
        df.reset_index(drop=True, inplace=True)  # Reset index after slicing
        print(f"Data successfully loaded. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None


def create_shortcuts_and_rename(df):
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
        "irr_horiz": "Intenzita záření na horizontálu",
        "irr_diff": "Difúzní záření na vodorovné rovině",
        "temp_out": "Venkovní teplota",
        "sun_alt_w": "Budovy 02-Oblast modulu Západ: Výška slunce",
        "irr_tilt_w": "Budovy 02-Oblast modulu Západ: Intenzita záření na skloněnou plochu",
        "temp_mod_w": "Budovy 02-Oblast modulu Západ: Teplota modulu",
        "sun_alt_e": "Budovy 02-Oblast modulu Východ: Výška slunce",
        "irr_tilt_e": "Budovy 02-Oblast modulu Východ: Intenzita záření na skloněnou plochu",
        "temp_mod_e": "Budovy 02-Oblast modulu Východ: Teplota modulu",
        "ac_current": "Proud (AC) Střídač 1 (GoodWe Technologies Co.,Ltd. GW15K-ET)",
        "voc_1": "Napětí naprázdno (MPP 1, Střídač 1 (GoodWe Technologies Co.,Ltd. GW15K-ET))",
        "vmp_1": "Napětí MPP (MPP 1, Střídač 1 (GoodWe Technologies Co.,Ltd. GW15K-ET))",
        "voc_2": "Napětí naprázdno (MPP 2, Střídač 1 (GoodWe Technologies Co.,Ltd. GW15K-ET))",
        "vmp_2": "Napětí MPP (MPP 2, Střídač 1 (GoodWe Technologies Co.,Ltd. GW15K-ET))",
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
        "irr_global_horiz_w": "Budovy 02-Oblast modulu Západ: Globální záření - horizontální",
        "spec_loss_w": "Budovy 02-Oblast modulu Západ: Ztráty kvůli Odchylka od standardního spektra",
        "albedo_loss_w": "Budovy 02-Oblast modulu Západ: Ztráty kvůli Odraz od země (Albedo)",
        "module_yield_w": "Budovy 02-Oblast modulu Západ: Výnosy Vyrovnání a sklon úrovně modulu",
        "shade_loss_w": "Budovy 02-Oblast modulu Západ: Ztráty kvůli Odstínění, zaclonění",
        "module_refl_w": "Budovy 02-Oblast modulu Západ: Odraz na povrchu modulu",
        "irr_back_w": "Budovy 02-Oblast modulu Západ: Intenzita záření na zadní části modulu",
        "irr_global_mod_w": "Budovy 02-Oblast modulu Západ: Globální záření na modul",
        "irr_global_horiz_e": "Budovy 02-Oblast modulu Východ: Globální záření - horizontální",
        "spec_loss_e": "Budovy 02-Oblast modulu Východ: Ztráty kvůli Odchylka od standardního spektra",
        "albedo_loss_e": "Budovy 02-Oblast modulu Východ: Ztráty kvůli Odraz od země (Albedo)",
        "module_yield_e": "Budovy 02-Oblast modulu Východ: Výnosy Vyrovnání a sklon úrovně modulu",
        "shade_loss_e": "Budovy 02-Oblast modulu Východ: Ztráty kvůli Odstínění, zaclonění",
        "module_refl_e": "Budovy 02-Oblast modulu Východ: Odraz na povrchu modulu",
        "irr_back_e": "Budovy 02-Oblast modulu Východ: Intenzita záření na zadní části modulu",
        "irr_global_mod_e": "Budovy 02-Oblast modulu Východ: Globální záření na modul",
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
        "grid_feed": "Dodávky energie do sítě",
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
    df['date_time'] = pd.to_datetime(
        df['time'].str.replace(' ', ''),  # Remove spaces for cleaner parsing
        format='%d.%m.%H:%M'
    )

    # Extract hour and month from the parsed datetime
    df['hour'] = df['date_time'].dt.hour
    df['month'] = df['date_time'].dt.month

    return df, shortcuts




def plot_monthly_hourly_averages(df, date_col, q_col, color, ax=None):
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
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Use Seaborn lineplot with confidence interval
    sns.lineplot(
        data=df,
        x='hour',
        y=q_col,
        hue='month',
        palette=sns.color_palette([color] * 12),
        #errorbar='sd',  # Use standard deviation for the shaded area
        ax=ax
    )

    # Customize the plot
    ax.set_title(f"Hourly Averages and Standard Deviations of '{q_col}' by Month", fontsize=14)
    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel(f"Average {q_col}", fontsize=12)
    ax.legend(title="Month", loc='upper left', fontsize=10)
    ax.grid(True)

    return ax

@attrs.define
class PanelConfig:
    n: int
    azimuth: int
    inclination: int

# Example usage:
# fig, ax = plt.subplots()
# plot_monthly_hourly_averages(df, 'Date', 'Q', 'blue', ax=ax)
# plt.show()

# Example usage:
# df = read_csv_to_dataframe("your_file.csv")
# print(df.head())
# print(column_shortcuts)

def main():
    fname = "FVE_Kadlec_strisky_SIM_241115.csv"
    lat, lon = 50.7663, 15.0543  # Liberec
    panel_groups = {
        # inclination of panel, i.e. normal is 90 - inclination
        'East': PanelConfig(n=10, azimuth=120, inclination=90),
        'West': PanelConfig(n=10, azimuth=210, inclination=90),
    }

    df = read_csv_to_dataframe(fname)
    df, column_shortcuts = create_shortcuts_and_rename(df)
    #plot_monthly_hourly_averages(df, 'time', 'irr_tilt_e', 'blue')
    #plt.show()

    for name, cfg in panel_groups.items():
        df[name+'_panel_cos'] = compute_sun_normal_angle_with_pvlib(df, lat, lon, cfg.azimuth, cfg.inclination)
        df[name+'_group_cos'] = cfg.n * compute_sun_normal_angle_with_pvlib(df, lat, lon, cfg.azimuth, cfg.inclination)

    print(df.columns)
    print(df.head())
    plot_monthly_hourly_averages(df, 'time', 'East_panel_cos', 'blue')
    plt.show()
    plot_monthly_hourly_averages(df, 'time', 'West_panel_cos', 'blue')
    plt.show()

if __name__ == "__main__":
    main()
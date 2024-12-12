from typing import List, Tuple, Dict
import json

import attrs
import pandas as pd
import numpy as np
import seaborn as sns
import math
from datetime import datetime, timedelta

import pvlib_model
import pickle
#from pysolar.solar import get_azimuth, get_altitude
from pvlib_model import *
from spot import *
from zoom import zoom_plot_df

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
    cols_out = {'irr_tilt':'sum', 'irr_eff':'sum', 'irr_global_pv':'sum', 'pv_energy_DC':'sum', 'pv_net_energy':'sum'}
    cols_full = {}
    for col in cols_out.keys():
        col_e = col + '_e'
        col_w = col + '_w'
        for c in [col, col_e, col_w]:
            if c in df.columns:
                cols_full[c] = cols_out[col]
        if col_e in df.columns:
            unit_e = units.get(col_e, "-")
            unit_w = units.get(col_w, "-")
            assert unit_e == unit_w, f"Units for {col} are different: {unit_e} vs {unit_w}"
            fig, axes = plt.subplots(2, 2, figsize=(10, 6))
            plot_monthly_hourly_averages(df, col_e, unit_e, axes=axes[0])
            plot_monthly_hourly_averages(df, col_w, unit_w, axes=axes[1])
            fig.savefig(workdir / (file_prefix + col + ".pdf"))
            plt.close(fig)
        elif col in df.columns:
            unit = units.get(col, "-")
            fig, axes = plt.subplots(1, 2, figsize=(10, 6))
            plot_monthly_hourly_averages(df, col, unit, axes=axes)
            fig.savefig(workdir / (file_prefix + col + ".pdf"))
            plt.close(fig)

    # print month sums
    print_df = df[cols_full.keys()].groupby(df['month']).agg(cols_full)
    print_df.to_csv(workdir / (file_prefix + "month_sums.csv"))

def concat_columns(df1, df2, cols, idx=None):
    """
    Extracts a specified column from two dataframes, optionally sets an index,
    and concatenates them into a new dataframe.

    Parameters:
        df1 (pd.DataFrame): The first dataframe.
        df2 (pd.DataFrame): The second dataframe.
        col (str): The column to extract and concatenate.
        idx (str, optional): Column to use as index before concatenation. Defaults to None.

    Returns:
        pd.DataFrame: The concatenated dataframe.
    """
    # Extract the specified column
    concat_cols = []
    for col in cols:
        if idx in df1 and idx in df2:
            df1_col = df1.set_index(df1[idx])[col]
            df2_col = df2.set_index(df2[idx])[col]
        else:
            raise ValueError(f"Index column '{idx}' not found in both dataframes.")
        diff =  pd.Series(df1_col - df2_col, name=f"{col}_diff")

        df1_col.name = f"{col}_df1"
        df2_col.name = f"{col}_df2"
        concat_cols.extend([df1_col, df2_col, diff])

    # Concatenate the two columns into a new dataframe
    concatenated_df = pd.concat(concat_cols, axis=1)
    return concatenated_df


def estimate_time_shift(ts1, ts2):
    """
    Estimate the time shift (dt) between two time series using derivatives.

    Parameters:
        ts1 (pd.Series): First time series.
        ts2 (pd.Series): Second time series.

    Returns:
        float: Estimated time shift (dt).
    """
    # Compute first differences (derivatives)
    d_f1 = np.diff(ts1.values)
    d_f2 = np.diff(ts2.values)

    # Compute mean derivative
    d_mean = (d_f1 + d_f2) / 2

    # Mask where abs(d_mean) is greater than the lower quartile
    mask = np.abs(d_mean) > 0.01

    # Select valid points for f1, f2, and d_mean (accounting for the difference size due to np.diff)
    f1_values = ts1.values[1:]  # First differences reduce the array size by 1
    f2_values = ts2.values[1:]

    f1_f2_diff = f1_values[mask] - f2_values[mask]
    d_mean_masked = d_mean[mask]

    # Estimate the time shift
    dt = np.mean(f1_f2_diff / d_mean_masked)

    return dt


# Apply the fractional shift to ts2
def apply_fractional_shift(ts, shift):
    # Shift the index by the fractional amount
    shifted_index = ts.index + pd.to_timedelta(shift, unit="H")

    # Interpolate back to the original index
    ts_shifted = ts.reindex(ts.index.union(shifted_index)).interpolate(method='cubicspline').reindex(ts.index)
    return ts_shifted


def plot_2d_data(x, y, z):
    print(f"X range: {np.min(x)}, {np.max(x)}")
    print(f"Y range: {np.min(y)}, {np.max(y)}")
    print(f"Z range: {np.min(z)}, {np.max(z)}")
    # Plot function and sample points
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    #c = ax.contourf(x, y, z, 15, cmap=plt.cm.RdBu);
    scatter = ax.scatter(x, y, marker='.', c=z, cmap=plt.cm.RdBu)
    #ax.set_ylim(-1, 1)
    #ax.set_xlim(-1, 1)
    ax.set_xlabel(r"$x$", fontsize=20)
    ax.set_ylabel(r"$y$", fontsize=20)
    cb = fig.colorbar(scatter, ax=ax)
    cb.set_label(r"$z$", fontsize=20)
    sc2 =ax2.scatter(x, z, c=y, marker='.', label='x', cmap=plt.cm.RdBu)
    cb2 = fig.colorbar(sc2, ax=ax2)
    plt.show()

def approx_2d(in1, in2, out=None):
    """
    If 'out' is given:
        - Construct approximation
        - write down approximationg function
        - apply to inputs
    else:
        - read approximation
        - apply to inputs
    :param in1:
    :param in2:
    :param out:
    :return:
    """
    points = np.stack([in1, in2], axis=1)
    func_file = workdir / "dc_approx"
    if out is None:
        with open(func_file, 'rb') as f:
            func = pickle.load(f)
    else:
        values = out
        func = interpolate.CloughTocher2DInterpolator(points, values, tol=1e-2, rescale=True)
        with open(func_file, 'wb') as f:
            pickle.dump(func, f)
    return func(points), func

@attrs.define
class approx_fun:
    basis: List[Tuple[int, int]]
    coeffs: np.ndarray

    def __call__(self, X, y):
        # Define the callable object
        X = np.atleast_2d(X)
        X, y = np.asarray(X), np.asarray(y)
        if X.shape[1] != y.shape[0]:
            raise ValueError("x and y must have the same shape.")
        # Evaluate the polynomial
        return sum(c * np.sum(X ** px, axis=0) * (y ** py) for c, (px, py) in zip(self.coeffs, self.basis))


def least_squares_fit(X, Y, Z, basis):
    """
    Perform a least squares fit for Z as a function of X and Y using a specified polynomial basis.

    Parameters:
        X (array-like): Input X values.
        Y (array-like): Input Y values.
        Z (array-like): Output Z values (target).
        basis (list of tuple): List of (px, py) pairs where px and py are powers of X and Y.

    Returns:
        callable: A function fun(x, y) for evaluating the fitted surface.
    """
    # Validate input
    X = np.atleast_2d(X)   # (2, N)
    Y, Z = np.asarray(Y), np.asarray(Z) #(N,)
    if X.shape[1] != Y.shape[0] or X.shape[1] != Z.shape[0]:
        raise ValueError("X, Y, and Z must have the same shape.")

    # Build the design matrix
    A = np.column_stack([np.sum(X**px,axis=0) * Y**py for px, py in basis])

    # Solve the least squares problem
    coeffs, _, _, _ = np.linalg.lstsq(A, Z, rcond=None)
    return approx_fun(basis, coeffs)

def lin_2d_approx(in1, in2, out=None):
    base = [(0,0), (1,0), (0,1), (1,1), (2, 0), (0, 2)]
    func = least_squares_fit(in1, in2, out, base)
    return func
    # X = np.stack([np.ones(len(in1)), in1, in2, in1**2, in2**2, in1*in2], axis=1)
    # func_file = workdir / "dc_approx"
    # if out is None:
    #     with open(func_file, 'rb') as f:
    #         beta = pickle.load(f)
    # else:
    #     beta = np.linalg.lstsq(X.T, out)[0]
    # func = lambda dc, temp : beta[0] + beta[1] * dc + beta[2] * temp
    # return func(in1, in2), func

def main():




    # Reference calculation
    fname = "FVE_Kadlec_strisky_SIM_241115.csv"
    df, units = read_csv_to_dataframe(fname)
    df, units = create_shortcuts_and_rename(df, units)
    df.to_csv(workdir / "Kadlec_df.csv")
    fve_plots(df, units, "Kadlec_")

    # Own model
    # vertical panels
    panel_groups = {
        # inclination of panel, i.e. normal is 90 - inclination
        'East': PanelConfig(n=18, azimuth=120, inclination=5),
        'West': PanelConfig(n=18, azimuth=210, inclination=5),
    }

    # Extract the named index and selected columns into a new DataFrame
    columns_to_extract = ["date_time", "hour", "month", "DHI", "DNI", "GHI", "temp_out", "temp_mod_w", "temp_mod_e"]
    new_df = df[columns_to_extract].reset_index()
    model_df = pvlib_model.pv_model(panel_groups, new_df)
    model_df.to_csv(workdir / "model_df.csv")
    fve_plots(model_df, units, "JB_")

    ################################################
    # Perform linear fit
    fit_col = 'irr_global_pv'
    x = model_df[fit_col]
    y = df[fit_col]
    print(f"Squared error[{fit_col}: {np.sqrt(np.sum((np.array(y) - np.array(x))**2)/len(x))}")

    coefficients = np.polyfit(x, y, deg=1)
    slope, intercept = coefficients
    print(f"Linear fit coefficients: slope={slope}, intercept={intercept}")

    # Predict y_mod using the linear fit
    y_mod = slope * x + intercept
    dt = estimate_time_shift(y_mod, y)
    print(f"Estimated time shift: {dt:.2f} hours)")
    y_mod = apply_fractional_shift(pd.Series(y_mod, index=x.index), dt)
    print(f"Squared error [{fit_col}] fitted: {np.sqrt(np.sum((np.array(y) - np.array(y_mod))**2)/len(x))}")

    fit_col_new = fit_col + "_mod"
    df[fit_col_new] = df[fit_col]
    model_df[fit_col_new] = y_mod

    ##################################
    # Nonlinear fit  Kadlec pv_energy_DC as function of model pv_energy_DC to model irr_global_pv and temparature



    # x = model_df['pv_energy_DC']
    irr_w = np.array(model_df['irr_eff_w'])
    irr_e = np.array(model_df['irr_eff_e'])
    model_dc = np.array(model_df['pv_energy_DC'])
    kadlec_dc = np.array(df['pv_energy_DC'])
    temp_out = np.array(df['temp_out'])

    # mask = np.logical_and(model_dc > 1e-2,  kadlec_dc > 1e-2)
    # assert len(kadlec_dc) == len(model_dc), "Lengths of Kadlec and model dataframes do not match"
    # assert len(kadlec_dc) == len(temp_out), "Lengths of Kadlec and temperature dataframes do not match"
    #
    # #plot_2d_data(model_dc, temp_out, kadlec_dc - model_dc)
    #
    # frac = kadlec_dc[mask]/model_dc[mask]
    # coefficients = np.polyfit(temp_out[mask], frac, deg=2)
    # lin_model_dc = lambda T : coefficients[2] + coefficients[1] * T + coefficients[0] * T**2
    # # plt.scatter(temp_out[mask], frac)
    # # T = np.linspace(np.min(temp_out), np.max(temp_out), 100)
    # # plt.plot(T, lin_model_dc(T), c='red')
    # # plt.show()
    # print(f"Temperature model fit coefficients: {coefficients}")
    # # dc_mod = slope * x + intercept
    # model_dc_non_lin_T = lin_model_dc(temp_out) * model_dc
    # #plot_2d_data(model_dc, temp_out, kadlec_dc - model_dc_non_lin_T)
    #
    # # Linear fit to handle extrapolation
    # lin_func = lin_2d_approx(model_dc, temp_out, out=kadlec_dc)
    # model_dc_lin = lin_func(model_dc, temp_out)
    # # plot_2d_data(model_dc, temp_out, kadlec_dc - model_dc_lin)
    # #model_dc_new, dc_func = approx_2d(model_dc, temp_out, out=kadlec_dc)
    #
    # coefficients = np.polyfit(model_dc, kadlec_dc, deg=1)
    # slope, intercept = coefficients
    # print(f"Linear fit coefficients: slope={slope}, intercept={intercept}")
    # dc_mod_lin_X = slope * model_dc + intercept
    # # plot_2d_data(model_dc, temp_out, kadlec_dc - dc_mod_lin_X)

    # Least sq . for temperature, west and east irrad.
    # We want to get model that is independent of angle, so it must be:
    # total = (X_w  + X_e) @ beta
    base = [(0,0), (1,0), (0,1), (1,1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2), (3, 0), (0, 3)]
    func_we = least_squares_fit((irr_e, irr_w), temp_out, kadlec_dc, base)
    dc_mod_we = func_we((irr_e, irr_w), temp_out)
    plot_2d_data(irr_e, temp_out, kadlec_dc - dc_mod_we)
    plot_2d_data(irr_w, temp_out, kadlec_dc - dc_mod_we)

    func_file = workdir / "dc_approx"
    with open(func_file, 'wb') as f:
        pickle.dump(func_we, f)
    model_df['pv_energy_DC'] = dc_mod_we
    cmp_df = concat_columns(df, model_df, ['irr_global_pv', 'pv_energy_DC'], 'date_time')
    zoom_plot_df(cmp_df)


def optimize():
    # Plot together production and spot
    df_spot = get_spot_price([2023])
    print(df_spot.head())

    weather = pvlib_model.get_weather(df_spot['date_time'])
    df_spot.set_index('date_time', inplace=True)
    # interpolate to common times
    interpolated_weather = weather\
        .reindex(weather.index.union(df_spot.index))\
        .interpolate(method='cubicspline').loc[df_spot.index]

    # Combine spot prices and interpolated weather data
    input_df = pd.concat([df_spot, interpolated_weather], axis=1)

    panel_groups = {
        # inclination of panel, i.e. normal is 90 - inclination
        'East': PanelConfig(n=18, azimuth=120, inclination=5),
        'West': PanelConfig(n=18, azimuth=210, inclination=5),
    }

    # model_df = pvlib_model.pv_model(panel_groups, new_df)
    # model_df.to_csv(workdir / "model_df.csv")
    # fve_plots(model_df, units, "JB_")

    zoom_plot_df(input_df)


if __name__ == "__main__":
    #main()
    optimize()
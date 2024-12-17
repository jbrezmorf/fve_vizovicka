import itertools
import shutil

import numpy as np

import pvlib_model
import pickle

import fit_models as  fit_mod
from import_model import read_model_xlsx
from operation import model_operation, OperationCfg, model_operation_lp
from plot_days import fve_plots
#from pysolar.solar import get_azimuth, get_altitude
from pvlib_model import *
from spot import *
from zoom import zoom_plot_df

import pandas as pd

workdir = cfg.workdir

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
        if idx is not None:
            df1.reset_index(inplace=True)
            df2.reset_index(inplace=True)
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


# Apply the fractional shift to ts2


def main():

    # Reference calculation
    fname = "FVE_Kadlec_strisky_SIM_241115.csv"
    cols_out, df, units = read_model_xlsx(fname, workdir)

    # Own model
    # vertical panels
    panel_groups = {
        # inclination of panel, i.e. normal is 90 - inclination
        'East': PanelConfig(n=18, azimuth=120, inclination=5),
        'West': PanelConfig(n=18, azimuth=210, inclination=5),
    }

    # Extract the named index and selected columns into a new DataFrame
    columns_to_extract = ["date_time", "sun_elevation",  "DHI", "DNI", "GHI", "temp_out", "temp_mod_w", "temp_mod_e"]
    new_df = df[columns_to_extract]
    # sun_elevation,  DHI, GHI
    model_df = pvlib_model.pv_model(panel_groups, new_df, horizon=False)
    model_df.to_csv(workdir / "model_df.csv")
    fve_plots(workdir, model_df, units, "JB_", cols_out)

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
    dt = fit_mod.estimate_time_shift(y_mod, y)
    print(f"Estimated time shift: {dt:.2f} hours)")
    y_mod = fit_mod.apply_fractional_shift(pd.Series(y_mod, index=x.index), dt)
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
    # #fit_mod.plot_2d_data(model_dc, temp_out, kadlec_dc - model_dc)
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
    # #fit_mod.plot_2d_data(model_dc, temp_out, kadlec_dc - model_dc_non_lin_T)
    #
    # # Linear fit to handle extrapolation
    # lin_func = lin_2d_approx(model_dc, temp_out, out=kadlec_dc)
    # model_dc_lin = lin_func(model_dc, temp_out)
    # # fit_mod.plot_2d_data(model_dc, temp_out, kadlec_dc - model_dc_lin)
    # #model_dc_new, dc_func = approx_2d(workdir, model_dc, temp_out, out=kadlec_dc)
    #
    # coefficients = np.polyfit(model_dc, kadlec_dc, deg=1)
    # slope, intercept = coefficients
    # print(f"Linear fit coefficients: slope={slope}, intercept={intercept}")
    # dc_mod_lin_X = slope * model_dc + intercept
    # # fit_mod.plot_2d_data(model_dc, temp_out, kadlec_dc - dc_mod_lin_X)

    # Least sq . for temperature, west and east irrad.
    # We want to get model that is independent of angle, so it must be:
    # total = (X_w  + X_e) @ beta
    base = [(0,0), (1,0), (0,1), (1,1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2), (3, 0), (0, 3)]
    func_we = fit_mod.least_squares_fit((irr_e, irr_w), temp_out, kadlec_dc, base)
    dc_mod_we = func_we((irr_e, irr_w), temp_out)
    fit_mod.plot_2d_data(irr_e, temp_out, kadlec_dc - dc_mod_we)
    fit_mod.plot_2d_data(irr_w, temp_out, kadlec_dc - dc_mod_we)

    func_file = workdir / "dc_approx"
    with open(func_file, 'wb') as f:
        pickle.dump(func_we, f)
    model_df['pv_energy_DC'] = dc_mod_we
    cmp_df = concat_columns(df, model_df, ['irr_global_pv', 'pv_energy_DC'], 'date_time')
    fve_plots(workdir, model_df, units, "JB_fix_", cols_out)
    zoom_plot_df(cmp_df)



def operation_cfg(**kw_args):
    bat_cap = kw_args.get('bat_cap', 14.33)
    eff_cap = kw_args.get('bat_eff_cap', 13.5)
    slack = (bat_cap - eff_cap) / 2 / bat_cap
    op_cfg=dict(
        year_consumption=11200,
        heat_consumption=2500,
        TV_consumption=2500,
        heat_temp_out=10,
        heat_temp_in=21,
        peak_power=20,
        bat_min_charge=slack,
        bat_max_charge=1.0 - slack,
        bat_cap=bat_cap,               # [kWh] setting unrealistic high capacity, turn effectively off the day sim model
        conv_eff= 0.95,             # [-] DC/AC conversion efficiency
        distr_sell_fee=0.5445,      # [Kč/kWh] (s DPH) sell distribution fee per kWh
        distr_buy_fee=0.5445,       # [Kč/kWh] (s DPH) buy distribution fee per kWh
        distr_per_day=5.05,         # [Kč / day] s DPH, : float  # distribution fee per day
        eur_to_czk=25,              #  exchange rate
        sell_limit=10,               # [kWh / h] maximum amount of kWh that can be sold in a single hour
        heating_mean_period=2 * 24
        # buy_price_limit=: float  # maximum amount of kWh that can be bought in a single hour
    )
    op_cfg.update(kw_args)
    op_cfg.pop('bat_eff_cap', None)

    return OperationCfg(**op_cfg)


def get_input_df():
    # Plot together production and spot
    spot_df = get_spot_price([2023])    # EUR / MWh
    print(spot_df.head())
    kadlec_df = pd.read_csv(workdir / "Kadlec_df.csv", index_col=0)
    spot_df['raw_consumption'] = np.array(kadlec_df['consumption'])

    # Weather in UTC
    weather = pvlib_model.get_weather(spot_df['date_time'])

    # Combine spot prices and interpolated weather data
    input_df = weather

    # SPot and consumption in CET
    #input_df['spot'] = df_spot
    #input_df['spot_date_time'] = df_spot.index
    #pd.concat([df_spot, interpolated_weather], axis=1)

    #print("Total consumption:", np.sum(input_df['consumption']))
    input_df['GHI'] = (input_df['Ibeam'] + input_df['Idiff']) / 1000 # kW/m2
    input_df['DHI'] = (input_df['Idiff']) / 1000 # kW/m2
    input_df.reset_index(inplace=True)
    return input_df, spot_df


def simulate(workdir, input_df, spot_df, panel_groups, op_cfg: OperationCfg, plot=False):

    model_df = pvlib_model.pv_model(panel_groups, input_df, horizon=True)

    # merge with spot_df
    # 1. shift model from UTC to CET
    model_df.reset_index(inplace=True)
    model_df['date_time'] = model_df['date_time'] + pd.Timedelta(hours=1)
    model_df.set_index('date_time', inplace=True)
    model_df.index = pd.to_datetime(model_df.index)
    # 2. interpolate to common times
    # interpolate to common times

    spot_df.set_index('date_time', inplace=True)
    spot_df.index = pd.to_datetime(spot_df.index)
    interpolated_model = model_df\
        .reindex(model_df.index.union(spot_df.index))\
        .interpolate(method='cubicspline').loc[spot_df.index]
    model_df = pd.concat([interpolated_model, spot_df], axis=1)
    # Replace all NaN values with 0
    model_df.fillna(0, inplace=True)
    for col in ['raw_consumption', 'pv_energy_DC']:
        model_df[col] = np.maximum(0.0, model_df[col])  # fix oscilations

    #3. corelate consumption with temperature
    model_df['consumption'] = model_df['raw_consumption']
    total_consumption = np.sum(model_df['consumption'])

    est_total_consumption = op_cfg.year_consumption   # kWh
    est_heating = op_cfg.heat_consumption
    est_TV = op_cfg.TV_consumption
    est_other = est_total_consumption - est_heating - est_TV

    other_consumption = model_df['consumption'] / total_consumption * est_other
    heating_temp_factor = (op_cfg.heat_temp_in- model_df['temp_out']) * (model_df['temp_out'] < op_cfg.heat_temp_out)
    heating_consumption = heating_temp_factor * est_heating / np.sum(heating_temp_factor)
    TV_consumption = np.full_like(heating_consumption, est_TV / len(heating_consumption))
    model_df['consumption'] = other_consumption + heating_consumption + TV_consumption
    model_df['heat_consumption'] = heating_consumption + TV_consumption
    model_df['other_consumption'] = other_consumption
    #model_df.set_index('date_time', inplace=True)


    #irr_cols = ['Ibeam', 'Idiff', 'irr_eff_w', 'poa_global_West', 'poa_direct_West', 'poa_diffuse_West', 'poa_sky_diffuse_West',
    #   'poa_ground_diffuse_West']
    # irr_cols = ['Ibeam', 'Idiff', 'irr_eff_e', 'poa_global_East', 'poa_direct_East', 'poa_diffuse_East', 'poa_sky_diffuse_East',
    #    'poa_ground_diffuse_East']
    # if plot:
    #     zoom_plot_df(model_df[irr_cols])
    # plot_df = fve_plots(workdir, model_df, {}, "irr_w_", irr_cols)

    func_file = workdir / "dc_approx"
    with open(func_file, 'rb') as f:
        func_we = pickle.load(f)
    irr_w = np.array(model_df['irr_eff_w'])
    irr_e = np.array(model_df['irr_eff_e'])
    temp_out = np.array(model_df['temp_out'])
    dc_mod_we = func_we((irr_e, irr_w), temp_out)
    model_df['pv_energy_DC'] = np.maximum(0.0, dc_mod_we)

    peak = np.percentile(model_df['pv_energy_DC'], 98)    # cut possible outlayers
    factor = op_cfg.peak_power / (op_cfg.conv_eff * peak)
    model_df['pv_energy_DC'] = factor * model_df['pv_energy_DC']
    #operation_df = model_operation(model_df, op_cfg)
    operation_df = model_operation_lp(model_df, op_cfg)


    operation_df.to_csv(workdir / "opt_model_df_2023.csv")
    operation_df.idex = pd.to_datetime(operation_df.index)
    #cols_out = {'price':'sum', 'consumption':'sum', 'ghi':'sum', 'dhi':'sum', 'irr_global_pv':'sum', 'irr_eff':'sum', 'pv_energy_DC':'sum'}
    cols_out = {'price':'sum', 'consumption':'sum', 'pv_energy_DC':'sum', 'sell':'sum', 'irr_global_pv':'sum', 'revenue':'sum', 'bat_energy':['min', 'max']}
    plot_df = fve_plots(workdir, operation_df, {}, "opt_", cols_out)
    if plot:
        zoom_plot_df(plot_df)
    return operation_df


def single_run():
    input_df, spot_df = get_input_df()
    # panel_groups = {
    #     # inclination of panel, i.e. normal is 90 - inclination
    #     'East': PanelConfig(n=18, azimuth=90, inclination=5),
    #     'West': PanelConfig(n=18, azimuth=210, inclination=5),
    # }
    panel_groups = {
        # inclination of panel, i.e. normal is 90 - inclination
        'East': PanelConfig(n=18, azimuth=90, inclination=5),
        'West': PanelConfig(n=18, azimuth=210, inclination=5),
    }
    simulate(workdir, input_df, spot_df, panel_groups, operation_cfg(), plot=True)

# Function to write with aligned spaces
def append_with_alignment(df, file_path):
    # Calculate the maximum width for each column
    column_widths = {col: max(len(col), df[col].astype(str).str.len().max()) for col in df.columns}

    # Format the header
    header = " ".join([f"{col:<{column_widths[col]}}" for col in df.columns])

    # Format the rows
    rows = "\n".join([
        " ".join([f"{str(value):<{column_widths[col]}}" for col, value in row.items()])
        for _, row in df.iterrows()
    ])

    # Write the header if the file doesn't exist or is empty
    try:
        with open(file_path, 'r') as f:
            if not f.read().strip():  # Empty file
                write_header = True
            else:
                write_header = False
    except FileNotFoundError:
        write_header = True

    # Append to the file
    with open(file_path, 'a') as f:
        if write_header:
            f.write(header + "\n")
        f.write(rows + "\n")


def optimize():
    input_df = get_input_df()
    cases = []
    # params = itertools.chain(
    #     itertools.product([True, False], [90,  110, 130,  140, 160, 180], [5, 15, 35, 50], [210], [5]),
    #     #itertools.product([True, False], [120], [50], [180, 190, 200, 210], [5, 10, 20, 30, 40, 50])
    # )
    params = itertools.chain(
        itertools.product([True, False], [110], [50], [130, 150, 170, 190,  210], [5, 15, 35, 50])
    )
    for bat, az1, tl1, az2, tl2 in params:

        case_dir = workdir / f"case_{az1}_{tl1}_{az2}_{tl2}_{bat}"
        if not case_dir.exists():
            case_dir.mkdir()
        shutil.copyfile(workdir / "dc_approx", case_dir / "dc_approx")
        print(case_dir)
        panel_groups = {
            'East': PanelConfig(n=18, azimuth=az1, inclination=tl1),
            'West': PanelConfig(n=18, azimuth=az2, inclination=tl2),
        }
        case_dir = workdir / f"case_{az1}_{tl1}_{az2}_{tl2}_{bat}"
        op_df = simulate(case_dir, input_df, panel_groups, operation_cfg(), plot=False)
        revenue_sum = op_df['revenue'].sum()
        peak_power = op_df['pv_energy_DC'].max() * 0.95
        total_dc = op_df['pv_energy_DC'].sum()
        total_irr = op_df['irr_global_pv'].sum()
        sell = op_df['sell']
        total_sell = sell[sell > 0].sum()
        total_buy = sell[sell < 0].sum()
        print("  Peak:", peak_power)
        print("  Revenue: ", revenue_sum)

        cases.append(((revenue_sum, peak_power, total_dc, total_irr, total_buy, total_sell, bat, az1, tl1, az2, tl2)))
    cases.sort(reverse=True)
    cases_df = pd.DataFrame(cases, columns=['revenue', 'peak_power', 'total_dc', 'total_irr', 'total_buy', 'total_sell', 'bat', 'az1', 'tl1', 'az2', 'tl2'])
    # Append to CSV with aligned spaces
    append_with_alignment(cases_df, workdir / "cases.csv")

if __name__ == "__main__":
    #main()
    single_run()
    #optimize()
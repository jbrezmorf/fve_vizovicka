import numpy as np
import pandas as pd
from plot_days import fve_plots


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
        "sun_elevation": df["sun_alt_w"],
        'DNI': (df["GHI"] - df["DHI"]) / np.cos(np.radians(zenith)),
    })
    df = pd.concat([df, new_cols], axis=1)

    # rename units keys
    units_dict = {k.strip(' '): v for k, v in units.to_dict().items()}
    units_dict = {reverse_mapping[k]:v for k,v in units_dict.items() if k in reverse_mapping}
    return df, units_dict


def read_model_xlsx(fname, workdir):
    df, units = read_csv_to_dataframe(fname)
    df, units = create_shortcuts_and_rename(df, units)
    df.to_csv(workdir / "Kadlec_df.csv")
    cols_out = {'irr_tilt': 'sum', 'irr_eff': 'sum', 'irr_global_pv': 'sum', 'pv_energy_DC': 'sum'}
    fve_plots(workdir, df, units, "Kadlec_", cols_out)
    return cols_out, df, units

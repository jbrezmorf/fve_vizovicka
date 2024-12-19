"""
Test dependency of irradiance on the pannel orientation and tilt
"""
import pvlib
from pvlib import location
from pvlib import irradiance
from pvlib import tracking
from pvlib.iotools import read_tmy3
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import pathlib

# get full path to the data directory
DATA_DIR = pathlib.Path(pvlib.__file__).parent / 'data'

# get TMY3 dataset
tmy, metadata = read_tmy3(DATA_DIR / '723170TYA.CSV', coerce_year=1990,
                          map_variables=True)
# TMY3 datasets are right-labeled (AKA "end of interval") which means the last
# interval of Dec 31, 23:00 to Jan 1 00:00 is labeled Jan 1 00:00. When rolling
# up hourly irradiance to monthly insolation, a spurious January value is
# calculated from that last row, so we'll just go ahead and drop it here:
tmy = tmy.iloc[:-1, :]

# create location object to store lat, lon, timezone
location = location.Location.from_tmy(metadata)

# calculate the necessary variables to do transposition.  Note that solar
# position doesn't depend on array orientation, so we just calculate it once.
# Note also that TMY datasets are right-labeled hourly intervals, e.g. the
# 10AM to 11AM interval is labeled 11.  We should calculate solar position in
# the middle of the interval (10:30), so we subtract 30 minutes:
times = tmy.index - pd.Timedelta('30min')
solar_position = location.get_solarposition(times)
# but remember to shift the index back to line up with the TMY data:
solar_position.index += pd.Timedelta('30min')


# create a helper function to do the transposition for us
def calculate_poa(tmy, solar_position, surface_tilt, surface_azimuth):
    # Use the get_total_irradiance function to transpose the irradiance
    # components to POA irradiance
    poa = irradiance.get_total_irradiance(
        surface_tilt=surface_tilt,
        surface_azimuth=surface_azimuth,
        dni=tmy['dni'],
        ghi=tmy['ghi'],
        dhi=tmy['dhi'],
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth'],
        model='isotropic')
    return poa['poa_global']  # just return the total in-plane irradiance


# create a dataframe to keep track of our monthly insolations
df_monthly = pd.DataFrame()
cases = []
# fixed-tilt:
fig, axes = plt.subplots(2, 3, figsize=(20, 8))
for tilt,ax in zip(range(30, 90, 10), axes.flatten()):
    for az in range(110, 260, 20):
        print(f"Calculating for tilt={tilt}, azimuth={az}")
        # we will hardcode azimuth=180 (south) for all fixed-tilt cases
        poa_irradiance = calculate_poa(tmy, solar_position, tilt, az).copy()

        # TMYs are hourly, so we can just sum up irradiance [W/m^2] to get
        # insolation [Wh/m^2]:
        date = pd.Series(poa_irradiance.index)
        df = pd.DataFrame({
            'irrad': poa_irradiance,
            'hour': date.dt.hour.values,
            'month': date.dt.month.values,
            'azimuth': az,
        }, index=date)
        #df.set_index('hour', inplace=True)
        cases.append(df)
    df = pd.concat(cases)
    df = df[df['month'].isin({11, 12, 1})]
    df = df.groupby(['hour', 'azimuth']).mean().reset_index()
    #df.set_index('hour', inplace=True)

# calculate the percent difference from GHI
    #ghi_monthly = tmy['ghi'].resample('m').sum()
    #df_monthly = 100 * (df_monthly.divide(ghi_monthly, axis=0) - 1)
    # Plot using seaborn
    sns.lineplot(
        data=df,
        x='hour',
        y='irrad',
        hue='azimuth',
        palette='viridis',
        ax=ax
    )
    #df.plot(ax=ax)
    ax.set_xlabel('Hour')
    ax.set_ylabel(f'Plane irrad ({tilt})')
    ax.grid()
plt.tight_layout()
plt.show()
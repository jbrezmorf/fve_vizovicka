import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


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


def fve_plots(workdir, df, units, file_prefix, cols_out):
    df = df.reset_index()
    if isinstance(cols_out, list):
        cols_out = {c: 'sum' for c in cols_out}
    df['hour'] = df.date_time.dt.hour
    df['month'] = df.date_time.dt.month
    #df.set_index('date_time', inplace=True)
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
    year_df = pd.DataFrame([df[cols_full.keys()].sum(axis=0)], index=['Total'])
    print_df = pd.concat([print_df, year_df])
    print_df.to_csv(workdir / (file_prefix + "month_sums.csv"))
    df.set_index('date_time', inplace=True)
    plot_df = df[cols_full.keys()]
    return plot_df

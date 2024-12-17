import numpy as np
import attr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@attr.s(auto_attribs=True)
class SensitivityResults:
    S_x: float  # Sensitivity index for x
    S_y: float  # Sensitivity index for y
    S_z: float  # Residual variance (attributed to z)
    S_xy: float  # Interaction sensitivity (x and y combined)
    correlation_xy: float  # Correlation between x and y


def compute_sensitivities(df: pd.DataFrame, columns: list = None, alpha: float = 1.0,
                          plot: bool = False) -> SensitivityResults:
    """
    Compute sensitivity indices for x, y, z (residual), and xy interaction, and correlation between x and y.

    Parameters:
    - df: pd.DataFrame, input data with at least three columns representing f, x, and y.
    - columns: list, optional, names of the columns to use (f, x, y). Defaults to the first three columns.
    - alpha: float, regularization strength for Ridge regression. Default is 1.0.
    - plot: bool, optional, whether to plot the original data and fitted function. Default is False.

    Returns:
    - SensitivityResults: An attrs class with sensitivity indices and correlation.
    """
    # Extract relevant columns
    if columns is None:
        columns = list(df.columns[:3])
    if len(columns) != 3:
        raise ValueError(
            "The DataFrame must have exactly three columns for f, x, and y, or specify exactly three columns in the 'columns' parameter.")

    f, x, y = [df[col].values for col in columns[:3]]

    # Ensure inputs are numpy arrays
    x, y, f = map(np.asarray, (x, y, f))

    # 1. Fit a regression model with regularization
    model = Ridge(alpha=alpha)
    X = np.column_stack((x, y))
    model.fit(X, f)
    f_pred = model.predict(X)

    # 2. Compute residuals
    residuals = f - f_pred

    # 3. Compute variances
    total_variance = np.var(f)
    explained_variance = np.var(f_pred)
    residual_variance = np.var(residuals)

    # 4. Compute sensitivity indices
    # Marginal effect of x
    S_x = np.var(model.predict(np.column_stack((x, np.mean(y) * np.ones_like(y))))) / total_variance

    # Marginal effect of y
    S_y = np.var(model.predict(np.column_stack((np.mean(x) * np.ones_like(x), y)))) / total_variance

    # Joint effect of x and y (interaction)
    S_xy = explained_variance / total_variance - S_x - S_y

    # Residual (unexplained variance)
    S_z = residual_variance / total_variance

    # 5. Correlation between x and y
    correlation_xy = np.corrcoef(x, y)[0, 1]

    # Ensure indices add up to ~1 due to numerical precision
    assert np.isclose(S_x + S_y + S_xy + S_z, 1.0, atol=1e-2), "Sensitivity indices do not sum to 1"

    # Plot if requested
    if plot:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of the original data colored by residuals
        scatter = ax.scatter(x, y, f, c=residuals, cmap='coolwarm', label='Original Data')
        plt.colorbar(scatter, ax=ax, label='Residuals')

        # Surface plot of the fitted function
        x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 50), np.linspace(y.min(), y.max(), 50))
        z_grid = model.predict(np.column_stack((x_grid.ravel(), y_grid.ravel()))).reshape(x_grid.shape)
        ax.plot_surface(x_grid, y_grid, z_grid, alpha=0.5, cmap='viridis', label='Fitted Surface')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('F')
        ax.set_title('3D Plot of Original Data and Fitted Function')
        plt.show()

    return SensitivityResults(S_x=S_x, S_y=S_y, S_z=S_z, S_xy=S_xy, correlation_xy=correlation_xy)


# Example usage
if __name__ == "__main__":
    # Generate example data
    n_samples = 1000
    data = {
        "f": 3 * np.random.rand(n_samples) ** 2 + 2 * np.random.rand(n_samples) + np.random.normal(0, 0.1, n_samples),
        "x": np.random.rand(n_samples),
        "y": np.random.rand(n_samples)
    }
    df = pd.DataFrame(data)

    # Compute sensitivities
    results = compute_sensitivities(df, plot=True)

    # Print results
    print(f"S_x: {results.S_x:.3f}")
    print(f"S_y: {results.S_y:.3f}")
    print(f"S_z: {results.S_z:.3f}")
    print(f"S_xy: {results.S_xy:.3f}")
    print(f"Correlation (x, y): {results.correlation_xy:.3f}")

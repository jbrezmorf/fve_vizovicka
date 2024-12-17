import attrs
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csc_matrix


@attrs.define
class OperationCfg:
    year_consumption: float     # [kWh] total year energy
    heat_consumption: float     # [kWh] energy for heating by one year
    TV_consumption: float
    heat_temp_out:  float       # [deg C] outer temperature that could be covered by passive heating
    heat_temp_in: float         # [deg C] target indoor temperature
    peak_power: float           # Installed power in kW peak
    bat_min_charge: float       # minimum charge percentage (0 - 1)
    bat_max_charge: float       # maximum charge percentage (0 - 1)
    bat_cap: float              # setting unrealistic high capacity, turn effectively off the day sim model
    conv_eff: float             # conversion efficiency
    distr_sell_fee: float       # distribution fee per kWh
    distr_buy_fee: float        # distribution fee per kWh
    distr_per_day: float        # distribution fee per day
    eur_to_czk: float           # exchange rate
    sell_limit: float           # maximum amount of kWh that can be sold in a single hour
    #buy_limit: float        # maximum amount of kWh that can be bought in a single hour
    heating_mean_period: int  # Window length parameter for heating requirement (e.g., 6 hours means 7-hour window)


def model_operation(df, operation_cfg):
    arr = lambda df, n: np.array(df[n]).reshape(-1, 24)
    dc = arr(df, 'pv_energy_DC')
    consumption = arr(df, 'consumption')
    price = arr(df, 'price')
    consumption[np.isnan(consumption)] = 0.0

    bat_cap = operation_cfg.bat_cap
    conv_eff = operation_cfg.conv_eff

    def charge(bat_last, surplus):
        if surplus > 0:
            charge = min(surplus, operation_cfg.bat_max_charge * bat_cap - bat_last)
            loss = max(0.0, charge * (1 - conv_eff))
            return bat_last + charge * conv_eff, surplus - charge, charge, loss
        else:
            charge = max(surplus, bat_cap * operation_cfg.bat_min_charge - bat_last)
            loss = max(0.0, -charge * (1 - conv_eff) / conv_eff)
            return bat_last + charge / conv_eff, surplus - charge, charge, loss

    # Start year with half charged battery
    bat_last = bat_cap / 2.0
    bat_energy = np.zeros_like(dc)
    sell = np.zeros_like(dc)
    over_production = np.zeros_like(dc)
    conv_loss = np.zeros_like(dc)

    for i in range(len(dc)):
        ac = dc[i] * conv_eff
        conv_loss[i,:] = (dc[i] - ac) / 24
        total_production = np.sum(ac)
        total_consumption = np.sum(consumption[i])
        if total_production < total_consumption:
            lack  = total_consumption - total_production
            sell[i, np.argmin(price[i])] = -lack
        else:
            # simple_loss = sell * (sell > 0) * (price[h] - distr_sell) + sell * (sell < 0) * (price + distr_buy)
            # balance = ac - sell - consumption
            # under constrains:
            # sell < sell_limit
            #


            #
            surplus = total_production - total_consumption
            max_sell = ac * price[i]
            i_sorted = np.argsort(max_sell)[::-1]
            ac_cum = np.cumsum(ac[i_sorted])
            k_index = np.searchsorted(ac_cum, surplus)
            if k_index >= len(i_sorted):
                print("Can not find sell tipping point.", ac_cum, surplus)  # Should be in bounds since surplus is smaller than ac_cum[-1]
            sell_row = np.zeros_like(ac)
            sell_row[i_sorted[:k_index]] = ac[i_sorted[:k_index]]
            sell_row[i_sorted[k_index]] = ac_cum[k_index] - surplus
            sell[i, :] = np.minimum(sell_row, operation_cfg.sell_limit)
            #over_sell_row = np.sell_row - sell[i, :]

        for j in range(24):
            surplus = ac[j] - consumption[i, j] - sell[i, j]
            bat_last, surplus, bat_charge, loss = charge(bat_last, surplus)
            conv_loss[i, j] += loss
            sell_actual = ac[j] - consumption[i, j] - surplus
            if surplus > 0.0:   # over production
                sell[i, j] = min(operation_cfg.sell_limit, sell_actual)
                over_production[i, j] = sell_actual - sell[i, j]
            elif surplus < 0.0: # need to buy; buy limit (that must model adapted consumtion + overdischarging)
                sell[i, j] = ac[j] - consumption[i, j] - surplus

            bat_energy[i, j] = bat_last

    distr_per_day = operation_cfg.distr_per_day
    eur_to_czk = operation_cfg.eur_to_czk

    sell = sell.flatten()
    price = price.flatten() * eur_to_czk / 1000
    price_eff = price.copy()
    price_eff[sell > 0] = price[sell > 0] - operation_cfg.distr_sell_fee
    price_eff[sell <= 0] = price[sell <= 0] + operation_cfg.distr_buy_fee
    revenue = sell * price - distr_per_day / 24

    df_operation = pd.DataFrame({
        'sell': sell,
        'revenue': revenue,
        'bat_energy': bat_energy.flatten(),
        'over_production':over_production.flatten()
    }, index=df.index)

    print("Peak AC: ", np.max(ac))
    print("\n======================")
    print("+DC:             ", np.sum(dc))
    print("-sell:           ", np.sum(sell)*(sell>0))
    print("+buy:            ", np.sum(sell)*(sell<0))
    print("-concumption    :", np.sum(consumption))
    print("-over_production:", np.sum(over_production))
    print("-conv_loss:      ", np.sum(conv_loss))
    print("\n======================")
    print("balance:         ", np.sum(dc) - np.sum(sell) - np.sum(consumption) - np.sum(over_production) - np.sum(conv_loss))
    res_df = pd.concat([df, df_operation], axis=1)
    return res_df







@attrs.define
class CoordMat:
    n_cols: int
    data: list = attrs.field(factory=list)
    rows: list = attrs.field(factory=list)
    cols: list = attrs.field(factory=list)

    def append(self, i, j, val):
        self.data.append(val)
        self.rows.append(i)
        self.cols.append(j)

    def extend(self, i_list, j_list, val_list):
        inputs = [np.atleast_1d(l) for l in [i_list, j_list, val_list]]
        sizes = [len(l) for l in inputs]
        n = max(sizes)
        #assert n > 1, f"At least one list must be non-empty: {sizes}"
        i_list, j_list, val_list = [np.full(n, l[0]) if len(l) == 1 else l for l in inputs]
        self.rows.extend(i_list)
        self.cols.extend(j_list)
        self.data.extend(val_list)

    def get_csc(self, shape):
        return csc_matrix((self.data, (self.rows, self.cols)), shape=shape)

@attrs.define
class BlockMat:
    buy: CoordMat
    sell: CoordMat
    heat: CoordMat
    bat_E: CoordMat
    over_heat: CoordMat

    @property
    def _blocks(self):
        return [self.buy, self.sell, self.heat, self.bat_E, self.over_heat]

    def get_csc(self):
        # Step 1: Compute start indices for each block
        col_starts = np.cumsum([0] + [block.n_cols for block in self._blocks[:-1]])

        # Step 2: Get maximum row index across all blocks
        max_row = max((max(block.rows) if block.rows else 0) for block in self._blocks)
        n_rows = max_row + 1

        # Step 3: Recompute column indices with block offsets
        data, rows, cols = [], [], []
        for block, start in zip(self._blocks, col_starts):
            data.extend(block.data)
            rows.extend(block.rows)
            cols.extend([start + col for col in block.cols])

        # Step 4: Form the composed CSC matrix
        n_cols = sum(block.n_cols for block in self._blocks)
        return csc_matrix((data, (rows, cols)), shape=(n_rows, n_cols))

    def get_full(self):
        # Step 1: Compute start indices for each block
        col_starts = np.cumsum([0] + [block.n_cols for block in self._blocks[:-1]])

        # Step 2: Get maximum row index across all blocks
        max_row = max((max(block.rows) if block.rows else 0) for block in self._blocks)
        n_rows = max_row + 1

        mat = np.zeros((n_rows, sum(block.n_cols for block in self._blocks)))
        for block, start in zip(self._blocks, col_starts):
            cols = [start + col for col in block.cols]
            for i,j,v in zip(block.rows, cols, block.data):
                mat[i,j] = v
        return mat

    def copy(self):
        blk_copy = [CoordMat(blk.n_cols) for blk in self._blocks]
        return BlockMat(*blk_copy)





def period_operation(df, operation_cfg, start_i, end_i, E_init, price_buy, price_sell):
    """
    Solve the linear programming problem for a single period.

    Parameters:
        df (pd.DataFrame): Input data with columns ['pv_energy_DC', 'consumption', 'price_buy', 'price_sell'].
        operation_cfg (OperationCfg): Configuration for the problem, including bounds and parameters.

    Returns:
        scipy.optimize.OptimizeResult: Optimization result from linprog.
    """
    print(f"LP solve ({start_i}: {end_i})")
    df = df.iloc[start_i:end_i]
    price_buy = price_buy[start_i:end_i]
    price_sell = price_sell[start_i:end_i]
    n = len(df)
    m = operation_cfg.heating_mean_period
    assert m <= n, f"Heating window length {m} must be less than the period length {n}"
    w = n - m + 1

    b_min = operation_cfg.bat_min_charge * operation_cfg.bat_cap
    b_max = operation_cfg.bat_max_charge * operation_cfg.bat_cap

    # Extract parameters
    net_production = (df['pv_energy_DC'].values * operation_cfg.conv_eff - df['other_consumption'].values)

    # Initialize block matrices
    block_mat = BlockMat(
        buy=CoordMat(n_cols=n),
        sell=CoordMat(n_cols=n),
        heat=CoordMat(n_cols=n),
        bat_E=CoordMat(n_cols=n),
        over_heat=CoordMat(n_cols=w)
    )

    # Objective vector, minimization loss: - revenue = - (sell * price_sel  - buy * price_buy)
    assert np.all(price_buy >= price_sell), "Assert buy_price > sell_price"
    c_mat = block_mat.copy()
    c_mat.buy.extend(0, list(range(n)), price_buy)
    c_mat.sell.extend(0, list(range(n)), -price_sell)
    c_mat.over_heat.extend(0, list(range(w)), 0.01)     # panalize overheating a bit
    c = c_mat.get_full().flatten()


    # Constraints
    b_eq = []

    # Battery evolution constraints
    for i in range(n):
        row_idx = i
        if i > 0:
            block_mat.bat_E.append(row_idx, i - 1, -1.0)  # E[i-1]
        block_mat.bat_E.append(row_idx, i, 1.0)  # E[i]
        block_mat.buy.append(row_idx, i, -1.0)  # -B[i]
        block_mat.sell.append(row_idx, i, 1.0)  # S[i]
        block_mat.heat.append(row_idx, i, 1.0)  # -H[i]
        b_eq.append(net_production[i])  # Include net_production[i] in the RHS, adjusted for E_init

    # Adjust for initial battery state
    b_eq[0] +=E_init

    # Heating window constraints
    for i in range(w):
        row_idx = n + i
        cols = np.arange(i, i + m, dtype=int)
        block_mat.heat.extend(row_idx, cols, 1.0)  # H[j]
        block_mat.over_heat.append(row_idx, i, -1.0)  # -O[i]
        h_mean = np.sum(df['heat_consumption'].iloc[i: i + m].values)
        b_eq.append(h_mean)  # Heating requirement

    # Convert constraints to sparse matrices
    A_eq = block_mat.get_csc()
    b_eq = np.array(b_eq)

    # Bounds for variables
    bounds_mat = block_mat.copy()
    bounds_mat.buy.extend(1, np.arange(n, dtype=int), 25)
    bounds_mat.sell.extend(1, np.arange(n, dtype=int), operation_cfg.sell_limit)
    bounds_mat.heat.extend(1, np.arange(n, dtype=int), np.nan)
    bounds_mat.bat_E.extend(0, np.arange(n, dtype=int), b_min)
    bounds_mat.bat_E.extend(1, np.arange(n, dtype=int), b_max)
    bounds_mat.over_heat.extend(1, np.arange(w, dtype=int), np.nan)
    full_bounds_mat  = bounds_mat.get_full()

    # Convert to bounds parameter for linprog, replacing np.nan with None
    bounds = [
        (lb if not np.isnan(lb) else None, ub if not np.isnan(ub) else None)
        for lb, ub in zip(full_bounds_mat[0], full_bounds_mat[1])
    ]

    # Solve the LP
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    # Split the result into components
    if res.success:
        x = res.x
        result_df = pd.DataFrame({
            'buy': x[:n],
            'sell': x[n:2*n],
            'heat': x[2*n:3*n],
            'bat_E': x[3*n:4*n],

        })
        overheat = x[4*n:]
    else:
        raise ValueError("Optimization failed: " + res.message)
    return result_df, overheat


def generate_subproblems(T, sim_period, overlap):
    """
    Generate a list of subproblem indices based on total time T, simulation period, and overlap.

    Parameters:
        T (int): Total time in hours.
        sim_period (int): Length of each subproblem in hours.
        overlap (int): Overlap between subsequent subproblems in hours.

    Returns:
        list: List of (start, end) indices for subproblems.
    """
    subproblems = []
    start = 0
    while start < T:
        end = min(start + sim_period, T)
        subproblems.append((start, end))
        if end == T:
            break
        start = end - overlap
    return subproblems



def model_operation_lp(df, operation_cfg, sim_period=2*24, overlap=24, E_init=0.5):
    """
    Simulate operation of the energy system by dividing the simulation into overlapping LP subproblems.

    Parameters:
        df (pd.DataFrame): Contains columns ['pv_energy_DC', 'consumption', 'price', 'heating'].
        operation_cfg (OperationCfg): Configuration parameters for the operation.
        sim_period (int): Length of each subproblem in hours.
        overlap (int): Overlap between subsequent subproblems in hours.
        E_init (float): Initial battery charge (in kWh).

    Returns:
        list: Results of solved LP subproblems.
    """

    T = len(df)
    eur_to_czk = operation_cfg.eur_to_czk
    price_buy = df['price'].values * eur_to_czk / 1000 + operation_cfg.distr_buy_fee
    price_sell = df['price'].values * eur_to_czk / 1000 - operation_cfg.distr_sell_fee

    subproblems = generate_subproblems(T, sim_period, overlap)

    results = []

    E_last = E_init
    for (start_i, end_i) in subproblems:
        df_period, overheat = period_operation(df, operation_cfg, start_i, end_i, E_last, price_buy, price_sell)
        print(overheat)
        results.append(df_period.iloc[:-overlap, :])
        E_last = df_period['bat_E'].iloc[-1]
    results.append(df_period.iloc[-overlap:, :])
    res_df = pd.concat(results, axis=0)
    assert len(res_df) == len(df), f"Result length {len(res_df)} must match input length {len(df)}"
    res_df.index = df.index
    sell = res_df['sell'].values
    buy =  res_df['buy'].values
    consumption = res_df['heat'].values + df['other_consumption'].values
    overheat = res_df['heat'].values - df['heat_consumption'].values

    dc = df['pv_energy_DC'].values
    ac = dc * operation_cfg.conv_eff
    conv_loss = dc - ac
    revenue = sell * price_sell - buy * price_buy
    print("Peak AC: ", np.max(ac))
    print("\n======================")
    print("+DC:             ", np.sum(dc))
    print("-sell:           ", np.sum(sell))
    print("+buy:            ", np.sum(buy))
    print("-concumption    :", np.sum(consumption))
    print("-over_heat:", np.sum(overheat))
    print("-conv_loss:      ", np.sum(conv_loss))
    print("\n======================")
    print("balance:         ", np.sum(dc) +np.sum(buy) - np.sum(sell) - np.sum(consumption) - np.sum(conv_loss))
    print("revenue:         ", np.sum(revenue))
    df_operation = pd.DataFrame({
        'sell': sell - buy,
        'buy': buy,
        'revenue': revenue,
        'bat_energy': res_df['bat_E'].values,
        'over_heat': overheat
    }, index=df.index)
    result_df = pd.concat([df, df_operation], axis=1)
    return result_df

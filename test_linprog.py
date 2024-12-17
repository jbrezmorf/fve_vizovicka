import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

# Define the cost function coefficients
c = np.array([1, 1])  # Minimize x0 + x1

# Define the inequality constraints in triplet form
# A_ub * x <= b_ub

# We have:
# -x0 - 2x1 <= -6
# -3x0 - x1  <= -8

# Rows and columns for A_ub in COO format:
# Let's say we have 2 constraints and 2 variables:
# Constraint 1: row = 0, columns for x0 and x1 = 0,1
# Constraint 2: row = 1, columns for x0 and x1 = 0,1

row_indices = np.array([0, 0, 1, 1])
col_indices = np.array([0, 1, 0, 1])
values = np.array([-1, -2, -3, -1])

A_ub_sparse = coo_matrix((values, (row_indices, col_indices)), shape=(2, 2))
b_ub = np.array([-6, -8])

# We have no equality constraints in this example
A_eq = None
b_eq = None

# Solve the linear program using one of the HiGHS methods to ensure sparse support.
res = linprog(c, A_ub=A_ub_sparse, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')

print("Optimal solution status:", res.message)
print("Optimal solution:", res.x)
print("Optimal objective value:", res.fun)

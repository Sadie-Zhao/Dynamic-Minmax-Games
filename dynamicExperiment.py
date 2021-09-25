import dynamicMinmax as dm
import dynamicLibrary as lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date


num_iterations = 20000
num_buyers =  5
num_goods = 8
learning_rate =  5
prices_0  = np.random.rand(num_goods) * 10 + 5

prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist, cumulative_loss_hist = dm.dynamic_GDA_with_oracle(200, num_goods, num_buyers,
"linear", prices_0, 
10, 5,
10, 100,
10, 1,
learning_rate)

print(cumulative_loss_hist)

# print(lib.get_p_cumulative_regret(num_buyers, num_goods, demands_hist, supplies_hist, budgets_hist, valuations_hist, cumulative_loss_hist, lib.get_linear_obj))
plt.plot(cumulative_loss_hist)
plt.show()
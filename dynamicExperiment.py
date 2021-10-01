import dynamicMinmax as dm
import dynamicLibrary as lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date


num_iterations = 500
num_experiment = 10
num_buyers =  5
num_goods = 8
learning_rate_linear =  ((3,3), (1000**(-1/2),1000**(-1/2)))
learning_rate_cd =  ((5, 0.01), (500**(-1/2), 500**(-1/2)))
learning_rate_leontief =  ((5,0.001), (500**(-1/2),500**(-1/2)))
prices_0  = np.random.rand(num_goods) * 10 + 5

value_avg_hist_all = []
for iter in range(num_experiment):
    prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist = dm.dynamic_GDA_with_oracle(num_iterations, num_goods, num_buyers,
    "leontief", prices_0, 
    10, 5,
    10, 10,
    10, 5,  #if linear: 10,100; otherwise: 10,5
    learning_rate_leontief[0])

    value_avg_hist = lib.get_average_value_static(lib.get_leontief_value, prices_hist, demands_hist, budgets_hist[0], valuations_hist[0])
    value_avg_hist_all.append(value_avg_hist)

# print(lib.get_p_cumulative_regret(num_buyers, num_goods, demands_hist, supplies_hist, budgets_hist, valuations_hist, cumulative_loss_hist, lib.get_linear_obj))

value_avg_hist_final = np.mean(value_avg_hist_all, axis = 0)
print(value_avg_hist_final)
plt.plot(value_avg_hist_final)
plt.show()
import numpy as np
from numpy.core.fromnumeric import size
import dynamicLibrary as lib
import fisherMinmax as fm
import matplotlib.pyplot as plt


num_buyers =  5
num_goods = 8
learning_rate_linear =  ((3,1), (1000**(-1/2),1000**(-1/2)))
learning_rate_cd =  ((3,1), (500**(-1/2), 500**(-1/2)))
learning_rate_leontief =  ((5,0.001), (500**(-1/2),500**(-1/2)))
prices_0  = np.random.rand(num_goods)*10 + 5

valuations = np.random.rand(num_buyers, num_goods)*10 + 5
budgets = np.random.rand(num_buyers)*10 + 10


# print(valuations)
# print()
# print(budgets)
# print()
# print(demands)
# print()

prices_hist_gda_cd_all_low = []
demands_hist_gda_cd_all_low = []
obj_hist_gda_cd_all_low = []
num_iters_cd = 500
valuations_cd = (valuations.T/ np.sum(valuations, axis = 1)).T
value_avg_hist_all = []

for num in range(5):

    demands_gda, prices_gda, demands_hist_gda, prices_hist_gda = fm.gda_cd(valuations_cd, budgets, prices_0, learning_rate_cd[0], num_iters_cd)

    prices_hist_gda_cd_all_low.append(prices_gda)
    demands_hist_gda_cd_all_low.append(demands_gda)
    objective_values = []

    # print(prices_hist_gda)
    # print(demands_hist_gda)
    value_avg_hist = lib.get_average_value_static(lib.get_cd_value, prices_hist_gda, demands_hist_gda, budgets, valuations)
    value_avg_hist_all.append(value_avg_hist)

value_avg_hist_final = np.mean(value_avg_hist_all, axis = 0)
plt.plot(value_avg_hist_final)
plt.show()
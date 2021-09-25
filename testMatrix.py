import numpy as np
from numpy.core.fromnumeric import size
import dynamicLibrary as lib


num_buyers = 3 #n
num_goods = 4 #m

valuations = np.array([i+1 for i in range(12)], dtype = np.float64).reshape(3,4)
budgets = np.array([9, 10, 11], dtype = np.float64)
demands = np.ones(12, dtype = np.float64).reshape(3,4)
old_prices = np.random.rand(4)*10 

# print(valuations)
# print(budgets)
# print(demands)

util_func = lib.get_linear_utility
util_gradient_func = lib.get_linear_util_gradient
learning_rate = 0.02


# constants_list = []
# for v_i, b_i, x_i in zip(valuations, budgets, demands):
#     c_i = b_i / util_func(x_i, v_i)
#     constants_list.append(c_i)
# constants = (np.array(constants_list) * learning_rate).reshape(num_buyers, 1)
# demands += (constants * util_gradient_func(demands, valuations)) - old_prices


# print(constants.T)
# print(constants * util_gradient_func(demands, valuations))
# print(old_prices)
# print(demands)

print(lib.get_leontief_util_gradient(demands, valuations))

import dynamicMinmax as dm
import dynamicLibrary as lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date
import fisherNestedMinmax as nminmax


num_iterations = 1000
num_experiment = 3
num_buyers =  5
num_goods = 8 
learning_rate_linear =  ((5,0.01), (1000**(-1/2),1000**(-1/2))) #0.001,0.0001 for gda #5,0.01 for LGDA!
learning_rate_cd =  ((5, 0.01), (500**(-1/2), 500**(-1/2)))  #0.0001,0.001 for gda    #5,0.01 for LGDA!
learning_rate_leontief =  ((5, 0.01), (500**(-1/2),500**(-1/2))) #0.0001,0.01 for gda #5,0.001 for LGDA
prices_0  = np.random.rand(num_goods) * 5 + 50

valuations = np.random.rand(num_buyers, num_goods)*10 + 5
budgets = np.random.rand(num_buyers)*10 + 100
supplies = np.random.rand(num_goods)*10 + 5

value_avg_hist = []
value_avg_hist_all = []
value_avg_final = []

value_hist = []
obj_hist = []

prices_dist_hist = []
demands_dist_hist = []
p_plus_x_dist_hist = []
obj_dist_hist = []
value_dist_hist = []
util_type = "leontief"

for iter in range(num_experiment):
    # prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist = dm.dynamic_LGDA(num_iterations, num_goods, num_buyers,
    # util_type, prices_0, 
    # 10, 5,
    # 10, 10,
    # 10, 100,  #if linear: 10,100; otherwise: 10,5
    # learning_rate_leontief[0],
    # decay_outer= True, decay_inner=True)

    # prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist = dm.dynamic_GDA_with_lagrangian_oracle(num_iterations, num_goods, num_buyers,
    # "linear", prices_0, 
    # 10, 5,
    # 10, 10,
    # 10, 100, 
    # learning_rate_linear[0],
    # decay_outer= True, decay_inner=True)

    # prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist = dm.dynamic_LGDA(valuations, budgets, supplies, num_iterations, num_goods, num_buyers,
    # "cd", prices_0, 
    # 10, 5,
    # 10, 10,
    # 10, 5, 
    # learning_rate_cd[0],
    # decay_outer= True, decay_inner=True)

    # prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist = dm.dynamic_LGDA(valuations, budgets, supplies, num_iterations, num_goods, num_buyers,
    # "linear", prices_0, 
    # 10, 5,
    # 10, 10,
    # 10, 5, 
    # learning_rate_linear[0],
    # decay_outer=False, decary_inner=True)

    prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist = dm.dynamic_max_oracle_GD(num_iterations, num_goods, num_buyers,
    util_type, prices_0, 
    10,10,
    10, 5,
    10, 5,  #10,5 for leontief
    learning_rate = 1.5)

    ######################## averaged values #####################################
    # value_avg_hist = lib.get_average_value_static(lib.get_cd_value, prices_hist, demands_hist, budgets, valuations)
    # value_avg_hist_all.append(value_avg_hist)
# value_avg_hist_final = np.mean(value_avg_hist_all, axis = 0)
# print(value_avg_hist_final)
# plt.plot(value_avg_hist_final)
# plt.show()


    ######################## Normal values ###################################
#     value_values = []
#     for x, p in zip(demands_hist, prices_hist):
#         value = lib.get_cd_value(p, x, budgets, valuations)
#         value_values.append(value)
#     value_hist.append(value_values)
# value_hist_final = np.mean(value_hist, axis = 0)
# print(value_hist_final)
# plt.plot(value_hist_final)
# plt.show()

# obj_hist_final = np.mean(obj_hist, axis = 0)
# print(obj_hist_final)
# plt.plot(obj_hist_final)
# plt.show()



    ######################## plot distances ################################
    prices_dists = []
    demands_dists = []
    p_plus_x_dists = []
    obj_dists = []
    value_dists = []
    for prices, demands, valuations, budgets, supplies in zip(prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist):
        p_plus_x_dist, obj_dist, value_dist = lib.get_dist_to_equilibrium(util_type, prices, demands, valuations, budgets, supplies)
        # prices_dists.append(prices_dist)
        # demands_dists.append(demands_dist)
        p_plus_x_dists.append(p_plus_x_dist)
        obj_dists.append(obj_dist)
        value_dists.append(value_dist)
    # prices_dist_hist.append(prices_dists)
    # demands_dist_hist.append(demands_dists)
    p_plus_x_dist_hist.append(p_plus_x_dists)
    obj_dist_hist.append(obj_dists)
    value_dist_hist.append(value_dists)
prices_dist_final = np.mean(prices_dist_hist, axis=0)
# plt.plot(prices_dist_final)
# plt.xlabel('iterations') 
# plt.ylabel('distance to equilibrium')
# plt.title('distance to equilibrium in terms of prices ({})'.format(util_type))
# plt.show()

# demands_dist_final = np.mean(demands_dist_hist, axis=0)
# plt.plot(demands_dist_final)
# plt.xlabel('iterations') 
# plt.ylabel('distance to equilibrium')
# plt.title('distance to equilibrium in terms of demands ({})'.format(util_type))
# plt.show()

p_plus_x_dist_final = np.mean(p_plus_x_dist_hist, axis=0)
plt.plot(p_plus_x_dist_final)
plt.xlabel('iterations') 
plt.ylabel('distance to equilibrium')
plt.title('distance to equilibrium in terms of prices + demands ({})'.format(util_type))
plt.show()

obj_dist_final = np.mean(obj_dist_hist, axis=0)
plt.plot(obj_dist_final)
plt.xlabel('iterations') 
plt.ylabel('distance to equilibrium')
plt.title('distance to equilibrium in terms of obj ({})'.format(util_type))
plt.show()

value_dist_final = np.mean(value_dist_hist, axis=0)
plt.plot(value_dist_final)
plt.xlabel('iterations') 
plt.ylabel('distance to equilibrium')
plt.title('distance to equilibrium in terms of value ({})'.format(util_type))
plt.show()

print("pplusx:", p_plus_x_dist_final)

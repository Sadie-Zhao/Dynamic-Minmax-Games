import numpy as np
import dynamicLibrary as lib
import consumerUtility as cu
np.set_printoptions(precision=3)


############# Projection onto affine positive half-space, i.e., budget set ###############
def project_to_bugdet_set(X, p, b):
    X_prec = X
    while (True): 
        X -= ((X @ p - b).clip(min= 0)/(np.linalg.norm(p)**2).clip(min= 0.01) * np.tile(p, reps = (b.shape[0], 1)).T).T
        X = X.clip(min = 0)
        if(np.linalg.norm(X - X_prec) <= np.sum(X_prec)*0.05):
            break
        # print(f"Current iterate {X}\nPrevious Iterate {X_prec}")
        X_prec = X
    return X





def dynamic_max_oracle_GD(num_iters, num_goods, num_buyers, utility_type, prices_0,
valuations_range, valuations_addition,
budgets_range, budgets_addition,
supplies_range, supplies_addition,
learning_rate, decay = True):
    # # Set up functions to be used
    if utility_type == "linear":
        marshallian_demand_func = cu.get_linear_marshallian_demand
    elif utility_type == "leontief":
        marshallian_demand_func = cu.get_leontief_marshallian_demand
    elif utility_type == "cd":
        marshallian_demand_func = cu.get_CD_marshallian_demand


    # # Set up lists for data recording
    prices_hist = []
    demands_hist = []
    valuations_hist = []
    budgets_hist = []
    supplies_hist = []

    prices = np.copy(prices_0)
    prices_hist = []
    prices_hist.append(prices)
    demands_hist = []

    ################### Start the iterations ####################
    for iter in range(1, num_iters):
        if (not iter % 200):
            print(f" ----- Iteration {iter}/{num_iters} ----- ")


        ################# Get the problem f_t ################
        # valuations
        valuations = lib.get_valuations(num_buyers, num_goods, valuations_range, valuations_addition)
        # valuations = lib.get_valuations(num_buyers, num_goods, valuations_range - (iter-1)/num_iters*valuations_range, valuations_addition+(iter-1)/num_iters*valuations_range/2)
        if utility_type == "cd": # Normalize valuations for Cobb-Douglas
            valuations = (valuations.T/ np.sum(valuations, axis = 1)).T
        valuations_hist.append(valuations)
        # budgets
        budgets = lib.get_budgets(num_buyers, budgets_range, budgets_addition)
        # budgets = lib.get_budgets(num_buyers, budgets_range - (iter-1)/num_iters*budgets_range, budgets_addition+(iter-1)/num_iters*budgets_range/2)
        budgets_hist.append(budgets)
        # supplies
        supplies = lib.get_supplies(num_goods, supplies_range, supplies_addition)
        # supplies = lib.get_supplies(num_goods, supplies_range - (iter-1)/num_iters*supplies_range, supplies_addition+(iter-1)/num_iters*supplies_range/2)
        supplies_hist.append(supplies)
        
        if utility_type == "cd":
            demands = np.zeros(valuations.shape).clip(min = 0.001)  


        ################## Demands Step ####################
        demands = np.zeros(valuations.shape)
        for buyer in range(budgets.shape[0]):
            demands[buyer,:] = marshallian_demand_func(prices, budgets[buyer], valuations[buyer])
       
        demands_hist.append(demands)
        
        ################### Prices Step #########################
        demand = np.sum(demands, axis = 0)
        demand = demand.clip(min = 0.01)
        excess_demand = demand - supplies
        if (decay) :
            step_size = iter ** (-1/2) * excess_demand
            prices += step_size * (prices > 0)
        else:
            step_size = learning_rate * excess_demand
            prices += step_size * (prices > 0)

        prices = prices.clip(min = 0.01)
        prices_hist.append(prices)
        
    return prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist


def static_GDA_with_lagrangian_oracle(valuations, budgets, supplies, num_iters, num_goods, num_buyers, utility_type, prices_0,
valuations_range, valuations_addition,
budgets_range, budgets_addition,
supplies_range, supplies_addition,
learning_rate, decay_outer = False, decay_inner = False):

    # Set up functions to be used
    if utility_type == "linear":
        util_func = lib.get_linear_utility
        util_gradient_func = lib.get_linear_util_gradient
        obj_func = lib.get_linear_obj
    elif utility_type == "leontief":
        util_func = lib.get_leontief_utility
        util_gradient_func = lib.get_leontief_util_gradient
        obj_func = lib.get_leontief_obj
    elif utility_type == "cd":
        util_func = lib.get_cd_utility
        util_gradient_func = lib.get_cd_util_gradient
        obj_func = lib.get_cd_obj


    # Set up lists for data recording
    prices_hist = []
    demands_hist = []
    valuations_hist = []
    budgets_hist = []
    supplies_hist = []
    prices = np.zeros(num_goods)
    demands = np.zeros((num_buyers, num_goods))

    
    ################# Get the problem f_t ################
    # valuations = lib.get_valuations(num_buyers, num_goods, valuations_range, valuations_addition)
    # Normalize valuations for Cobb-Douglas
    if utility_type == "cd":
        valuations = (valuations.T/ np.sum(valuations, axis = 1)).T
    valuations_hist.append(valuations)
    # budgets = lib.get_budgets(num_buyers, budgets_range, budgets_addition)
    budgets_hist.append(budgets)
    # supplies = lib.get_supplies(num_goods, supplies_range, supplies_addition)
    supplies_hist.append(supplies)
    
    if utility_type == "cd":
        demands = np.zeros(valuations.shape).clip(min = 0.001)  


    ################### Start the iterations ####################
    for iter in range(1, num_iters):
        if (not iter % 50):
            print(f" ----- Iteration {iter}/{num_iters} ----- ")


        ##################### Price Step #####################
        demand_row_sum = np.sum(demands, axis = 0)
        excess_demands = demand_row_sum - supplies

        if iter == 1 :
            prices = np.copy(prices_0)
            old_prices = prices
        else:
            step_size = learning_rate[0] * excess_demands
            prices += step_size * (prices > 0)
            old_prices = prices_hist[-1]

        prices = np.clip(prices, a_min=0.001, a_max = None) # Make sure the price is positive
        prices_hist.append(prices)


        ##################### Demand Step ####################

        # Gradient Step
        constants_list = []
        for v_i, b_i, x_i in zip(valuations, budgets, demands):
            c_i = b_i / max(util_func(x_i, v_i), 0.001)
            constants_list.append(c_i)
        constants = (np.array(constants_list) * learning_rate[1]).reshape(num_buyers, 1)
        demands += constants * util_gradient_func(demands, valuations)

        # Projection Step
        demands = project_to_bugdet_set(demands, old_prices, budgets) #is here p^(t-1) or p^(t)
        demands_hist.append(demands)
    
    return prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist

def dynamic_GDA_with_lagrangian_oracle(num_iters, num_goods, num_buyers, utility_type, prices_0,
valuations_range, valuations_addition,
budgets_range, budgets_addition,
supplies_range, supplies_addition,
learning_rate, decay_outer = False, decay_inner = False):

    # Set up functions to be used
    if utility_type == "linear":
        util_func = lib.get_linear_utility
        util_gradient_func = lib.get_linear_util_gradient
        obj_func = lib.get_linear_obj
    elif utility_type == "leontief":
        util_func = lib.get_leontief_utility
        util_gradient_func = lib.get_leontief_util_gradient
        obj_func = lib.get_leontief_obj
    elif utility_type == "cd":
        util_func = lib.get_cd_utility
        util_gradient_func = lib.get_cd_util_gradient
        obj_func = lib.get_cd_obj


    # Set up lists for data recording
    prices_hist = []
    demands_hist = []
    valuations_hist = []
    budgets_hist = []
    supplies_hist = []
    prices = np.zeros(num_goods)
    demands = np.zeros((num_buyers, num_goods))



    ################### Start the iterations ####################
    for iter in range(1, num_iters):
        if (not iter % 50):
            print(f" ----- Iteration {iter}/{num_iters} ----- ")


        ################# Get the problem f_t ################
        valuations = lib.get_valuations(num_buyers, num_goods, valuations_range, valuations_addition)
        # Normalize valuations for Cobb-Douglas
        if utility_type == "cd":
            valuations = (valuations.T/ np.sum(valuations, axis = 1)).T
        valuations_hist.append(valuations)
        budgets = lib.get_budgets(num_buyers, budgets_range, budgets_addition)
        budgets_hist.append(budgets)
        supplies = lib.get_supplies(num_goods, supplies_range, supplies_addition)
        supplies_hist.append(supplies)
        
        if utility_type == "cd":
            demands = np.zeros(valuations.shape).clip(min = 0.001)  



        ##################### Price Step #####################
        demand_row_sum = np.sum(demands, axis = 0)
        excess_demands = demand_row_sum - supplies

        if iter == 1 :
            prices = np.copy(prices_0)
            old_prices = prices
        else:
            if decay_inner:
                step_size = learning_rate[0] * (iter**(-1/2)) * excess_demands
            else:
                step_size = learning_rate[0] * excess_demands
            prices += step_size * (prices > 0)
            old_prices = prices_hist[-1]

        prices = np.clip(prices, a_min=0.001, a_max = None) # Make sure the price is positive
        prices_hist.append(prices)


        ##################### Demand Step ####################

        # Gradient Step
        constants_list = []
        for v_i, b_i, x_i in zip(valuations, budgets, demands):
            c_i = b_i / max(util_func(x_i, v_i), 0.001)
            constants_list.append(c_i)
        if decay_outer: 
            constants = (np.array(constants_list) * learning_rate[1] * (iter**(-1/2))).reshape(num_buyers, 1)
        else:
            constants = (np.array(constants_list) * learning_rate[1]).reshape(num_buyers, 1)
            
        demands += constants * util_gradient_func(demands, valuations)

        # Projection Step
        demands = project_to_bugdet_set(demands, old_prices, budgets) #is here p^(t-1) or p^(t)
        demands_hist.append(demands)
    
    return prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist




def static_LGDA(valuations, budgets, supplies, num_iters, num_goods, num_buyers, utility_type, prices_0,
valuations_range, valuations_addition,
budgets_range, budgets_addition,
supplies_range, supplies_addition,
learning_rate, decay_outer = False, decay_inner = False):

    # Set up functions to be used
    if utility_type == "linear":
        util_func = lib.get_linear_utility
        util_gradient_func = lib.get_linear_util_gradient
        obj_func = lib.get_linear_obj
    elif utility_type == "leontief":
        util_func = lib.get_leontief_utility
        util_gradient_func = lib.get_leontief_util_gradient
        obj_func = lib.get_leontief_obj
    elif utility_type == "cd":
        util_func = lib.get_cd_utility
        util_gradient_func = lib.get_cd_util_gradient
        obj_func = lib.get_cd_obj


    # Set up lists for data recording
    prices_hist = []
    demands_hist = []
    valuations_hist = []
    budgets_hist = []
    supplies_hist = []
    prices = np.zeros(num_goods)
    demands = np.zeros((num_buyers, num_goods))

    
    ################# Get the problem f_t ################
    # valuations = lib.get_valuations(num_buyers, num_goods, valuations_range, valuations_addition)
    # Normalize valuations for Cobb-Douglas
    if utility_type == "cd":
        valuations = (valuations.T/ np.sum(valuations, axis = 1)).T
    valuations_hist.append(valuations)
    # budgets = lib.get_budgets(num_buyers, budgets_range, budgets_addition)
    budgets_hist.append(budgets)
    # supplies = lib.get_supplies(num_goods, supplies_range, supplies_addition)
    supplies_hist.append(supplies)
    
    if utility_type == "cd":
        demands = np.zeros(valuations.shape).clip(min = 0.001)  


    ################### Start the iterations ####################
    for iter in range(1, num_iters):
        if (not iter % 50):
            print(f" ----- Iteration {iter}/{num_iters} ----- ")


        ##################### Price Step #####################
        demand_row_sum = np.sum(demands, axis = 0)
        excess_demands = demand_row_sum - supplies

        if iter == 1 :
            prices = np.copy(prices_0)
            old_prices = prices
        else:
            if decay_inner:
                step_size = learning_rate[0] * iter**(-1/2) * excess_demands
            else:
                step_size = learning_rate[0] * excess_demands
            # step_size = learning_rate[0] * excess_demands
            prices += step_size * (prices > 0)
            old_prices = prices_hist[-1]

        prices = np.clip(prices, a_min=0.001, a_max = None) # Make sure the price is positive
        prices_hist.append(prices)


        ##################### Demand Step ####################

        # Gradient Step
        constants_list = []
        for v_i, b_i, x_i in zip(valuations, budgets, demands):
            c_i = b_i / max(util_func(x_i, v_i), 0.001)
            constants_list.append(c_i)
        if decay_outer:
            constants = (np.array(constants_list) * learning_rate[1] * iter**(-1/2)).reshape(num_buyers, 1)
            demands += (constants * util_gradient_func(demands, valuations) - learning_rate[1] * iter**(-1/2) * old_prices)
        else:
            constants = (np.array(constants_list) * learning_rate[1]).reshape(num_buyers, 1)
            demands += (constants * util_gradient_func(demands, valuations) - learning_rate[1] * old_prices)

        # Projection Step
        # demands = project_to_bugdet_set(demands, old_prices, budgets) #is here p^(t-1) or p^(t)
        demands = demands.clip(min=0.0001)
        demands_hist.append(demands)
    
    return prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist


def dynamic_LGDA(num_iters, num_goods, num_buyers, utility_type, prices_0,
valuations_range, valuations_addition,
budgets_range, budgets_addition,
supplies_range, supplies_addition,
learning_rate, decay_outer = False, decay_inner = False):

    # Set up functions to be used
    if utility_type == "linear":
        util_func = lib.get_linear_utility
        util_gradient_func = lib.get_linear_util_gradient
        obj_func = lib.get_linear_obj
    elif utility_type == "leontief":
        util_func = lib.get_leontief_utility
        util_gradient_func = lib.get_leontief_util_gradient
        obj_func = lib.get_leontief_obj
    elif utility_type == "cd":
        util_func = lib.get_cd_utility
        util_gradient_func = lib.get_cd_util_gradient
        obj_func = lib.get_cd_obj


    # Set up lists for data recording
    prices_hist = []
    demands_hist = []
    valuations_hist = []
    budgets_hist = []
    supplies_hist = []
    prices = np.zeros(num_goods)
    demands = np.zeros((num_buyers, num_goods))

    


    ################### Start the iterations ####################
    for iter in range(1, num_iters):
        if (not iter % 200):
            print(f" ----- Iteration {iter}/{num_iters} ----- ")

        ################# Get the problem f_t ################
        valuations = lib.get_valuations(num_buyers, num_goods, valuations_range, valuations_addition)
        # Normalize valuations for Cobb-Douglas
        if utility_type == "cd":
            valuations = (valuations.T/ np.sum(valuations, axis = 1)).T
        valuations_hist.append(valuations)
        budgets = lib.get_budgets(num_buyers, budgets_range, budgets_addition)
        budgets_hist.append(budgets)
        supplies = lib.get_supplies(num_goods, supplies_range, supplies_addition)
        supplies_hist.append(supplies)
        
        if utility_type == "cd":
            demands = np.zeros(valuations.shape).clip(min = 0.001)  


        ##################### Price Step #####################
        demand_row_sum = np.sum(demands, axis = 0)
        excess_demands = demand_row_sum - supplies

        if iter == 1 :
            prices = np.copy(prices_0)
            old_prices = prices
        else:
            if decay_inner:
                step_size = learning_rate[0] * iter**(-1/2) * excess_demands
            else:
                step_size = learning_rate[0] * excess_demands
            # step_size = learning_rate[0] * excess_demands
            prices += step_size * (prices > 0)
            old_prices = prices_hist[-1]

        prices = np.clip(prices, a_min=0.001, a_max = None) # Make sure the price is positive
        prices_hist.append(prices)


        ##################### Demand Step ####################

        # Gradient Step
        constants_list = []
        for v_i, b_i, x_i in zip(valuations, budgets, demands):
            c_i = b_i / max(util_func(x_i, v_i), 0.001)
            constants_list.append(c_i)
        if decay_outer:
            constants = (np.array(constants_list) * learning_rate[1] * iter**(-1/2)).reshape(num_buyers, 1)
            demands += (constants * util_gradient_func(demands, valuations) - learning_rate[1] * iter**(-1/2) * old_prices)
        else:
            constants = (np.array(constants_list) * learning_rate[1]).reshape(num_buyers, 1)
            demands += (constants * util_gradient_func(demands, valuations) - learning_rate[1] * old_prices)

        # Projection Step
        # demands = project_to_bugdet_set(demands, old_prices, budgets) #is here p^(t-1) or p^(t)
        demands = demands.clip(min=0.0001)
        demands_hist.append(demands)
    
    return prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist



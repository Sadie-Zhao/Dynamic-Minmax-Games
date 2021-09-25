import numpy as np
import dynamicLibrary as lib
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


def dynamic_GDA_with_oracle(num_iters, num_goods, num_buyers, utility_type, prices_0,
valuations_range, valuations_addition,
budgets_range, budgets_addition,
supplies_range, supplies_addition,
learning_rate, decay_outer = False, decay_inner = False):
#if learning rate the same for each iterations?

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
    cumulative_loss_hist = [0]
    prices = np.zeros(num_goods)
    demands = np.zeros((num_buyers, num_goods))

    
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

    for iter in range(num_iters):
        if (not iter % 500):
            print(f" ----- Iteration {iter}/{num_iters} ----- ")



        ##################### Price Step #####################
        if iter == 0:
            prices = np.copy(prices_0)
            old_prices = prices
        else:
            prices = supplies - np.sum(demands, axis = 0)
            old_prices = prices_hist[-1]

        prices = np.clip(prices, 0.01, a_max = None) # Make sure the price is positive
        prices_hist.append(prices)


        ##################### Demand Step ####################

        # Gradient Step
        constants_list = []
        for v_i, b_i, x_i in zip(valuations, budgets, demands):
            c_i = b_i / max(util_func(x_i, v_i), 0.01)
            constants_list.append(c_i)
        constants = (np.array(constants_list) * learning_rate).reshape(num_buyers, 1)
        demands += constants * util_gradient_func(demands, valuations)

        # Projection Step
        demands = project_to_bugdet_set(demands, old_prices, budgets) #is here p^(t-1) or p^(t)
        demands_hist.append(demands)

        # Update cumulative loss
        lib.update_cumulative_loss(prices, demands, supplies, budgets, valuations, cumulative_loss_hist, obj_func)
    
    return prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist, cumulative_loss_hist







def dynamic_LGDA(num_iters, num_goods, num_buyers, utility_type, prices_0,
valuations_range, valuations_addition,
budgets_range, budgets_addition,
supplies_range, supplies_addition,
learning_rate, decay_outer = False, decay_inner = False):
#if learning rate the same for each iterations?

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
    cumulative_loss_hist = [0]
    prices = np.zeros(num_goods)
    demands = np.zeros((num_buyers, num_goods))
    

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


    for iter in range(num_iters):
        if (not iter % 500):
            print(f" ----- Iteration {iter}/{num_iters} ----- ")


        ##################### Price Step #####################
        if iter == 0:
            prices = np.copy(prices_0)
            old_prices = prices
        else:
            prices = supplies - np.sum(demands, axis = 0)
            old_prices = prices_hist[-1]

        prices = np.clip(prices, 0.01, a_max = None) # Make sure the price is positive
        prices_hist.append(prices)



        ##################### Demand Step ####################

        # Gradient Step
        constants_list = []
        for v_i, b_i, x_i in zip(valuations, budgets, demands):
            c_i = b_i / max(util_func(x_i, v_i), 0.01)
            constants_list.append(c_i)
        constants = (np.array(constants_list) * learning_rate).reshape(num_buyers, 1)
        demands += (constants *util_gradient_func(demands, valuations)) - old_prices

        # No projection Step
        demands_hist.append(demands)

        # Update cumulative_loss
        lib.update_cumulative_loss(prices, demands, supplies, budgets, valuations, cumulative_loss_hist, obj_func)

    return prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist, cumulative_loss_hist

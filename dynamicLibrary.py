# Library Imports
from cvxpy import constraints
import consumerUtility as cu
import numpy as np
import cvxpy as cp
from numpy.core.numeric import ones
from numpy.lib.twodim_base import triu_indices_from
np.set_printoptions(precision=3)


#################### Market Parameters #######################
def get_valuations(num_buyers, num_goods, range, addition_term):
    valuations = np.random.rand(num_buyers, num_goods) * range + addition_term
    return valuations

def get_budgets(num_buyers, range, addition_term):
    budgets = np.random.rand(num_buyers) * range + addition_term
    return budgets

def get_supplies(num_goods, range, addition_term):
    supplies = np.random.rand(num_goods) * range + addition_term
    return supplies



##################### Utility Function ########################

# Linear utility function: Perfect Substitutes
def get_linear_utility(allocation, valuations): 
    # Returns utility value for linear agent
    return allocation.T @ valuations

# Leontief utility function: Perfect Complements
def get_leontief_utility(allocation, valuations):
    return np.min(allocation / valuations)

# Cobb-Douglas utility function
def get_cd_utility(allocation, valuations):
    # For easy of calculation, normalize valautions
    # This does not change the preference relation that the utility function represents
    normalized_vals = valuations / np.sum(valuations)
    return np.prod(np.power(allocation, normalized_vals))



#################### Utility Gradient Functions #####################
def get_linear_util_gradient(allocations, valuations):
    return valuations

def get_leontief_util_gradient(allocations, valuations):
    grad_matrix = []
    for x_i, v_i in zip(allocations, valuations):
        argmin_j = np.argmin([v_ij / x_ij for v_ij, x_ij in zip(x_i, v_i)])
        grad_array = np.zeros(len(x_i))
        grad_array[argmin_j] = 1 / v_i[argmin_j]
        grad_matrix.append(grad_array)
    
    return np.array(grad_matrix)

def get_cd_util_gradient(allocations, valuations):
    return ( np.prod(np.power(allocations, valuations), axis = 1)*( valuations / allocations.clip(min = 0.001) ).T).T




################### Objective Functions ##########################

def get_linear_obj(prices, demands, supplies, budgets, valuations):
    utils = np.sum(valuations * demands, axis = 1)
    return supplies.T @ prices + budgets.T @ np.log(utils.clip(min=0.0001))


def get_leontief_obj(prices, demands, supplies, budgets, valuations):
    utils = np.min(demands/valuations, axis = 1)
    return supplies.T @ prices + budgets.T @ np.log(utils.clip(min= 0.0001))


def get_cd_obj(prices, demands, supplies, budgets, valuations):
    utils = np.prod(np.power(demands, valuations), axis= 1)
    return supplies.T @ prices + budgets.T @ np.log(utils.clip(min=0.0001))






#################### Value Functions ###################################
def get_linear_value(prices, demands, budgets, valuations):
    utils = np.zeros(budgets.shape[0])
    for buyer in range(budgets.shape[0]):
        b = budgets[buyer]
        v = valuations[buyer,:]
        utils[buyer] = cu.get_linear_indirect_utill(prices.clip(min= 0.0001), b, v)
    # return np.sum(prices) + budgets.T @ np.log(utils) + np.sum(budgets) - np.sum(demands @ prices )
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min= 0.01))

def get_cd_value(prices, demands, budgets, valuations):
    utils = np.zeros(budgets.shape[0])
    for buyer in range(budgets.shape[0]):
        b = budgets[buyer]
        v = valuations[buyer,:]
        utils[buyer] = cu.get_CD_indirect_util(prices.clip(min= 0.0001), b, v)
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min= 0.0001)) 

def get_leontief_value(prices, demands, budgets, valuations):
    utils = np.zeros(budgets.shape[0])
    for buyer in range(budgets.shape[0]):
        b = budgets[buyer]
        v = valuations[buyer,:]
        utils[buyer] = cu.get_leontief_indirect_util(prices.clip(min= 0.0001), b, v)
    return np.sum(prices) + budgets.T @ np.log(utils.clip(min= 0.0001)) 


def get_average_value_static(value_func, prices_hist, demands_hist, budgets, valuations):
    value_avg_hist = []
    for i in range(0, len(demands_hist)):
        x = np.mean(np.array(demands_hist[:i+1]).clip(min = 0), axis = 0)
        p = np.mean(np.array(prices_hist[:i+1]).clip(min = 0), axis = 0)
        value = value_func(p, x, budgets, valuations)
        value_avg_hist.append(value)
    return value_avg_hist

def get_average_value_dynamic(value_func, prices_hist, demands_hist, budgets_hist, valuations_hist):
    value_avg_hist = []
    for i in range(0, len(demands_hist)):
        x = np.mean(np.array(demands_hist[:i+1]).clip(min = 0), axis = 0)
        p = np.mean(np.array(prices_hist[:i+1]).clip(min = 0), axis = 0)
        budgets = np.mean(np.array(budgets_hist[:i+1]).clip(min = 0), axis = 0)
        valuations = np.mean(np.array(valuations_hist[:i+1]).clip(min = 0), axis = 0)
        value = value_func(p, x, budgets, valuations)
        value_avg_hist.append(value)
    return value_avg_hist




################### Cumulative Regret Functions #######################

def update_cumulative_loss(obj_hist, cumulative_loss_hist):
    cumulative_loss = cumulative_loss_hist[-1]
    cumulative_loss_hist.append(cumulative_loss + obj_hist[-1]) 


def get_p_cumulative_regret(num_buyers, num_goods, demands_hist, supplies_hist, budgets_hist, valuations_hist, cumulative_loss_hist, obj_func):
    p = cp.Variable(num_goods)
    cum_loss = np.sum([obj_func(p, demands, supplies, budgets, valuations) for demands, supplies, budgets, valuations in zip(demands_hist, supplies_hist, budgets_hist, valuations_hist)])
    objective = cp.Minimize(cum_loss)
    constr_z_matrix = [b_t - (x_t @ p) >= 0 for b_t, x_t in zip(budgets_hist, demands_hist)]
    constraints = [p >= 0]
    constraints += (constr_z_matrix)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    cum_loss_with_constant_p = prob.value

    return cumulative_loss_hist[-1]-cum_loss_with_constant_p


def get_X_cumulative_regret(num_buyers, num_goods, prices_hist, supplies_hist, budgets_hist, valuations_hist, cumulative_loss_hist, obj_func):
    X = cp.Variable(num_buyers, num_goods)
    cum_loss = np.sum([obj_func(prices, X, supplies, budgets, valuations) for prices, supplies, budgets, valuations in zip(prices_hist, supplies_hist, budgets_hist, valuations_hist)])
    objective = cp.Minimize(cum_loss)
    constr_z_matrix = [b_t - (X @ p_t) >= 0 for b_t, p_t in zip(budgets_hist, prices_hist)]
    constraints = [X >= 0]
    constraints += (constr_z_matrix)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    cum_loss_with_constant_X = prob.value

    return cumulative_loss_hist[-1] - cum_loss_with_constant_X

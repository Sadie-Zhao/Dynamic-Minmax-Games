import dynamicMinmax as dm
import dynamicLibrary as lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date
import pandas as pd


def run_linear_gd_exp(num_exp, num_iterations, num_goods, num_buyers, prices_0):
    linear_gd_p_plus_x_dist_hist = []
    linear_gd_obj_dist_hist = []
    linear_gd_value_dist_hist = []
    for iter in range(num_exp):
        prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist = dm.dynamic_max_oracle_GD(num_iterations, num_goods, num_buyers,
        "linear", prices_0, 
        10, 10,
        10, 5,
        10, 100, 
        learning_rate = 1)
    
        ######################## plot distances ################################
        p_plus_x_dists = []
        obj_dists = []
        value_dists = []
        for prices, demands, valuations, budgets, supplies in zip(prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist):
            p_plus_x_dist, obj_dist, value_dist = lib.get_dist_to_equilibrium("linear", prices, demands, valuations, budgets, supplies)
            p_plus_x_dists.append(p_plus_x_dist)
            obj_dists.append(obj_dist)
            value_dists.append(value_dist)
        linear_gd_p_plus_x_dist_hist.append(p_plus_x_dists)
        linear_gd_obj_dist_hist.append(obj_dists)
        linear_gd_value_dist_hist.append(value_dists)

    return linear_gd_p_plus_x_dist_hist, linear_gd_obj_dist_hist, linear_gd_value_dist_hist

def run_cd_gd_exp(num_exp, num_iterations, num_goods, num_buyers, prices_0):
    cd_gd_p_plus_x_dist_hist = []
    cd_gd_obj_dist_hist = []
    cd_gd_value_dist_hist = []
    for iter in range(num_exp):
        prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist = dm.dynamic_max_oracle_GD(num_iterations, num_goods, num_buyers,
        "cd", prices_0, 
        10, 10,
        10, 5,
        10, 100, 
        learning_rate = 1)
    
        ######################## plot distances ################################
        p_plus_x_dists = []
        obj_dists = []
        value_dists = []
        for prices, demands, valuations, budgets, supplies in zip(prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist):
            p_plus_x_dist, obj_dist, value_dist = lib.get_dist_to_equilibrium("cd", prices, demands, valuations, budgets, supplies)
            p_plus_x_dists.append(p_plus_x_dist)
            obj_dists.append(obj_dist)
            value_dists.append(value_dist)
        cd_gd_p_plus_x_dist_hist.append(p_plus_x_dists)
        cd_gd_obj_dist_hist.append(obj_dists)
        cd_gd_value_dist_hist.append(value_dists)

    return cd_gd_p_plus_x_dist_hist, cd_gd_obj_dist_hist, cd_gd_value_dist_hist

def run_leontief_gd_exp(num_exp, num_iterations, num_goods, num_buyers, prices_0):
    leontief_gd_p_plus_x_dist_hist = []
    leontief_gd_obj_dist_hist = []
    leontief_gd_value_dist_hist = []
    for iter in range(num_exp):
        prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist = dm.dynamic_max_oracle_GD(num_iterations, num_goods, num_buyers,
        "leontief", prices_0, 
        10, 10,
        10, 5,
        10, 100,  
        learning_rate = 1)
    
        ######################## plot distances ################################
        p_plus_x_dists = []
        obj_dists = []
        value_dists = []
        for prices, demands, valuations, budgets, supplies in zip(prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist):
            p_plus_x_dist, obj_dist, value_dist = lib.get_dist_to_equilibrium("leontief", prices, demands, valuations, budgets, supplies)
            p_plus_x_dists.append(p_plus_x_dist)
            obj_dists.append(obj_dist)
            value_dists.append(value_dist)
        leontief_gd_p_plus_x_dist_hist.append(p_plus_x_dists)
        leontief_gd_obj_dist_hist.append(obj_dists)
        leontief_gd_value_dist_hist.append(value_dists)

    return leontief_gd_p_plus_x_dist_hist, leontief_gd_obj_dist_hist, leontief_gd_value_dist_hist



def run_linear_lgda_exp(num_exp, num_iterations, num_goods, num_buyers, prices_0):
    linear_lgda_p_plus_x_dist_hist = []
    linear_lgda_obj_dist_hist = []
    linear_lgda_value_dist_hist = []
    for iter in range(num_exp):
        prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist = dm.dynamic_LGDA(num_iterations, num_goods, num_buyers,
        "linear", prices_0, 
        10, 5,
        10, 10,
        10, 5, 
        learning_rate_linear[0],
        decay_outer=True, decay_inner=True)

        ######################## plot distances ################################
        p_plus_x_dists = []
        obj_dists = []
        value_dists = []
        for prices, demands, valuations, budgets, supplies in zip(prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist):
            p_plus_x_dist, obj_dist, value_dist = lib.get_dist_to_equilibrium("linear", prices, demands, valuations, budgets, supplies)
            p_plus_x_dists.append(p_plus_x_dist)
            obj_dists.append(obj_dist)
            value_dists.append(value_dist)
        linear_lgda_p_plus_x_dist_hist.append(p_plus_x_dists)
        linear_lgda_obj_dist_hist.append(obj_dists)
        linear_lgda_value_dist_hist.append(value_dists)

    return linear_lgda_p_plus_x_dist_hist, linear_lgda_obj_dist_hist, linear_lgda_value_dist_hist


def run_cd_lgda_exp(num_exp, num_iterations, num_goods, num_buyers, prices_0):
    cd_lgda_p_plus_x_dist_hist = []
    cd_lgda_obj_dist_hist = []
    cd_lgda_value_dist_hist = []
    for iter in range(num_exp):
        prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist = dm.dynamic_LGDA(num_iterations, num_goods, num_buyers,
        "cd", prices_0, 
        10, 5,
        10, 10,
        10, 5, 
        learning_rate_cd[0],
        decay_outer=True, decay_inner=True)

        ######################## plot distances ################################
        p_plus_x_dists = []
        obj_dists = []
        value_dists = []
        for prices, demands, valuations, budgets, supplies in zip(prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist):
            p_plus_x_dist, obj_dist, value_dist = lib.get_dist_to_equilibrium("cd", prices, demands, valuations, budgets, supplies)
            p_plus_x_dists.append(p_plus_x_dist)
            obj_dists.append(obj_dist)
            value_dists.append(value_dist)
        cd_lgda_p_plus_x_dist_hist.append(p_plus_x_dists)
        cd_lgda_obj_dist_hist.append(obj_dists)
        cd_lgda_value_dist_hist.append(value_dists)

    return cd_lgda_p_plus_x_dist_hist, cd_lgda_obj_dist_hist, cd_lgda_value_dist_hist

def run_leontief_lgda_exp(num_exp, num_iterations, num_goods, num_buyers, prices_0):
    leontief_lgda_p_plus_x_dist_hist = []
    leontief_lgda_obj_dist_hist = []
    leontief_lgda_value_dist_hist = []
    for iter in range(num_exp):
        prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist = dm.dynamic_LGDA(num_iterations, num_goods, num_buyers,
        "leontief", prices_0, 
        10, 5,
        10, 10,
        10, 5, 
        learning_rate_leontief[0],
        decay_outer=True, decay_inner=True)

        ######################## plot distances ################################
        p_plus_x_dists = []
        obj_dists = []
        value_dists = []
        for prices, demands, valuations, budgets, supplies in zip(prices_hist, demands_hist, valuations_hist, budgets_hist, supplies_hist):
            p_plus_x_dist, obj_dist, value_dist = lib.get_dist_to_equilibrium("leontief", prices, demands, valuations, budgets, supplies)
            p_plus_x_dists.append(p_plus_x_dist)
            obj_dists.append(obj_dist)
            value_dists.append(value_dist)
        leontief_lgda_p_plus_x_dist_hist.append(p_plus_x_dists)
        leontief_lgda_obj_dist_hist.append(obj_dists)
        leontief_lgda_value_dist_hist.append(value_dists)

    return leontief_lgda_p_plus_x_dist_hist, leontief_lgda_obj_dist_hist, leontief_lgda_value_dist_hist


if __name__ == '__main__':
    num_iterations = 1000
    num_experiment = 1
    num_buyers =  5
    num_goods = 8 
    learning_rate_linear =  ((5,0.01), (1000**(-1/2),1000**(-1/2))) #0.001,0.0001 for gda #5,0.01 for LGDA!
    learning_rate_cd =  ((5, 0.01), (500**(-1/2), 500**(-1/2)))  #0.0001,0.001 for gda    #5,0.01 for LGDA!
    learning_rate_leontief =  ((5, 0.01), (500**(-1/2),500**(-1/2))) #0.0001,0.01 for gda #5,0.001 for LGDA

    prices_0  = np.random.rand(num_goods) * 5 + 50
    
    #################################################################### GD ################################################
    linear_gd_p_plus_x_dist_hist, linear_gd_obj_dist_hist, linear_gd_value_dist_hist = run_linear_gd_exp(num_experiment, num_iterations, num_goods, num_buyers, prices_0)
    linear_gd_p_plus_x_dist_final = (np.mean(linear_gd_p_plus_x_dist_hist, axis=0))[10:-10]
    linear_gd_obj_dist_final = (np.mean(linear_gd_p_plus_x_dist_hist, axis=0))[10:-10]
    linear_gd_value_dist_final = (np.mean(linear_gd_p_plus_x_dist_hist, axis=0))[10:-10]
    # linear_gd_p_plus_x_dist_df = pd.DataFrame(linear_gd_p_plus_x_dist_final)
    # linear_gd_obj_dist_df = pd.DataFrame(linear_gd_p_plus_x_dist_final)
    # linear_gd_value_dist_df = pd.DataFrame(linear_gd_p_plus_x_dist_final)


    cd_gd_p_plus_x_dist_hist, cd_gd_obj_dist_hist, cd_gd_value_dist_hist = run_cd_gd_exp(num_experiment, num_iterations, num_goods, num_buyers, prices_0)
    cd_gd_p_plus_x_dist_final = (np.mean(cd_gd_p_plus_x_dist_hist, axis=0))[10:-10]
    cd_gd_obj_dist_final = (np.mean(cd_gd_p_plus_x_dist_hist, axis=0))[10:-10]
    cd_gd_value_dist_final = (np.mean(cd_gd_p_plus_x_dist_hist, axis=0))[10:-10]
    # cd_gd_p_plus_x_dist_df = pd.DataFrame(cd_gd_p_plus_x_dist_final)
    # cd_gd_obj_dist_df = pd.DataFrame(cd_gd_p_plus_x_dist_final)
    # cd_gd_value_dist_df = pd.DataFrame(cd_gd_p_plus_x_dist_final)

    leontief_gd_p_plus_x_dist_hist, leontief_gd_obj_dist_hist, leontief_gd_value_dist_hist = run_leontief_gd_exp(num_experiment, num_iterations, num_goods, num_buyers, prices_0)
    leontief_gd_p_plus_x_dist_final = (np.mean(leontief_gd_p_plus_x_dist_hist, axis=0))[:-10]
    leontief_gd_obj_dist_final = (np.mean(leontief_gd_p_plus_x_dist_hist, axis=0))[10:-10]
    leontief_gd_value_dist_final = (np.mean(leontief_gd_p_plus_x_dist_hist, axis=0))[10:-10]
    # leontief_gd_p_plus_x_dist_df = pd.DataFrame(leontief_gd_p_plus_x_dist_final)
    # leontief_gd_obj_dist_df = pd.DataFrame(leontief_gd_p_plus_x_dist_final)
    # leontief_gd_value_dist_df = pd.DataFrame(leontief_gd_p_plus_x_dist_final)



    ##################################################### LGDA ##########################################################
    linear_lgda_p_plus_x_dist_hist, linear_lgda_obj_dist_hist, linear_lgda_value_dist_hist = run_linear_lgda_exp(num_experiment, num_iterations, num_goods, num_buyers, prices_0)
    linear_lgda_p_plus_x_dist_final = (np.mean(linear_lgda_p_plus_x_dist_hist, axis=0))[10:-10]
    linear_lgda_obj_dist_final = (np.mean(linear_lgda_p_plus_x_dist_hist, axis=0))[10:-10]
    linear_lgda_value_dist_final = (np.mean(linear_lgda_p_plus_x_dist_hist, axis=0))[10:-10]
    # linear_lgda_p_plus_x_dist_df = pd.DataFrame(linear_lgda_p_plus_x_dist_final)
    # linear_lgda_obj_dist_df = pd.DataFrame(linear_lgda_p_plus_x_dist_final)
    # linear_lgda_value_dist_df = pd.DataFrame(linear_lgda_p_plus_x_dist_final)


    cd_lgda_p_plus_x_dist_hist, cd_lgda_obj_dist_hist, cd_lgda_value_dist_hist = run_cd_lgda_exp(num_experiment, num_iterations, num_goods, num_buyers, prices_0)
    cd_lgda_p_plus_x_dist_final = (np.mean(cd_lgda_p_plus_x_dist_hist, axis=0))[10:-10]
    cd_lgda_obj_dist_final = (np.mean(cd_lgda_p_plus_x_dist_hist, axis=0))[10:-10]
    cd_lgda_value_dist_final = (np.mean(cd_lgda_p_plus_x_dist_hist, axis=0))[10:-10]
    # cd_lgda_p_plus_x_dist_df = pd.DataFrame(cd_lgda_p_plus_x_dist_final)
    # cd_lgda_obj_dist_df = pd.DataFrame(cd_lgda_p_plus_x_dist_final)
    # cd_lgda_value_dist_df = pd.DataFrame(cd_lgda_p_plus_x_dist_final)

    leontief_lgda_p_plus_x_dist_hist, leontief_lgda_obj_dist_hist, leontief_lgda_value_dist_hist = run_leontief_lgda_exp(num_experiment, num_iterations, num_goods, num_buyers, prices_0)
    leontief_lgda_p_plus_x_dist_final = (np.mean(leontief_lgda_p_plus_x_dist_hist, axis=0))[10:-10]
    leontief_lgda_obj_dist_final = (np.mean(leontief_lgda_p_plus_x_dist_hist, axis=0))[10:-10]
    leontief_lgda_value_dist_final = (np.mean(leontief_lgda_p_plus_x_dist_hist, axis=0))[10:-10]
    # leontief_lgda_p_plus_x_dist_df = pd.DataFrame(leontief_lgda_p_plus_x_dist_final)
    # leontief_lgda_obj_dist_df = pd.DataFrame(leontief_lgda_p_plus_x_dist_final)
    # leontief_lgda_value_dist_df = pd.DataFrame(leontief_lgda_p_plus_x_dist_final)


    ##################################################### GD graph ###################################
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

    plt.rc('font', **font)
    num_iter = len(linear_gd_value_dist_final)
    x_forall = np.linspace(1, num_iter, num_iter)

    fig, axs = plt.subplots(1, 3) # Create a figure containing a single axes.
    # First row for experiments with low initial prices and
    # second row for experiments with high initial prices.
    
    # Add shift in plots to make the difference clearer
    axs[0].plot([iter for iter in range(num_iter)], linear_gd_p_plus_x_dist_final, alpha = 1, color = "b")
    axs[0].plot((linear_gd_p_plus_x_dist_final[0] - linear_gd_p_plus_x_dist_final[-1],)/(x_forall**(1/2)) + linear_gd_p_plus_x_dist_final[-1], color='red', linestyle='dashed', label = r"$1/\sqrt{T}$")
    axs[0].set_title("Linear Market", fontsize = "medium")
    # axs[0,0].set_ylim(2100, 2500)

    axs[1].plot([iter for iter in range(num_iter)], cd_gd_p_plus_x_dist_final , color = "b")
    # axs[0,1].plot(x, (obj_gda_cd[0]/3)/x + obj_gda_cd[-1], color='green', linestyle='dashed', label = "1/T")
    axs[1].plot(x_forall, (cd_gd_p_plus_x_dist_final[0] - cd_gd_p_plus_x_dist_final[-1])/(x_forall**(1/2)) + cd_gd_p_plus_x_dist_final[-1], color='red', linestyle='dashed', label = r"$1/\sqrt(T)$")
    axs[1].set_title("Cobb-Douglas Market", fontsize = "medium")
    # axs[0,1].set_ylim(-330, 200)

    num_iter_leon_gd = len(leontief_gd_p_plus_x_dist_final)
    axs[2].plot([iter for iter in range(num_iter_leon_gd)], leontief_gd_p_plus_x_dist_final, color = "b")
    # axs[2].plot(x_forall, (leontief_gd_p_plus_x_dist_final[0] - leontief_gd_p_plus_x_dist_final[-1])/(x_forall**(1/3)) + leontief_gd_p_plus_x_dist_final[-1], color='green', linestyle='dashed', label = r"$1/T^{\frac{1}{3}}$")
    axs[2].plot(x_forall, (leontief_gd_p_plus_x_dist_final[0] - leontief_gd_p_plus_x_dist_final[-1])/(x_forall**(1/2)) + leontief_gd_p_plus_x_dist_final[-1], color='red', linestyle='dashed', label = r"$1/\sqrt(T)$")
    axs[2].set_title("Leontief Market", fontsize = "medium")


    for ax in axs.flat:
        ax.set(xlabel='Iteration Number', ylabel='Distance to Equilibrium')
        # ax.yaxis.set_ticks([])
    for ax in axs.flat:
        ax.label_outer()

    name = "gd_pplusx_dist_graphs_1_runs"

    fig.set_size_inches(18.5, 6.5)
    plt.savefig(f"graphs/{name}.jpg")
    plt.show()

    ######################################### LGDA graph #########################################
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

    plt.rc('font', **font)
    num_iter = len(linear_lgda_value_dist_final)
    x_forall = np.linspace(1, num_iter, num_iter)

    fig, axs = plt.subplots(1, 3) # Create a figure containing a single axes.
    # First row for experiments with low initial prices and
    # second row for experiments with high initial prices.
    
    # Add shift in plots to make the difference clearer
    axs[0].plot([iter for iter in range(num_iter)], linear_lgda_p_plus_x_dist_final, alpha = 1, color = "b")
    # axs[0].plot((linear_lgda_p_plus_x_dist_final[0] - linear_lgda_p_plus_x_dist_final[-1],)/(x_forall**(1/3)) + linear_lgda_p_plus_x_dist_final[-1], color='red', linestyle='dashed', label = r"$1/T^{\frac{1}{3}}$")
    axs[0].plot((linear_lgda_p_plus_x_dist_final[0] - linear_lgda_p_plus_x_dist_final[-1],)/(x_forall**(1/2)) + linear_lgda_p_plus_x_dist_final[-1], color='red', linestyle='dashed', label = r"$1/\sqrt{T}$")
    axs[0].set_title("Linear Market", fontsize = "medium")
    # axs[0,0].set_ylim(2100, 2500)

    axs[1].plot([iter for iter in range(num_iter)], cd_lgda_p_plus_x_dist_final , color = "b")
    # axs[0,1].plot(x, (obj_lgdaa_cd[0]/3)/x + obj_lgda_cd[-1], color='green', linestyle='dashed', label = "1/T")
    axs[1].plot(x_forall, (cd_lgda_p_plus_x_dist_final[0] - cd_lgda_p_plus_x_dist_final[-1])/(x_forall**(1/2)) + cd_lgda_p_plus_x_dist_final[-1], color='red', linestyle='dashed', label = r"$1/\sqrt(T)$")
    axs[1].set_title("Cobb-Douglas Market", fontsize = "medium")
    # axs[0,1].set_ylim(-330, 200)

    axs[2].plot([iter for iter in range(num_iter)], leontief_lgda_p_plus_x_dist_final, color = "b")
    # axs[2].plot(x_forall, (leontief_gd_p_plus_x_dist_final[0] - leontief_gd_p_plus_x_dist_final[-1])/(x_forall**(1/3)) + leontief_gd_p_plus_x_dist_final[-1], color='green', linestyle='dashed', label = r"$1/T^{\frac{1}{3}}$")
    axs[2].plot(x_forall, (leontief_lgda_p_plus_x_dist_final[0] - leontief_lgda_p_plus_x_dist_final[-1])/(x_forall**(1/2)) + leontief_lgda_p_plus_x_dist_final[-1], color='red', linestyle='dashed', label = r"$1/\sqrt(T)$")
    axs[2].set_title("Leontief Market", fontsize = "medium")


    for ax in axs.flat:
        ax.set(xlabel='Iteration Number', ylabel='Distance to Equilibrium')
        # ax.yaxis.set_ticks([])
    for ax in axs.flat:
        ax.label_outer()

    name = "lgda_pplusx_dist_graphs_1_runs"

    fig.set_size_inches(18.5, 6.5)
    plt.savefig(f"graphs/{name}.jpg")
    plt.show()

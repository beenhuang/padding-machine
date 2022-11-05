#!/usr/bin/env python3

"""
<file>    dist_fit.py
<brief>   fit data to distribution
"""

from scipy import stats
from scipy.stats import uniform, logistic, fisk, geom, weibull_min, genpareto

import numpy as np
import matplotlib.pyplot as plt


def fit_data_to_dist(data, dist):
    if dist == "uniform":
        return data_fit_to_uniform(data)
    elif dist == "logistic":
        return data_fit_to_logistic(data)
    elif dist == "fisk":  
        return data_fit_to_fisk(data)
    elif dist == "geom":  
        return data_fit_to_geometric(data)
    elif dist == "weibull":  
        return data_fit_to_weibull(data)
    elif dist == "pareto":  
        return data_fit_to_pareto(data)
    else:
        print("[ERROR] unrecognized distribution.")


def data_fit_to_uniform(data):
    uni_res = uniform.fit(data)

    return f"[uniform]: (param1, lower):{uni_res[-2]}, (param2, upper):{uni_res[-1]}\n"


def data_fit_to_logistic(data):
    log_res = logistic.fit(data)

    return f"[logistic]: (param1, mu, loc):{log_res[-2]}, (param2, sigma, scale):{log_res[-1]}\n"


def data_fit_to_fisk(data):
    fisk_res = fisk.fit(data, floc=0)

    return f"[log-logistic]: (C, beta/alpha):{fisk_res[:-2]} (loc):{fisk_res[-2]}, (param1, alpha, scale):{fisk_res[-1]}, (param2, 1.0/beta):{1.0/(fisk_res[-1]*fisk_res[-3])}\n"


def data_fit_to_geometric(data):
    geom_res = stats.fit(geom, data)

    return f"[geomeotric]: {geom_res.params}\n"


def data_fit_to_weibull(data):
    weibull_res = weibull_min.fit(data, floc=0)

    return f"[weibull]: (param1, k, shape):{weibull_res[-3]}, (loc):{weibull_res[-2]}, (param2, lambda, scale):{weibull_res[-1]}\n"


def data_fit_to_pareto(data):
    pareto_res = genpareto.fit(data, floc=0)

    return f"[generialized pareto]: (param2, xi, shape):{pareto_res[-3]}, (loc, mu):{pareto_res[-2]}, (param1, sigma, scale):{pareto_res[-1]}\n"


 
if __name__ == "__main__":
    #
    data = np.random.randint(low=1, high=100, size=1000)
    #print(data)

    result = data_fit_to_geometric(data)
    print(result)

    result.plot()
    plt.show()


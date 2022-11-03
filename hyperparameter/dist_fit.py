#!/usr/bin/env python3

"""
<file>    dist_fit.py
<brief>   fit data to distribution
"""

from distfit import distfit
from scipy import stats
from scipy.stats import uniform, logistic, fisk, geom, exponweib, genpareto

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

    return f"[uniform]: {uni_res}\n"


def data_fit_to_logistic(data):
    log_res = logistic.fit(data)

    return f"[logistic]: {log_res}\n"


def data_fit_to_fisk(data):
    fisk_res = fisk.fit(data)

    return f"[log-logistic]: {fisk_res}\n"


def data_fit_to_geometric(data):
    geom_res = stats.fit(geom, data)

    return f"[geomeotric]: {geom_res.params}\n\n"


def data_fit_to_weibull(data):
    weibull_res = exponweib.fit(data)

    return f"[weibull]: {weibull_res}\n"


def data_fit_to_pareto(data):
    pareto_res = genpareto.fit(data, floc=0)

    return f"[generialized pareto]: {pareto_res}\n"

 
if __name__ == "__main__":
    #
    data = np.random.randint(low=1, high=100, size=1000)
    #print(data)

    result = data_fit_to_geometric(data)
    print(result)

    result.plot()
    plt.show()


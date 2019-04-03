import numpy as np

def running_mean(x):
    mu = 0
    mean_values = []
    for k in np.arange(0, len(x)):
        mu = mu + (1.0/(k+1))*(x[k] - mu)
        mean_values.append(mu)
    return mean_values
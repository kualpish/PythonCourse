import numpy as np

def Linear_Regression(x, y):
    
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    sum_of_cross_deviations_YX = np.sum(x * y) - len(x) * y_mean * x_mean
    sum_of_squared_devations_X = np.sum(x**2) - len(x) * x_mean * x_mean

    beta1 = sum_of_cross_deviations_YX / sum_of_squared_devations_X
    beta0 = y_mean - beta1 * x_mean
    estimatedY = beta0 + beta1 * x
    
    return (estimatedY, beta0, beta1)
    


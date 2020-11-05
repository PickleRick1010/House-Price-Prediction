from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd

#Gather data

data = load_boston()

df = pd.DataFrame(data = data.data , columns = data.feature_names)

features = df.drop(["INDUS" , "AGE"] , axis = 1)

log_prices = np.log(data.target)

target = pd.DataFrame(log_prices, columns=["PRICE"])

property_stats = np.ndarray(shape = (1,11))

property_stats = features.mean().values.reshape(1, 11)

regr = LinearRegression()

regr.fit(features,target)

fitted_vals = regr.predict(features)

MSE = mean_squared_error(target , fitted_vals)

RMSE = np.sqrt(MSE)

def get_log_estimates(nr_rooms , students_per_classroom , next_to_river = False , high_confidence = True):

    #Configure property
    property_stats[0,4] = nr_rooms

    property_stats[0,8] = students_per_classroom

    if next_to_river:
        property_stats[0,2] = 1
    else:
        property_stats[0,2] = 0

    #Make Prediction
    log_estimate = regr.predict(property_stats)[0,0]

    #Calc Range

    if high_confidence:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE

        interval = 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE

        interval = 68

    return (log_estimate, upper_bound, lower_bound, interval)
    
def get_dollar_est(rm , ptratio , chas = False , large_range = True):

    """
        Estimate the price of a property in Boston

        Keyword arguments: 
        rm -- no. of rooms in the property
        ptratio -- ratio of number of students per teacher in the classroom for a school area
        chas -- if True then property is near the charles river,False otherwise
        large_range -- True for a 95% prediction interval ,False for a 68% prediction interval

    """

    modern_median_price = 583.322

    scale_factor = modern_median_price/np.median(data.target)

    if rm < 1 or ptratio <1:
        print("That is unrealistic")
        return

    log_est , upper , lower , coef = get_log_estimates(rm ,ptratio ,next_to_river = chas , high_confidence = large_range)

    #Convcerting to today's prices
    dollar_est = (np.e**log_est) * 1000 * scale_factor
    high = (np.e**upper) * 1000 * scale_factor
    low = (np.e**lower) * 1000 * scale_factor

    #Rounding the price to nearest thousands
    rounded_est = np.around(dollar_est , -3)
    rounded_high = np.around(high , -3)
    rounded_low = np.around(low , -3)

    return(f"The estimated property value is: {rounded_est} \nAt {coef}% confidence the valuation range is\nusd {rounded_low} to usd {rounded_high}. ")
    

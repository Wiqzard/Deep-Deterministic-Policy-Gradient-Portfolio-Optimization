import itertools
import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import ndarray as ndarray

num_assets: int = 8  # number of preselected non-cash assets
num_features: int = 3  # closing, highest and lowest price during period
num_periods: int = 50  # number of periods before t

num_data_points: int = 2000

# available data per coin
raw_data_matrix: ndarray = np.random.rand(num_features, num_periods, num_assets)
data_asset_i: ndarray = np.random.rand(num_features, num_data_points)
df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))

# df = pd.DataFrame(
#   np.random.randint(0, 100, size=(100, 3)), columns=["Open", "High", "Close"]
# )
print(df.tail())

# The actual prices for the assets at period t
v_t: ndarray = np.random.rand(num_assets)
v_high_t: ndarray = np.random.rand(num_assets)
v_low_t: ndarray = np.random.rand(num_assets)

# Returns a normalized price matrix for asset
def normalized_price_martix_asset(data_asset: ndarray, idx: int) -> ndarray:
    temp_x_t = data_asset[:, idx - num_periods + 1 : idx + 1]
    # V_x_t = np.divide(temp_x_t, data_asset[:, idx].view(1))
    V_x_t = temp_x_t / data_asset[:, idx, None]
    return V_x_t[:, :, np.newaxis]


# Put in whole datamatrix [feature_number, num_data_points, number_assets] + idx >,  pandas dataframe?
def normalized_price_matrix(data_matrix: ndarray, idx: int) -> ndarray:
    assert idx >= num_periods
    X_t = np.empty(data_matrix.shape)
    for _ in range(num_assets):
        X_t = np.concatenate(
            (X_t, normalized_price_martix_asset(data_asset_i, idx)),
            axis=2,
        )
    X_t = X_t[:, :, 1:]
    return X_t


# -> is the correct price tensor
print(normalized_price_martix_asset(data_asset_i, 333).shape)
print(
    normalized_price_matrix(raw_data_matrix, 333).shape
)  # [num_features, num_periods, num_assets]


action: ndarray = np.random.rand(num_assets)
# action = action / action.sum(axis=1, keepdims=1)

"""
* price tensor X_t = [feature_number, number_periods, number_assets]
                = cat((V_t, V^high_t, V^lo_t), 1) 
    With V^x_t = [v_t-n+1 / v_t | ] 
* states: s_t=(price tensor X_t, action_t-1)
* action: number_assets x 1 array
* (state_t, action_t) -> (state_t+1=(price_t+1, action_t), action_t+1)
    -> (state_t+2=(price_t+2, action_t+1), action_t+2) -> ...
* prices are independent of a


"""

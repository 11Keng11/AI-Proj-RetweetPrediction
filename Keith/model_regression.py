import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import pickle

data_train = pd.read_csv('datasets/mod_train_1.csv')
x_train = data_train.iloc[:, 0:-1]
y_train = data_train.iloc[:,-1]

sc = StandardScaler()
X_train = sc.fit_transform(x_train)

# Comment the rest out when calling this script from another script
# regressor = RandomForestRegressor(n_estimators=200, random_state=0)
# regressor.fit(X_train, y_train)

# filename = 'models/randomForestRegressor.sav'
# pickle.dump(regressor, open(filename, 'wb'))







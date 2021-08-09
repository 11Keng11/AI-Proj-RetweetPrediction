import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
# from scitime import RuntimeEstimator
import pickle

print("Reading train data...")
data_train = pd.read_csv('datasets/final_train_mod_1.csv')
x_train = data_train.iloc[:, 0:-2]
y_train = data_train.iloc[:,-2]

print("Normalizing train data and saving scaler...")
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
pickle.dump(sc, open("models/scReg.pkl", 'wb'))

print("Training model...")
regressor = RandomForestRegressor(n_estimators=100, random_state=0).fit(X_train, y_train)

print("Saving model...")
pickle.dump(regressor, open('models/randomForestRegressor.sav', 'wb'))

print("Reading test data...")
data_test = pd.read_csv('datasets/final_test_mod_1.csv')
x_test = data_train.iloc[:, 0:-2]
y_test = data_train.iloc[:,-2]

print("Normalizing test data...")
X_test = sc.transform(x_test)
y_pred = regressor.predict(X_test)

print("Getting results...")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Log Error', metrics.mean_squared_log_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))




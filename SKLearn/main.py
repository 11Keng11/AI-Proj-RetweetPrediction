import numpy as np
import pandas as pd
from sklearn import metrics
import pickle
from tqdm import tqdm

print("Loading Models")
model_classifier = pickle.load(open('models/randomForestClassifier.sav', 'rb'))
model_regressor = pickle.load(open('models/randomForestRegressor.sav', 'rb'))
sc_bin = pickle.load(open('models/scBin.pkl', 'rb'))
sc_scaler = pickle.load(open('models/scReg.pkl', 'rb'))

print("Reading data...")
data_test = pd.read_csv('datasets/final_test_mod.csv')
x_test = data_test.iloc[:, :-2]
y_test = data_test.iloc[:, -2]
X_test = sc_bin.transform(x_test)
y_pred = [0]*len(X_test)
y_idx = []
x_1 = []

print("Predicting if retweet...")
temp_pred = model_classifier.predict(X_test)

print("Extracting if retweet is true...")
for idx in tqdm(range(len(temp_pred))):
    if temp_pred[idx] != 0:
        row = x_test.iloc[idx]
        row_fit = sc_scaler.transform([row])
        x_1.append(row_fit[0])
        y_idx.append(idx)

print("Predicting number of retweets...")
preds = model_regressor.predict(x_1)
for pred_id in tqdm(range(len(preds))):
    idx = y_idx[pred_id]
    y_pred[idx] = preds[pred_id]

y_pred = np.array(y_pred)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Log Error', metrics.mean_squared_log_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))



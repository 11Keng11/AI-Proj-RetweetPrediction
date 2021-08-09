import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_absolute_error
import pickle

print("Reading train data...")
data_train = pd.read_csv('datasets/final_train_mod.csv')
x_train = data_train.iloc[:, :-2]
y_train = data_train.iloc[:,-1]

print("Normalizing train data...")
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
pickle.dump(sc, open("models/scBin.pkl", 'wb'))

print("Training model...")
classifier = RandomForestClassifier(n_estimators=100, random_state=0).fit(X_train, y_train)

print("Saving model...")
pickle.dump(classifier, open('models/randomForestClassifier.sav', 'wb'))

print("Reading test data...")
data_train = pd.read_csv('datasets/final_test_mod.csv')
x_test = data_train.iloc[:,:-2]
y_test = data_train.iloc[:,-1]

print("Normalizing test data...")
X_test = sc.transform(x_test)
y_pred = classifier.predict(X_test)

print("Getting results...")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))



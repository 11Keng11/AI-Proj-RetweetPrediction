import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

data_train = pd.read_csv('datasets/mod_train.csv')
x_train = data_train.iloc[:, 0:-2]
y_train = data_train.iloc[:,-1]

# Only used for Random Forest
sc = StandardScaler()
X_train = sc.fit_transform(x_train)

# Comment all these out when calling this script from another script
# classifier = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=0).fit(X_train, y_train)
# classifier = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(x_train, y_train)
# classifier = svm.LinearSVC().fit(x_train, y_train)

# filename = 'models/logisticRegression.sav'
# pickle.dump(classifier, open(filename, 'wb'))






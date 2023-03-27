# Import required libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings("ignore")

# Add any other classifiers you want to try here

# Load data
dataset=pd.read_csv("Data_Fix_Final.csv")
X=dataset.drop(['pelvic tilt'] , axis=1)
y=dataset['pelvic tilt']
print (X)
print(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the classifiers to train
classifiers = [
    LogisticRegression(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    SVC(),
    KNeighborsClassifier(),
    GaussianNB(),
    MLPClassifier(),
    #XGBClassifier(),
    #LGBMClassifier(),
    #CatBoostClassifier()
    # Add any other classifiers you want to try here
]

# Train each classifier and print its accuracy
for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # print results
    print("Actual Values", list(y_test.values))
    print("Predicted Values", y_pred)
    print(f"{clf.__class__.__name__}: {acc}")
    print(f"MSE: {mse}")
    print(f"R-squared: {r2}")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier
#from catboost import CatBoostClassifier
import pandas as pd

# Load data
dataset=pd.read_csv("Data_Fix_Final.csv")
X=dataset.drop(['pelvic tilt'] , axis=1)
y=dataset['pelvic tilt']
print (X)
print(y)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize a dictionary to store the accuracy scores of each algorithm
accuracy_scores = {}

# Create and fit the models
linear_regression = LinearRegression().fit(X_train, y_train)
accuracy_scores['Linear Regression'] = accuracy_score(y_test, linear_regression.predict(X_test))

logistic_regression = LogisticRegression().fit(X_train, y_train)
accuracy_scores['Logistic Regression'] = accuracy_score(y_test, logistic_regression.predict(X_test))

svm = SVC().fit(X_train, y_train)
accuracy_scores['SVM'] = accuracy_score(y_test, svm.predict(X_test))

decision_tree = DecisionTreeClassifier().fit(X_train, y_train)
accuracy_scores['Decision Tree'] = accuracy_score(y_test, decision_tree.predict(X_test))

random_forest = RandomForestClassifier().fit(X_train, y_train)
accuracy_scores['Random Forest'] = accuracy_score(y_test, random_forest.predict(X_test))

gradient_boosting = GradientBoostingClassifier().fit(X_train, y_train)
accuracy_scores['Gradient Boosting'] = accuracy_score(y_test, gradient_boosting.predict(X_test))

naive_bayes = GaussianNB().fit(X_train, y_train)
accuracy_scores['Naive Bayes'] = accuracy_score(y_test, naive_bayes.predict(X_test))

knn = KNeighborsClassifier().fit(X_train, y_train)
accuracy_scores['KNN'] = accuracy_score(y_test, knn.predict(X_test))

mlp = MLPClassifier().fit(X_train, y_train)
accuracy_scores['MLP'] = accuracy_score(y_test, mlp.predict(X_test))

#xgboost = XGBClassifier().fit(X_train, y_train)
#accuracy_scores['XGBoost'] = accuracy_score(y_test, xgboost.predict(X_test))

#lgbm = LGBMClassifier().fit(X_train, y_train)
#accuracy_scores['LightGBM'] = accuracy_score(y_test, lgbm.predict(X_test))

#catboost = CatBoostClassifier().fit(X_train, y_train)
#accuracy_scores['CatBoost'] = accuracy_score(y_test, catboost.predict(X_test))

# Print the accuracy scores
for key, value in accuracy_scores.items():
    print(f'{key}: {value}')

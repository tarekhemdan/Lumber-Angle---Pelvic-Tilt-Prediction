# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import BayesianRidge, ARDRegression, Lars, LassoLars, OrthogonalMatchingPursuit, PassiveAggressiveRegressor, HuberRegressor, TheilSenRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")


# Add any other classifiers you want to try here

# Load data
dataset=pd.read_csv("Data_Fix_Final.csv")


X=dataset.drop(['lumbar angle'] , axis=1)
y=dataset['lumbar angle']
print (X)
print(y)


# Define the target variable and features
target = 'lumbar angle'
features = [col for col in dataset.columns if col != target]

# Normalize the features using StandardScaler
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.15, random_state=42)

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(dataset[features], dataset[target], test_size=0.15, random_state=42)


# Create a list of regression algorithms to loop through
models = [
    LinearRegression(),
    Lasso(),
    Ridge(),
    ElasticNet(),
    AdaBoostRegressor(),
    XGBRegressor(),
    CatBoostRegressor(),
    NuSVR(),
    SVR(kernel='poly'),
    SVR(kernel='rbf'),
    BayesianRidge(),
    #ARDRegression(),
    LassoLars(),
    OrthogonalMatchingPursuit(),
    PassiveAggressiveRegressor(),
    #HuberRegressor(),
    #TheilSenRegressor(),
    LogisticRegression(),
    KNeighborsRegressor(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    GradientBoostingRegressor(),
    XGBRegressor(),
    SVC()
    #XGBClassifier(),
    #LGBMClassifier(),
    #CatBoostClassifier()
]

# Loop through the regression algorithms to make predictions and evaluate performance
mse_scores = {}
for model in models:
    model_name = type(model).__name__
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse_scores[model_name] = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print({model_name})
    print( "y_test:", list(y_test.values))
    print(" y_pred:", y_pred)
    print("MSE:  ", mse_scores[model_name] )
    print(f"R-squared: {r2}")
    print("=================================================================")

# Print the MSE scores for all regression algorithms
#print(mse_scores)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load the data into a pandas dataframe
df = pd.read_csv("//home//rajesh//Desktop//TGT//FEB//Week 1//Practice//Predicting House Prices//HousingData.csv")

# Split the data into features (X) and target (y)
X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_num = X.select_dtypes(include=[np.number])
X[X_num.columns] = imputer.fit_transform(X_num)

# Handle categorical features
cat_features = X.dtypes[X.dtypes == "object"].index
le = LabelEncoder()
X[cat_features] = X[cat_features].apply(lambda x: le.fit_transform(x))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regressor = RandomForestRegressor(n_estimators=100, max_depth=10)
regressor.fit(X_train, y_train)

# Evaluate the model
y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-Squared:", r2)

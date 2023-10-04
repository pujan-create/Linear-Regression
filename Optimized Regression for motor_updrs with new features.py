import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer

# Read dataset into a DataFrame
df = pd.read_csv("Ass3_updated.csv")

df = df.drop(["subject#","age","test_time","sex"], axis=1) # These columns are dropped to make sure that we are only considering the acoustic features
#print(df.info())  # we can uncomment this to check if the columns are excluded or not


#Gaussian transformatiom APPLY POWER TRANSFORMER

# separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,:-2]
y = df.iloc[:,-2]


# Create a Yeo-Johnson transformer
scaler = PowerTransformer()

# Apply the transformer to make all explanatory variables more Gaussian-looking
std_x= scaler.fit_transform(x.values)

# Restore column names of explanatory variables
std_x_df = pd.DataFrame(std_x, index=x.index, columns=x.columns)



# REBUILD AND REEVALUATE LINEAR REGRESSION USING STATSMODELS USING MORE GAUSSIAN-LIKE EXPLANATORY VARIABLES


# Build and evaluate the linear regression model
std_x_df = sm.add_constant(std_x_df)
model = sm.OLS(y,std_x_df).fit()
pred = model.predict(std_x_df)
model_details = model.summary()
print(model_details)




#building linear regression with Optimized/transformed model

# Separate explanatory variables (x) from the response variable (y)
x = std_x_df.drop(['const'], axis=1)


# Note: other % split can be used.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

# Build a linear regression model
model = LinearRegression()

# Train (fit) the linear regression model using the training set
model.fit(X_train, y_train)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of (y) in the test set
# based on the values of x in the test set
y_pred = model.predict(X_test)

# Optional: Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df_pred)

# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test, y_pred)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred))
# Normalised Root Mean Square Error
y_max = y.max()
y_min = y.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test, y_pred)

print("MLP performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)

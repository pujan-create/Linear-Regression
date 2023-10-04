#---for response variable Total_updrs---
import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import statsmodels.api as sm
# Read dataset into a DataFrame
df = pd.read_csv("Ass3.csv")

#separation of explanatory variables (x) and response variable motor updrs(y)


columns_to_drop=['subject#','age','sex','test_time']
df = df.drop(columns=columns_to_drop)

x = df.iloc[:,:-2].values #print("explanatory variable",x)
y = df.iloc[:,-1].values  # print("target variable",y)


# Split dataset into 50% training and 50% test sets 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

#build regression model
model= LinearRegression()

# Train (fit) the linear regression model using the training set
model.fit(x_train, y_train)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model.intercept_)
print("Coefficient: ", model.coef_)

# Use linear regression to predict the values of target variable(y) in the test set
# based on the values of x in the test set
y_pred = model.predict(x_test)

# Show the predicted values of (y) next to the actual values of (y)
df_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print(df_pred)

#evaluation
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

print("MLP performance on split 50-50:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)


# Split dataset into 60% training and 40% test sets 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)

#build regression model
model5= LinearRegression()

# Train (fit) the linear regression model using the training set
model5.fit(x_train, y_train)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model5.intercept_)
print("Coefficient: ", model5.coef_)

# Use linear regression to predict the values of target variable(y) in the test set
# based on the values of x in the test set
y_pred5 = model5.predict(x_test)

# Show the predicted values of (y) next to the actual values of (y)
df_pred5 = pd.DataFrame({"Actual": y_test, "Predicted": y_pred5})
print(df_pred5)

#evaluation
# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test, y_pred5)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred5)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred5))
# Normalised Root Mean Square Error
y_max = y.max()
y_min = y.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test, y_pred5)

print("MLP performance on split 60-40:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)

# Split dataset into 70% training and 30% test sets 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#build regression model
model3= LinearRegression()

# Train (fit) the linear regression model using the training set
model3.fit(x_train, y_train)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model3.intercept_)
print("Coefficient: ", model3.coef_)

# Use linear regression to predict the values of target variable(y) in the test set
# based on the values of x in the test set
y_pred3 = model3.predict(x_test)

# Show the predicted values of (y) next to the actual values of (y)
df_pred3 = pd.DataFrame({"Actual": y_test, "Predicted": y_pred3})
print(df_pred3)

#evaluation
# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test, y_pred3)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred3)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred3))
# Normalised Root Mean Square Error
y_max = y.max()
y_min = y.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test, y_pred3)

print("MLP performance on split 70-30:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)

# Split dataset into 80% training and 20% test sets 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

#build regression model
model2= LinearRegression()

# Train (fit) the linear regression model using the training set
model2.fit(x_train, y_train)

# Print the intercept and coefficient learned by the linear regression model
print("Intercept: ", model2.intercept_)
print("Coefficient: ", model2.coef_)

# Use linear regression to predict the values of target variable(y) in the test set
# based on the values of x in the test set
y_pred2 = model2.predict(x_test)

# Show the predicted values of (y) next to the actual values of (y)
df_pred2 = pd.DataFrame({"Actual": y_test, "Predicted": y_pred2})
print(df_pred2)

#evaluation
# Compute standard performance metrics of the linear regression:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test, y_pred2)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred2)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred2))
# Normalised Root Mean Square Error
y_max = y.max()
y_min = y.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test, y_pred2)

print("MLP performance on split 80-20:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)




#COMPARE THE PERFORMANCE OF THE LINEAR REGRESSION MODEL VS.A DUMMY MODEL (BASELINE) THAT USES MEAN AS THE BASIS OF ITS PREDICTION

# Compute mean of values in (y) training set
y_base = np.mean(y_train)

# Replicate the mean values as many times as there are values in the test set
y_pred_base = [y_base] * len(y_test)


# Optional: Show the predicted values of (y) next to the actual values of (y)
df_base_pred = pd.DataFrame({"Actual": y_test, "Predicted": y_pred_base})
print(df_base_pred)

# Compute standard performance metrics of the baseline model:

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test, y_pred_base)
# Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred_base)
# Root Mean Square Error
rmse =  math.sqrt(metrics.mean_squared_error(y_test, y_pred_base))

# Normalised Root Mean Square Error
y_max = y.max()
y_min = y.min()
rmse_norm = rmse / (y_max - y_min)

# R-Squared
r_2 = metrics.r2_score(y_test, y_pred_base)

print("Baseline performance:")
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("RMSE (Normalised): ", rmse_norm)
print("R^2: ", r_2)



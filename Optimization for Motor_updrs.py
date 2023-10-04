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
df = pd.read_csv("Ass3.csv")

df = df.drop(["subject#","age","test_time","sex"], axis=1) # These columns are dropped to make sure that we are only considering the acoustic features
print(df.info())  # we can uncomment this to check if the columns are excluded or not

# # BUILD AND EVALUATE LINEAR REGRESSION USING STATSMODELS

""""
#Separate explanatory variables (x) from the response variable (y) for motor_updrs
x = df.iloc[:,:-2]
y = df.iloc[:,-2]


# Build and evaluate the linear regression model
x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)

"""
"""""


#Optimizing the above linear regression through different optimization process

#1 checking for linearity of data to transform non linear data into linear

fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
fig1.tight_layout()
"""


# uncomment this figures to see the linearity of the data

""" FIGURE 1 
ax1.scatter(x = df['jitter(%)'], y = df['motor_updrs'])
ax1.set_xlabel("jitter(%)")
ax1.set_ylabel("motor_updrs)")

ax2.scatter(x = df['jitter(abs)'], y = df['motor_updrs'])
ax2.set_xlabel("jitter(abs)")
ax2.set_ylabel("motor_updrs")

ax3.scatter(x = df['jitter(rap)'], y = df['motor_updrs'])
ax3.set_xlabel("jitter(rap)")
ax3.set_ylabel("motor_updrs")

ax4.scatter(x = df['jitter(ppq5)'], y = df['motor_updrs'])
ax4.set_xlabel("jitter(ppq5)")
ax4.set_ylabel("motor_updrs")

plt.show()

"""

""" FIGURE 2 
ax1.scatter(x = df['jitter(ddp)'], y = df['motor_updrs'])
ax1.set_xlabel("jitter(ddp)")
ax1.set_ylabel("motor_updrs")

ax2.scatter(x = df['shimmer(%)'], y = df['motor_updrs'])
ax2.set_xlabel("shimmer(%)")
ax2.set_ylabel("motor_updrs")

ax3.scatter(x = df['shimmer(abs)'], y = df['motor_updrs'])
ax3.set_xlabel("shimmer(abs)")
ax3.set_ylabel("motor_updrs")

ax4.scatter(x = df['shimmer(apq3)'], y = df['motor_updrs'])
ax4.set_xlabel("shimmer(apq3)")
ax4.set_ylabel("motor_updrs")

plt.show()

"""

""" FIGURE 3 
ax1.scatter(x = df['shimmer(apq5)'], y = df['motor_updrs'])
ax1.set_xlabel("shimmer(apq5)")
ax1.set_ylabel("motor_updrs")

ax2.scatter(x = df['shimmer(apq11)'], y = df['motor_updrs'])
ax2.set_xlabel("shimmer(apq11)")
ax2.set_ylabel("motor_updrs")

ax3.scatter(x = df['shimmer(dda)'], y = df['motor_updrs'])
ax3.set_xlabel("shimmer(dda)")
ax3.set_ylabel("motor_updrs")

ax4.scatter(x = df['nhr'], y = df['motor_updrs'])
ax4.set_xlabel("nhr")
ax4.set_ylabel("motor_updrs")

plt.show()

"""

""" FIGURE 4 
ax1.scatter(x = df['hnr'], y = df['motor_updrs'])
ax1.set_xlabel("hnr")
ax1.set_ylabel("motor_updrs")

ax2.scatter(x = df['rpde'], y = df['motor_updrs'])
ax2.set_xlabel("rpde")
ax2.set_ylabel("motor_updrs")

ax3.scatter(x = df['dfa'], y = df['motor_updrs'])
ax3.set_xlabel("dfa")
ax3.set_ylabel("motor_updrs")

ax4.scatter(x = df['ppe'], y = df['motor_updrs'])
ax4.set_xlabel("ppe")
ax4.set_ylabel("motor_updrs")

plt.show()

"""

""""
# Re-read dataset into a DataFrame


# Apply non-linear transformation
df["logjitter(%)"] = df["jitter(%)"].apply(np.log)

# Rearrange the variables so that MV appears as the last column

# Visualise the effect of the transformation
plt.figure(figsize=(20,10))

plt.subplot(1,2,1)
plt.scatter(df["jitter(%)"], df["motor_updrs"], color="green")
plt.title("Original LSTAT")
plt.xlabel("jitter(%)")
plt.ylabel("motor_updrs")
plt.plot([0,2],[40,0])

plt.subplot(1,2,2)
plt.scatter(df["logjitter(%)"], df["motor_updrs"], color="red")
plt.title("Log Transformed LSTAT")
plt.xlabel("logjitter(%)")
plt.ylabel("motor_updrs")
plt.plot([0,-10],[40,0])

plt.show()

"""




# 2 Removing collinearity between the independent variables

# Plot correlation matrix
corr = df.corr()

# Plot the pairwise correlation as heatmap
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=False,
    annot=True
)

# customise the labels
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

#plt.show()   #uncomment this to see the collinearity between the variables


#REBUILD AND REEVALUATE LINEAR REGRESSION USING STATSMODELS WITH COLLINEARITY BEING FIXED

# Drop one or more of the correlated variables. Keep only one.
df = df.drop(["jitter(%)", "shimmer(apq3)", "shimmer(abs)","shimmer(%)"], axis=1)
#print(df.info())

# separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,:-2]
y = df.iloc[:,-2]

x = sm.add_constant(x)
model = sm.OLS(y,x).fit()
pred = model.predict(x)
model_details = model.summary()
print(model_details)



""""
#3 Rescaling using standard standarization

# separate explanatory variables (x) from the response variable (y)
x = df.iloc[:,:-2]
y = df.iloc[:,-2]

# APPLY Z-SCORE STANDARDISATION

scaler = StandardScaler()

# Apply z-score standardisation to all explanatory variables
std_x = scaler.fit_transform(x.values)

# Restore the column names of each explanatory variable
std_x_df = pd.DataFrame(std_x, index=x.index, columns=x.columns)



#REBUILD AND REEVALUATE LINEAR REGRESSION USING STATSMODELS USING STANDARDISED EXPLANATORY VARIABLES


# Build and evaluate the linear regression model
std_x_df = sm.add_constant(std_x_df)

print(std_x_df)
model = sm.OLS(y,std_x_df).fit()
pred = model.predict(std_x_df)
model_details = model.summary()
print(model_details)

"""

"""
#4 Gaussian transformatiom APPLY POWER TRANSFORMER

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




#building linear regression with sk learn

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

"""



























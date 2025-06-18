import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
from sklearn.linear_model import LinearRegression

### DATA LOADING
data = pd.read_csv(<DATA LOCATION>)

# Prepare data
# 'target' is the column we want to predict
X = data.drop(columns=['MOF_name', 'fun', 'delta_FE', 'delta_dLM_FE'])  # Features
y = data['delta_dLM_FE']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## Save training and test sets to CSV
#train_data = data.iloc[y_train.index]
#test_data = data.iloc[y_test.index]
#train_data.to_csv('DESIRED LOCATION/train_data.csv', index=False)
#test_data.to_csv('DESIRED LOCATION/test_data.csv', index=False)


# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the scaler on the training data, and then transform the test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the hyperparameters you want to use
params = {
    'colsample_bytree': 0.8,
    'gamma': 1,
    'learning_rate': 0.25,
    'max_depth': 8,
    'min_child_weight': 5, 
    'n_estimators': 220,
    'n_jobs': -1,          # Use all available cores
    'reg_alpha': 10,        # L1 regularization
    'reg_lambda': 10,       # L2 regularization
    'subsample': 0.8,       # Subsample ratio
    'objective':'reg:absoluteerror'
}

# Create and train the XGBoost Regressor model with the specified hyperparameters
model = xgb.XGBRegressor(**params)

# Define the KFold cross-validation strategy
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the scoring function for cross-validation (negative MAE)
scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# Perform k-fold cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring=scorer)

# Print the cross-validation results
print(f"Cross-Validation MAE (Negative): {cv_scores}")
print(f"Mean MAE: {-cv_scores.mean()}")  # Since it's negative, we negate it to get positive MAE
print(f"Standard Deviation of MAE: {cv_scores.std()}")

# Assess the model performance on the test set
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)  # Use scaled X_test
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error on Test Set: {mse}")
print(f"Mean Absolute Error on Test Set: {mae}")

# Save model and scaler
#model.save_model('DESIRED LOCATION/xgb_model.json')
#joblib.dump(scaler, 'DESIRED LOCATION/scaler.pkl')


# Linear Regression between y_test and y_pred
linear_reg = LinearRegression()
linear_reg.fit(y_test.values.reshape(-1, 1), y_pred)  # Fit the linear model

# Get the R² value
r2_score = linear_reg.score(y_test.values.reshape(-1, 1), y_pred)

# Plotting the results
plt.scatter(y_test, y_pred, s=25, edgecolors='black', linewidths=0.9,facecolor='cornflowerblue', alpha=0.5)
plt.plot([-5, 10], [-5, 10], color='grey', linestyle='--') 
plt.ylabel(r'$∆_{LM}F_{FL}$ XGBoost Prediction [kJ/mol]', fontsize=14)
plt.xlabel(r'$∆_{LM}F_{FL}$ Simulation [kJ/mol]', fontsize=14)

plt.plot(y_test, linear_reg.predict(y_test.values.reshape(-1, 1)), color='red', 
                linewidth=1.0, linestyle='--', label=f'Linear Regression (R² = {r2_score:.2f})')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Display the MSE and MAE on the plot
plt.text(-4, 4.0, f'MSE: {mse:.2f} kJ/mol', fontsize=14, color='black')
plt.text(-4, 3.0, f'MAE: {mae:.2f} kJ/mol', fontsize=14, color='black')

# Display the R² value on the plot
plt.text(-4, 2.0, f'R²: {r2_score:.2f}', fontsize=14, color='red')

plt.xlim(-5,5)
plt.ylim(-5,5)
plt.show()

import xgboost as xgb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# 1. Load the trained XGBoost model
xgb_model = xgb.XGBRegressor()
xgb_model.load_model('prediction_model/xgb_model_prediction.json')

# 2. Load the scaler
scaler = joblib.load('prediction_model/scaler_prediction.pkl')

# 3. Load test/interest dataset
data = pd.read_csv('prediction_model/test_data_prediction.csv')
X = data.drop(columns=['MOF_name', 'fun', 'delta_FE', 'delta_dLM_FE'])

# 4. Preprocess the data
X_scaled = scaler.transform(X)

# 5. Make predictions with the loaded XGBoost model
predictions = xgb_model.predict(X_scaled)

# 6. Print predicted values
true_values = data['delta_dLM_FE'] #For comparison if available
comparison = pd.DataFrame({'True Values': true_values, 'Predictions': predictions})
print(comparison)

# 7. Calculate MAE
mae = mean_absolute_error(true_values, predictions)
print(f'Mean Absolute Error (MAE): {mae}')

# 7. Plot to compare True vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(true_values, predictions, alpha=0.7, edgecolors='w', color='cornflowerblue', label='Predictions')
plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'k--', lw=2, c='red')  # Ideal line
plt.xlabel('Simulated $∆∆_{LM}F_{FL}$ [kJ/mol] per MOF atom')
plt.ylabel('XGBoost Predicted  $∆∆_{LM}F_{FL}$ [kJ/mol] per MOF atom')
plt.legend()
plt.show()
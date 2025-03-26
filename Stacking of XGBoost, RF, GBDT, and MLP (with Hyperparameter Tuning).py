#A Data-Driven Machine Learning Model for Radiation-Induced DBTT Shifts in RAFM Steels
#By Pengxin Wang and G. M. A. M. El-Fallah
#Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import optuna

# Load Excel file
data = pd.read_excel("Dataset.xlsx")

# Extract independent and dependent variables
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter optimization functions
def optimize_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }
    model = XGBRegressor(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    return scores.mean()

def optimize_gbdt(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0)
    }
    model = GradientBoostingRegressor(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    return scores.mean()

def optimize_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
    }
    model = RandomForestRegressor(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    return scores.mean()

def optimize_mlp(trial):
    params = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(50,), (100,), (50, 50), (100, 100)]),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'alpha': trial.suggest_float('alpha', 0.0001, 0.1, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 0.001, 0.1, log=True),
        'max_iter': trial.suggest_int('max_iter', 200, 1000)
    }
    model = MLPRegressor(**params, random_state=42)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    return scores.mean()

# Perform hyperparameter tuning
study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(optimize_xgb, n_trials=50)
best_xgb_params = study_xgb.best_params

study_gbdt = optuna.create_study(direction="maximize")
study_gbdt.optimize(optimize_gbdt, n_trials=50)
best_gbdt_params = study_gbdt.best_params

study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(optimize_rf, n_trials=50)
best_rf_params = study_rf.best_params

study_mlp = optuna.create_study(direction="maximize")
study_mlp.optimize(optimize_mlp, n_trials=50)
best_mlp_params = study_mlp.best_params

print("Best Hyperparameters for XGBoost:", best_xgb_params)
print("Best Hyperparameters for GBDT:", best_gbdt_params)
print("Best Hyperparameters for RF:", best_rf_params)
print("Best Hyperparameters for MLP:", best_mlp_params)

# Train Stacking Model with MLP
stacking_model = StackingRegressor(
    estimators=[
        ('xgb', XGBRegressor(**best_xgb_params)),
        ('gbdt', GradientBoostingRegressor(**best_gbdt_params)),
        ('rf', RandomForestRegressor(**best_rf_params)),
        ('mlp', MLPRegressor(**best_mlp_params, random_state=42))
    ],
    final_estimator=Ridge()
)

stacking_model.fit(X_train_scaled, y_train)
y_stack_pred = stacking_model.predict(X_test_scaled)

# Evaluate model
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return [model_name, mse, rmse, mae, r2]

stacking_metrics = evaluate_model(y_test, y_stack_pred, "Stacking Model with MLP")

# Save evaluation metrics
metrics_df = pd.DataFrame([stacking_metrics], columns=['Model', 'MSE', 'RMSE', 'MAE', 'R^2'])
metrics_df.to_excel("metrics.xlsx", index=False)

# Plot scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_stack_pred, label='Stacking Model with MLP', alpha=0.5)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.title("Actual vs Predicted Values")
plt.savefig("scatter plot.png")
plt.show()

# Save scatter plot data
scatter_df = pd.DataFrame({
    'Actual Values': y_test,
    'Stacking Predictions': y_stack_pred
})
scatter_df.to_excel("scatter plot data.xlsx", index=False)

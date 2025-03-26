#A Data-Driven Machine Learning Model for Radiation-Induced DBTT Shifts in RAFM Steels
#By Pengxin Wang and G. M. A. M. El-Fallah
#Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import optuna
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use('TkAgg')

# ========== 1. Load dataset ==========
data = pd.read_excel("Dataset.xlsx")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# ========== 2. Data preprocessing (standardisation) ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========== 3. Hyperparameter optimisation ==========
def optimise_xgb(trial):
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

study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(optimise_xgb, n_trials=50)
best_xgb_params = study_xgb.best_params

# Optimise GBDT, RF, MLP
def optimise_gbdt(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0)
    }
    model = GradientBoostingRegressor(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    return scores.mean()

study_gbdt = optuna.create_study(direction="maximize")
study_gbdt.optimize(optimise_gbdt, n_trials=50)
best_gbdt_params = study_gbdt.best_params

def optimise_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
    }
    model = RandomForestRegressor(**params)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    return scores.mean()

study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(optimise_rf, n_trials=50)
best_rf_params = study_rf.best_params

def optimise_mlp(trial):
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

study_mlp = optuna.create_study(direction="maximize")
study_mlp.optimize(optimise_mlp, n_trials=50)
best_mlp_params = study_mlp.best_params

# ========== 4. Train Stacking Model ==========
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

# ========== 5. Compute Weighted Feature Importance ==========
xgb_model = stacking_model.named_estimators_['xgb']
gbdt_model = stacking_model.named_estimators_['gbdt']
rf_model = stacking_model.named_estimators_['rf']
mlp_model = stacking_model.named_estimators_['mlp']

xgb_importance = xgb_model.feature_importances_
gbdt_importance = gbdt_model.feature_importances_
rf_importance = rf_model.feature_importances_
mlp_importance_result = permutation_importance(mlp_model, X_test_scaled, y_test, scoring='r2', n_repeats=10, random_state=42)
mlp_importance = mlp_importance_result.importances_mean

# Calculate R² contribution of each base model
xgb_r2 = r2_score(y_test, xgb_model.predict(X_test_scaled))
gbdt_r2 = r2_score(y_test, gbdt_model.predict(X_test_scaled))
rf_r2 = r2_score(y_test, rf_model.predict(X_test_scaled))
mlp_r2 = r2_score(y_test, mlp_model.predict(X_test_scaled))

r2_scores = np.array([xgb_r2, gbdt_r2, rf_r2, mlp_r2])
r2_scores = np.clip(r2_scores, 0, None)
weights = r2_scores / r2_scores.sum()

weighted_importance = (
    xgb_importance * weights[0] +
    gbdt_importance * weights[1] +
    rf_importance * weights[2] +
    mlp_importance * weights[3]
)

# Combine and sort feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Weighted Importance': weighted_importance
}).sort_values(by="Weighted Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Weighted Importance'], color='skyblue')
plt.xlabel("Weighted Feature Importance")
plt.ylabel("Features")
plt.title("Stacking Model - Weighted Feature Importance Ranking")
plt.gca().invert_yaxis()
plt.show()

# ========== 6. Save Feature Importance to Excel ==========
feature_importance_df.to_excel("weighted_feature_importance.xlsx", index=False)

# ========== 7. Save Model Weights to Excel ==========
weights_df = pd.DataFrame({
    'Model': ['XGB', 'GBDT', 'RF', 'MLP'],
    'R² Score': r2_scores,
    'Weight': weights
})
weights_df.to_excel("model_weights.xlsx", index=False)

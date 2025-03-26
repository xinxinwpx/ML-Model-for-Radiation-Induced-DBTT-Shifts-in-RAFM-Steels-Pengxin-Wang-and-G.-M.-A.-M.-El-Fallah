#A Data-Driven Machine Learning Model for Radiation-Induced DBTT Shifts in RAFM Steels
#By Pengxin Wang and G. M. A. M. El-Fallah
#Contact Dr Gebril El-Fallah:  gmae2@leicester.ac.uk

import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# Use non-GUI mode to avoid TkAgg issues
import matplotlib
matplotlib.use('Agg')

# ===========================
# 1️⃣  Load the dataset
# ===========================
data = pd.read_excel(r"Dataset.xlsx", engine="openpyxl")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardise the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inverse transform function
def inverse_transform(scaled_data, scaler):
    return scaler.inverse_transform(scaled_data)

# ===========================
# 2️⃣  Train the Stacking model
# ===========================
best_xgb_params = {'n_estimators': 779, 'learning_rate': 0.1274, 'max_depth': 11,
                   'subsample': 0.8705, 'colsample_bytree': 0.9729}
best_gbdt_params = {'n_estimators': 1623, 'learning_rate': 0.1243, 'max_depth': 5,
                    'subsample': 0.7979}
best_rf_params = {'n_estimators': 1880, 'max_depth': 12, 'min_samples_split': 2, 'min_samples_leaf': 1}
best_mlp_params = {'hidden_layer_sizes': (100, 100), 'activation': 'relu', 'alpha': 0.000466,
                   'learning_rate_init': 0.0367, 'max_iter': 899}

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

# ===========================
# 3️⃣  Compute SHAP interactions
# ===========================
explainer_xgb = shap.TreeExplainer(stacking_model.named_estimators_['xgb'])
explainer_gbdt = shap.TreeExplainer(stacking_model.named_estimators_['gbdt'])
explainer_rf = shap.TreeExplainer(stacking_model.named_estimators_['rf'])

shap_values_xgb = explainer_xgb.shap_values(X_train_scaled)
shap_values_gbdt = explainer_gbdt.shap_values(X_train_scaled)
shap_values_rf = explainer_rf.shap_values(X_train_scaled)

shap_interaction_values_xgb = explainer_xgb.shap_interaction_values(X_train_scaled)
shap_interaction_values_gbdt = explainer_gbdt.shap_interaction_values(X_train_scaled)
shap_interaction_values_rf = explainer_rf.shap_interaction_values(X_train_scaled)

explainer_mlp = shap.KernelExplainer(stacking_model.named_estimators_['mlp'].predict, X_train_scaled)
shap_values_mlp = explainer_mlp.shap_values(X_train_scaled, nsamples=100)

shap_values_mlp_3d = np.expand_dims(shap_values_mlp, axis=2)
shap_values_mlp_3d = np.repeat(shap_values_mlp_3d, repeats=X_train.shape[1], axis=2)

shap_interaction_values_stacking = (
    shap_interaction_values_xgb +
    shap_interaction_values_gbdt +
    shap_interaction_values_rf +
    shap_values_mlp_3d
) / 4

# ===========================
# 4️⃣  Visualise SHAP interactions (after inverse transform, save plots)
# ===========================
X_train_original = inverse_transform(X_train_scaled, scaler)

# Interaction matrix (heatmap)
interaction_strength = np.abs(shap_interaction_values_stacking).mean(axis=0)
plt.figure(figsize=(10, 8))
plt.imshow(interaction_strength, cmap="coolwarm", interpolation="nearest")
plt.colorbar()
plt.xticks(range(len(X_train.columns)), X_train.columns, rotation=90)
plt.yticks(range(len(X_train.columns)), X_train.columns)
plt.title("SHAP Interaction Matrix - Stacking Model (Including MLP)")
plt.savefig("shap_interaction_matrix.png")
plt.close()

# Set font and style
matplotlib.rc("font", family="Times New Roman", weight="bold")
matplotlib.rc("axes", labelweight="bold", linewidth=2, titlesize=14)
matplotlib.rc("xtick", labelsize=12)
matplotlib.rc("ytick", labelsize=12)
matplotlib.rc("lines", linewidth=2)

# Select two features for interaction analysis
def format_colorbar(cb, feature_y):
    cb.ax.yaxis.label.set_fontsize(18)
    cb.ax.yaxis.label.set_weight("bold")
    cb.ax.yaxis.label.set_family("Times New Roman")
    cb.ax.set_ylabel(feature_y, fontsize=22, fontweight="bold", fontfamily="Times New Roman")
    cb.ax.tick_params(labelsize=18, width=2)

feature_x, feature_y = "Cr", "W"  # Replace with desired features
feature_x_idx, feature_y_idx = X_train.columns.get_loc(feature_x), X_train.columns.get_loc(feature_y)

plt.figure(figsize=(8, 6))
scatter = shap.dependence_plot(
    (feature_x, feature_y),
    shap_interaction_values_stacking,
    X_train_original,
    feature_names=X_train.columns,
    show=False
)
ax = plt.gca()
ax.set_xlabel(feature_x, fontsize=20, fontweight='bold', fontfamily='Times New Roman')
ax.set_ylabel("SHAP Value", fontsize=20, fontweight='bold', fontfamily='Times New Roman')
ax.set_title(f"SHAP Dependence Plot: {feature_x} vs {feature_y}", fontsize=20, fontweight='bold', fontfamily='Times New Roman')
for spine in ax.spines.values():
    spine.set_linewidth(2)
plt.xticks(fontsize=20, fontweight='bold', fontfamily='Times New Roman')
plt.yticks(fontsize=20, fontweight='bold', fontfamily='Times New Roman')
if isinstance(scatter, tuple) and len(scatter) > 1 and hasattr(scatter[1], "get_array"):
    cb = plt.colorbar(scatter[1])
    format_colorbar(cb, feature_y)
plt.savefig("shap_dependence_plot_Cr_W.png", dpi=300, bbox_inches='tight')
plt.close()

feature_x, feature_y = "Cr", "Ta"  # Replace with desired features
feature_x_idx, feature_y_idx = X_train.columns.get_loc(feature_x), X_train.columns.get_loc(feature_y)

plt.figure(figsize=(8, 6))
scatter = shap.dependence_plot(
    (feature_x, feature_y),
    shap_interaction_values_stacking,
    X_train_original,
    feature_names=X_train.columns,
    show=False
)
ax = plt.gca()
ax.set_xlabel(feature_x, fontsize=20, fontweight='bold', fontfamily='Times New Roman')
ax.set_ylabel("SHAP Value", fontsize=20, fontweight='bold', fontfamily='Times New Roman')
ax.set_title(f"SHAP Dependence Plot: {feature_x} vs {feature_y}", fontsize=20, fontweight='bold', fontfamily='Times New Roman')
for spine in ax.spines.values():
    spine.set_linewidth(2)
plt.xticks(fontsize=20, fontweight='bold', fontfamily='Times New Roman')
plt.yticks(fontsize=20, fontweight='bold', fontfamily='Times New Roman')
if isinstance(scatter, tuple) and len(scatter) > 1 and hasattr(scatter[1], "get_array"):
    cb = plt.colorbar(scatter[1])
    format_colorbar(cb, feature_y)
plt.savefig("shap_dependence_plot_Cr_Ta.png", dpi=300, bbox_inches='tight')
plt.close()

feature_x, feature_y = "Ta", "W"  # Replace with desired features
feature_x_idx, feature_y_idx = X_train.columns.get_loc(feature_x), X_train.columns.get_loc(feature_y)

plt.figure(figsize=(8, 6))
scatter = shap.dependence_plot(
    (feature_x, feature_y),
    shap_interaction_values_stacking,
    X_train_original,
    feature_names=X_train.columns,
    show=False
)
ax = plt.gca()
ax.set_xlabel(feature_x, fontsize=20, fontweight='bold', fontfamily='Times New Roman')
ax.set_ylabel("SHAP Value", fontsize=20, fontweight='bold', fontfamily='Times New Roman')
ax.set_title(f"SHAP Dependence Plot: {feature_x} vs {feature_y}", fontsize=20, fontweight='bold', fontfamily='Times New Roman')
for spine in ax.spines.values():
    spine.set_linewidth(2)
plt.xticks(fontsize=20, fontweight='bold', fontfamily='Times New Roman')
plt.yticks(fontsize=20, fontweight='bold', fontfamily='Times New Roman')
if isinstance(scatter, tuple) and len(scatter) > 1 and hasattr(scatter[1], "get_array"):
    cb = plt.colorbar(scatter[1])
    format_colorbar(cb, feature_y)
plt.savefig("shap_dependence_plot_Ta_W.png", dpi=300, bbox_inches='tight')
plt.close()

# SHAP interaction summary plot
shap.summary_plot(
    shap_interaction_values_stacking, X_train_original, feature_names=X_train.columns, show=False
)
plt.savefig("shap_summary_plot.png")
plt.close()

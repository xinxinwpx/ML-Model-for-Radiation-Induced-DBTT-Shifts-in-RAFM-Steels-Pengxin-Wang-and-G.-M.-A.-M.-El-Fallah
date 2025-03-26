# A Data-Driven Machine Learning Model for Radiation-Induced DBTT Shifts in RAFM Steels

## Authors

Pengxin Wang, G. M. A. M. El-Fallah

## Overview

This repository contains the complete machine learning workflow for predicting radiation-induced ductile-to-brittle transition temperature (DBTT) shifts in Reduced Activation Ferritic-Martensitic (RAFM) steels. A stacking ensemble model was built by integrating XGBoost, Random Forest (RF), Gradient Boosted Decision Trees (GBDT), and Multi-Layer Perceptron (MLP). The model leverages alloy composition and irradiation conditions to estimate DBTT variations, contributing to the understanding of radiation effects in structural materials.

## Contents

- 📊 **Dataset**  
  The processed Excel dataset containing alloy compositions, irradiation conditions, and DBTT values.

- 🧠 **Independent training of XGBoost, RF, GBDT, and MLP**  
  Scripts for individual training of the four base models, including hyperparameter optimisation using Optuna.

- 🔀 **Stacking of XGBoost, RF, GBDT, and MLP (with tuning)**  
  Code for constructing and training the stacking ensemble with Ridge as the meta-learner.

- 📈 **Feature Importance Analysis of the Stacking Model**  
  Weighted feature importance analysis based on each model's R² contribution, including permutation importance for MLP.

- 🧩 **SHAP Analysis of the Stacking Model**  
  SHAP-based visualisation and interaction analysis of feature contributions, including SHAP interaction matrix and dependence plots.

## Usage

1. **Clone the repository**

```bash
git clone https://github.com/your-repo-url/stacking-dbtt-rafa.git
```

2. **Install the dependencies**

```bash
pip install -r requirements.txt
```

3. **Run model training or feature analysis**

Example:

```bash
python stacking_model_training.py
```

## Contact

For questions or collaboration, please contact:

Dr. Gebril El-Fallah  
📧 gmae2@leicester.ac.uk

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

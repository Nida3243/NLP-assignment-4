# Step 8: Explainable AI (XAI) - SHAP for Model Interpretation

# Install SHAP if not installed
!pip install shap

import shap
import matplotlib.pyplot as plt

# Step 8.1: Create a SHAP Explainer
# Use the XGBoost model (better SHAP compatibility)
explainer = shap.Explainer(voting_clf.named_estimators_['xgb'], X_train)

# Step 8.2: Compute SHAP values for the test set
shap_values = explainer(X_test)

# Step 8.3: Plot Feature Importance - Bar Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values, 
    X_test, 
    plot_type="bar", 
    plot_size=(10, 6), 
    max_display=10,   # Top 10 features
    color='coolwarm'
)
plt.title('Feature Importance (Bar Plot)', fontsize=16)
plt.show()

# Step 8.4: Detailed Summary Plot (Beeswarm/Dot Plot)
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values, 
    X_test, 
    plot_size=(10, 6),
    color_bar=True,
    cmap='coolwarm'
)
plt.title('SHAP Summary Plot (Beeswarm)', fontsize=16)
plt.show()

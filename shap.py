import shap
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Create SHAP explainer (using the trained XGBoost model)
explainer = shap.Explainer(voting_clf.named_estimators_['xgb'], X_train)

# Step 2: Compute SHAP values for the test set
shap_values = explainer(X_test)

# Step 3: Plot Heatmap for SHAP values
shap_values_array = shap_values.values  # Convert SHAP values to a numpy array
sns.heatmap(shap_values_array, cmap="coolwarm", xticklabels=X_test.columns, yticklabels=None)
plt.xlabel('Features')
plt.ylabel('Samples')
plt.title('SHAP Value Heatmap')
plt.show()



..........................
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







#new
# Step 8: Explainable AI (XAI) - SHAP for Model Interpretation

# Install SHAP if not already installed
!pip install shap

import shap
import matplotlib.pyplot as plt

# Step 8.1: Create a SHAP explainer
explainer = shap.Explainer(voting_clf.named_estimators_['xgb'], X_train)

# Step 8.2: Compute SHAP values for the test set
shap_values = explainer(X_test)

# Step 8.3: Bar Plot - Feature Importance
print("🔵 Plotting Bar Plot for Global Feature Importance...")
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("Feature Importance (Bar Plot)")
plt.tight_layout()
plt.savefig('shap_bar_plot.png', dpi=300)
plt.show()

# Step 8.4: Beeswarm Plot - SHAP Value Distribution
print("🔵 Plotting Beeswarm Plot for Detailed Feature Effects...")
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)
plt.title("Feature Effects (Beeswarm Plot)")
plt.tight_layout()
plt.savefig('shap_beeswarm_plot.png', dpi=300)
plt.show()

# Step 8.5: Waterfall Plot - Single Prediction
print("🔵 Plotting Waterfall Plot for a Single Instance...")
sample_idx = 0  # Change index if you want another sample
waterfall_exp = shap.Explanation(values=shap_values.values[sample_idx],
                                 base_values=shap_values.base_values[sample_idx],
                                 data=X_test.iloc[sample_idx])

plt.figure()
shap.waterfall_plot(waterfall_exp, show=False)
plt.title(f"Waterfall Plot (Sample {sample_idx})")
plt.tight_layout()
plt.savefig(f'shap_waterfall_plot_sample{sample_idx}.png', dpi=300)
plt.show()

# Step 8.6: Force Plot - Local Explanation
print("🔵 Plotting Force Plot for a Single Instance...")
force_plot = shap.force_plot(shap_values.base_values[sample_idx],
                             shap_values.values[sample_idx],
                             X_test.iloc[sample_idx],
                             matplotlib=True)
plt.title(f"Force Plot (Sample {sample_idx})")
plt.tight_layout()
plt.savefig(f'shap_force_plot_sample{sample_idx}.png', dpi=300)
plt.show()

# Step 8.7: Dependence Plot - Feature Interaction
print("🔵 Plotting Dependence Plot for 'hear_rate' Feature...")
plt.figure()
shap.dependence_plot("hear_rate", shap_values.values, X_test, show=False)
plt.title("Dependence Plot for 'hear_rate'")
plt.tight_layout()
plt.savefig('shap_dependence_plot_heartrate.png', dpi=300)
plt.show()

# (Optional) You can repeat Dependence Plot for any feature like 'steps', 'distance', etc.


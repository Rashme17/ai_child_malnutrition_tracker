import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures  # Optional: for non-linear terms
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')  # Optional: Suppress convergence warnings if any

# Check NumPy version for compatibility (updated to allow 1.26+ or 2.x for broader support)
if np.__version__ < '1.26' or np.__version__ >= '3.0':
    raise RuntimeError(f"NumPy version {np.__version__} is not compatible. Please install numpy>=1.26,<3.0.")

# Load dataset
csv_path = "malnutrition_cases.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found. Run generate_data.py first to create it (with age in years).")

try:
    df = pd.read_csv(csv_path)
except Exception as e:
    raise ValueError(f"Error loading CSV: {e}")

# Validate columns and data
required_cols = ["age", "weight", "symptom_count", "severity_score", "outcome_days"]
if not all(col in df.columns for col in required_cols):
    raise ValueError(f"CSV must have columns: {required_cols}")

# Basic data cleaning (handle any NaNs or outliers if present)
df = df.dropna()
df = df[(df['age'] > 0) & (df['weight'] > 0) & (df['outcome_days'] > 0)]  # Remove invalid rows
if len(df) < 10:
    raise ValueError("Insufficient valid data after cleaning.")

print(f"Loaded {len(df)} rows from '{csv_path}' (age in years).")
print(f"Data shape: {df.shape}")
print(f"Age stats (years):\n{df['age'].describe()}")
print(f"Other features stats:\n{df[['weight', 'symptom_count', 'severity_score']].describe()}")
print(f"Target (outcome_days) stats:\n{df['outcome_days'].describe()}")

# Optional: Check for multicollinearity (high correlations between features)
corr_matrix = df[required_cols[:-1]].corr()  # Exclude target
print(f"\nFeature Correlation Matrix:\n{corr_matrix}")
if corr_matrix.abs().max().max() > 0.8:
    print("Warning: High multicollinearity detected—consider feature selection.")

# Prepare features and target
X = df[["age", "weight", "symptom_count", "severity_score"]]
y = df["outcome_days"]

# Optional: Add polynomial features for non-linearity (uncomment if needed)
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X = poly.fit_transform(X)
# print(f"Added polynomial features: New shape {X.shape}")

# Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()  # Simple linear regression
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "regression_model.pkl")
print("Model saved as 'regression_model.pkl' (Linear Regression, trained on age in years).")

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(X_test, y_test)  # R² score

print(f"\nModel Evaluation on Test Set:")
print(f"MAE: {mae:.2f} days")
print(f"RMSE: {rmse:.2f} days")
print(f"R² Score: {r2:.4f}")

# Coefficients (interpretation: impact of each feature on outcome_days)
if 'poly' not in locals():  # If no polynomial features
    coef_df = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_,
        'abs_impact': np.abs(model.coef_)  # Absolute value for ranking importance
    }).sort_values('abs_impact', ascending=False)
else:
    # For polynomial features, this is more complex—skip or use feature_names_
    coef_df = pd.DataFrame({'Note': ['Use feature_names_ for poly coeffs']})
print(f"\nFeature Coefficients (Linear Impact on outcome_days):\n{coef_df}")
print(f"Intercept: {model.intercept_:.2f} days")

# Optional: Save predictions for analysis
test_results = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred,
    'error': np.abs(y_test - y_pred)
})
test_results.to_csv('test_predictions.csv', index=False)
print("\nTest predictions saved to 'test_predictions.csv'.")
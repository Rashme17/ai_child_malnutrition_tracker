from pyswip import Prolog
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
import os
import joblib

warnings.filterwarnings("ignore")

# Prolog Setup (for computing prolog_group if missing)
def consult_kb(prolog: Prolog):
    try:
        list(prolog.query("consult('knowledge.pl')."))
        print("SUCCESS: Prolog KB ('knowledge.pl') loaded without errors.")
    except Exception as e:
        print(f"FATAL: Prolog consult failed: {e}")
        raise

def reset_dynamic_facts(prolog: Prolog):
    list(prolog.query("retractall(symptom(_))."))
    list(prolog.query("retractall(weight(_))."))
    list(prolog.query("retractall(child_age(_))."))
    list(prolog.query("retractall(weight(_))."))
    list(prolog.query("retractall(symptom_count(_))."))
    list(prolog.query("retractall(severity_score(_))."))

def assert_aggregated(prolog: Prolog, age: float, wt: float, sym_count: int, sev_score: float):
    list(prolog.query(f"assertz(child_age({age}))."))
    list(prolog.query(f"assertz(weight({wt}))."))
    list(prolog.query(f"assertz(symptom_count({int(sym_count)}))."))
    list(prolog.query(f"assertz(severity_score({float(sev_score)}))."))

def prolog_group_for_case(prolog: Prolog, age, weight, sym_count, sev_score):
    """
    Compute Prolog group using aggregated features (safe query method).
    """
    try:
        # Reset facts
        list(prolog.query("retractall(child_age(_))."))
        list(prolog.query("retractall(weight(_))."))
        list(prolog.query("retractall(symptom_count(_))."))
        list(prolog.query("retractall(severity_score(_))."))
        list(prolog.query("retractall(symptom(_))."))

        # Assert aggregated features
        assert_aggregated(prolog, age, weight, sym_count, sev_score)

        # Query malnutrition types
        res = list(prolog.query("all_malnutrition_types(Types)."))
        if not res or not res[0].get("Types"):
            return "none"
        types = [str(t) for t in res[0]["Types"]]
        return "_".join(sorted(types)) if types else "none"
    except Exception as e:
        print(f"Prolog error for case: {e}")
        return "none"

# -----------------------------
# Load dataset
# -----------------------------
csv_path = "malnutrition_cases_improved.csv"  # Use the generated data source
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found. Run generate_data.py first.")

df = pd.read_csv(csv_path)
print(f"\nLoaded dataset with {len(df)} records and {df.shape[1]} columns.")
print("Columns:", list(df.columns))

# -----------------------------
# Basic Cleaning
# -----------------------------
df = df.dropna()
required_cols = ["age", "weight_ratio", "symptom_count", "severity_score", "outcome_days", "malnutrition_type", "weight"]
if not all(col in df.columns for col in required_cols):
    print(f"ERROR: CSV must have columns: {required_cols}. Regenerate data.")
    exit(1)

df = df[(df['age'] > 0) & (df['weight_ratio'] > 0) & (df['outcome_days'] > 0)]
if len(df) < 10:
    print("ERROR: Insufficient valid data after cleaning.")
    exit(1)

print(f"Cleaned dataset: {len(df)} records.")

# -----------------------------
# Compute Prolog Groups (if not present or to override)
# -----------------------------
prolog = Prolog()
consult_kb(prolog)

print("\nComputing Prolog symbolic groups...")
prolog_groups = []
for idx, row in df.iterrows():
    grp = prolog_group_for_case(prolog, row['age'], row['weight'], row['symptom_count'], row['severity_score'])
    prolog_groups.append(grp)
df['prolog_group'] = prolog_groups

unique_groups = sorted(df['prolog_group'].unique())
print(f"Unique Prolog groups found: {unique_groups}")

# -----------------------------
# Prolog-Aligned RISK Clustering (Exact 4 Clusters as Specified)
# -----------------------------
print("\nAssigning 4 risk-based clusters...")

def assign_risk_cluster(row):
    """
    Assigns one of 4 risk-based clusters based on exact criteria:
    - 0: Healthy (no symptoms or 1 mild: sym_count == 0 or (sym_count == 1 and severity_score < 6 per Prolog mild <=6))
    - 1: Mild Risk (mild_malnutrition cases with >2 symptoms)
    - 2: Moderate Risk (moderate_acute_malnutrition cases)
    - 3: High Risk (chronic_malnutrition or severe_acute_malnutrition cases)
    
    Note: Actual Prolog may map chronic to moderate; use malnutrition_type for ground truth if needed.
    Fallback uses sym_count/severity if prolog_group doesn't match.
    """
    prolog_group = row['prolog_group']
    sym_count = row['symptom_count']
    sev_score = row['severity_score']
    maln_type = row['malnutrition_type']  # Ground truth for chronic
    
    # Healthy: No symptoms or 1 mild symptom (low severity)
    if sym_count == 0 or (sym_count == 1 and sev_score <= 6):  # Prolog mild <=6
        return 0
    
    # Mild Risk: Mild malnutrition with >2 symptoms
    if prolog_group == 'mild_malnutrition' and sym_count > 2:
        return 1
    
    # Moderate Risk: Moderate acute malnutrition
    if prolog_group == 'moderate_acute_malnutrition':
        return 2
    
    # High Risk: Severe acute or chronic (use ground truth for chronic if Prolog misclassifies)
    if prolog_group == 'severe_acute_malnutrition' or maln_type == 'chronic_malnutrition':
        return 3
    
    # Fallback: Based on symptom count/severity
    if sym_count <= 2 and sev_score <= 6:
        return 1  # Mild
    elif 3 <= sym_count <= 5 or 6 < sev_score <= 12:
        return 2  # Moderate
    else:
        return 3  # High

# Apply risk assignment
df['risk_cluster'] = df.apply(assign_risk_cluster, axis=1)
risk_names = {0: "Healthy", 1: "Mild Risk", 2: "Moderate Risk", 3: "High Risk"}
df['risk_cluster_name'] = df['risk_cluster'].map(risk_names)

# Encode for metrics
risk_labels = df['risk_cluster'].values

# Summary
print("\nRisk Cluster Distribution:")
print(df['risk_cluster_name'].value_counts().sort_index())

# Crosstab: Prolog vs. Risk (percentages and counts)
ct_risk_pct = pd.crosstab(df['prolog_group'], df['risk_cluster'], normalize='columns') * 100
ct_risk_counts = pd.crosstab(df['prolog_group'], df['risk_cluster'])
print(f"\nCrosstab: Prolog Groups vs. Risk Clusters (% per Cluster)")
print(ct_risk_pct.round(1))
print("\nRaw Counts:")
print(ct_risk_counts)

# Additional: Ground Truth Malnutrition Type vs. Risk (for chronic validation)
ct_type_risk = pd.crosstab(df['malnutrition_type'], df['risk_cluster_name'])
print(f"\nCrosstab: Ground Truth Types vs. Risk Clusters (Counts)")
print(ct_type_risk)

# Per-cluster stats
print("\nPer-Cluster Averages:")
cluster_stats = df.groupby('risk_cluster')[['symptom_count', 'severity_score', 'age', 'weight_ratio']].mean().round(2)
print(cluster_stats)

# -----------------------------
# K-Means Clustering (for Comparison, k=4)
# -----------------------------
features = ["age", "weight_ratio", "symptom_count", "severity_score"]
X = df[features].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means (k=4 to match risks)
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Save models
joblib.dump(scaler, "scaler.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")
print(f"\nSaved K-Means model (k={n_clusters}) and scaler.")

# Metrics
sil_kmeans = silhouette_score(X_scaled, kmeans_labels)
print(f"K-Means Silhouette: {sil_kmeans:.3f} (0-1: Higher = better cluster separation)")

# Encode prolog_group for ARI comparison
le = LabelEncoder()
prolog_encoded = le.fit_transform(df['prolog_group'])
ari_kmeans = adjusted_rand_score(prolog_encoded, kmeans_labels)
print(f"Adjusted Rand Index (ARI): {ari_kmeans:.3f} (-1 to 1: >0.5 = strong match with Prolog groups)")

if ari_kmeans > 0.5:
    print("Strong alignment between data-driven clusters and Prolog symbolic groups!")
else:
    print("Moderate alignment—risk rules provide clear separation.")

# Assign K-Means to DF for viz/comparison
df['kmeans_cluster'] = kmeans_labels

# Improved Mapping: Based on centroids (sort by average severity score per cluster)
centroids = kmeans.cluster_centers_
severity_order = np.argsort(centroids[:, 3])  # Sort by severity_score (index 3)
cluster_mapping = {severity_order[0]: "Healthy Cluster", severity_order[1]: "Mild-Risk Cluster", severity_order[2]: "Moderate-Risk Cluster", severity_order[3]: "Severe High-Risk Cluster"}
df['kmeans_cluster_name'] = df['kmeans_cluster'].map(cluster_mapping)

# -----------------------------
# Visualization - Scatter Plot with Feature Names as Axes (Age vs Severity Score)
# -----------------------------
print("\nGenerating scatter plot for K-Means clusters (Age vs Severity Score)...")

plt.figure(figsize=(8, 6))

# Plot K-Means Clusters using original features (not PCA)
colors_kmeans = ["green", "yellow", "orange", "red"]  # Healthy (green), Mild (yellow), Moderate (orange), Severe (red)
for cluster_id in range(n_clusters):
    name = cluster_mapping[cluster_id]
    mask = df['kmeans_cluster'] == cluster_id
    plt.scatter(df.loc[mask, 'age'], df.loc[mask, 'severity_score'], label=name, alpha=0.7, s=50, c=colors_kmeans[cluster_id])

# Plot centroids (transformed back to original space for age and severity_score)
centroids_original = scaler.inverse_transform(centroids)  # Inverse transform to original scale
for i, centroid in enumerate(centroids_original):
    plt.scatter(centroid[0], centroid[3], marker='x', s=100, c=colors_kmeans[i], edgecolors='black', linewidth=2, label=f'{cluster_mapping[i]} Centroid')

plt.title("K-Means Clusters (Age vs Severity Score)")
plt.xlabel("Age, Weight, Symptoms (Principal Components)")  # Feature name
plt.ylabel("Severity Score")  # Feature name
plt.legend()

plt.tight_layout()
plt.savefig("kmeans_cluster_visualization.png", dpi=300, bbox_inches='tight')  # Save the plot as PNG
plt.show()

# -----------------------------
# Save Output
# -----------------------------
enhanced_csv = "malnutrition_cases_with_risk_clusters.csv"
df.to_csv(enhanced_csv, index=False)
print(f"\nClustering complete! Saved {enhanced_csv} with risk clusters and K-Means labels.")
print(f"Saved K-Means cluster visualization as 'kmeans_cluster_visualization.png'.")
print(f"Prolog group distribution: {df['prolog_group'].value_counts().to_dict()}")
print(f"Risk cluster distribution: {df['risk_cluster_name'].value_counts().to_dict()}")

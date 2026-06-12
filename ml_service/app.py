from flask import Flask, request, jsonify
import os, joblib, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

app = Flask(__name__)

# Load artifacts if present (service will still run without them)
regression_model = joblib.load("regression_model.pkl") if os.path.exists("regression_model.pkl") else None
scaler          = joblib.load("scaler.pkl")           if os.path.exists("scaler.pkl")           else None
kmeans          = joblib.load("kmeans_model.pkl")     if os.path.exists("kmeans_model.pkl")     else None
model           = joblib.load("malnutrition_model.pkl") if os.path.exists("malnutrition_model.pkl") else None
mlb             = joblib.load("mlb.pkl")              if os.path.exists("mlb.pkl")              else None
le              = joblib.load("le.pkl")               if os.path.exists("le.pkl")               else None

@app.route("/predict_regression", methods=["POST"])
def predict_regression():
    data = request.get_json(silent=True) or {}
    feats = np.array([data.get("features", [0,0,0,0])], dtype=float)
    if regression_model is None:
        return jsonify({"error": "regression_model not loaded"}), 500
    pred = float(regression_model.predict(feats)[0])
    return jsonify({"prediction": pred})

@app.route("/predict_cluster", methods=["POST"])
def predict_cluster():
    data = request.get_json(silent=True) or {}
    feats = np.array([data.get("features", [0,1,0,0])], dtype=float)  # [age, weight_ratio, sym_count, sev_score]
    if (scaler is None) or (kmeans is None):
        return jsonify({"error": "clustering artifacts not loaded"}), 500
    Xs = scaler.transform(feats)
    label = int(kmeans.predict(Xs)[0])
    # IMPORTANT: main app expects 'kmeans_cluster'
    return jsonify({"kmeans_cluster": label})

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json(silent=True) or {}
    if any(x is None for x in (model, mlb, le)):
        return jsonify({"error": "classification artifacts not loaded"}), 500
    age = float(data.get("age", 0))
    wt  = float(data.get("weight", 0))
    syms = [s for s in data.get("symptoms", []) if s in mlb.classes_]
    sym_bin = mlb.transform([syms])[0]
    X = np.hstack([[age, wt], sym_bin]).reshape(1, -1)
    pred_idx = int(model.predict(X)[0])
    label = str(le.inverse_transform([pred_idx])[0])
    return jsonify({"prediction": label})

@app.route("/compare", methods=["POST"])
def compare():
    data = request.get_json(silent=True) or {}
    p = data.get("prolog_outcomes", [])
    m = data.get("ml_predictions", [])
    if len(p) != len(m) or len(p) == 0:
        return jsonify({"error": "Mismatched or empty arrays"}), 400
    p = np.array(p, dtype=float)
    m = np.array(m, dtype=float)
    mae = float(mean_absolute_error(p, m))
    rmse = float(np.sqrt(mean_squared_error(p, m)))
    return jsonify({"mae": mae, "rmse": rmse})

@app.route("/cluster", methods=["GET"])
def cluster():
    # Pure-ML clustering (no Prolog dependency here)
    csv_path = "malnutrition_cases_improved.csv"
    if not os.path.exists(csv_path):
        return jsonify({"error": f"{csv_path} not found"}), 404
    df = pd.read_csv(csv_path).dropna()
    req = ["age","weight_ratio","symptom_count","severity_score"]
    if not all(c in df.columns for c in req):
        return jsonify({"error": f"CSV must contain {req}"}), 400
    X = df[req].values
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    km = KMeans(n_clusters=5, n_init=10, random_state=42).fit(Xs)
    joblib.dump(sc, "scaler.pkl")
    joblib.dump(km, "kmeans_model.pkl")
    sil = float(silhouette_score(Xs, km.labels_))
    return jsonify({
        "sil_kmeans": round(sil, 3),
        "kmeans_totals": pd.Series(km.labels_).value_counts().sort_index().tolist()
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)

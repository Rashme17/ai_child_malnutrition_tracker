import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib
import ast
import os
import numpy as np
from collections import Counter
from pac_vc import empirical_vc_shatter, pac_sample_size_vc, pac_sample_size_finite, approximate_log2_hypotheses_from_features


CSV_FILE = "case_data.csv"
MODEL_FILE = "malnutrition_model.pkl"
MLB_FILE = "mlb.pkl"
LE_FILE = "le.pkl"

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    # Expect columns: age, weight, symptoms, label
    # Normalize symptoms to Python list
    def parse_sym(x):
        if pd.isna(x):
            return []
        if isinstance(x, str):
            x = x.strip()
            if x.startswith('[') and x.endswith(']'):
                try:
                    return [s.strip().replace("'", "").replace('"','') for s in ast.literal_eval(x)]
                except Exception:
                    pass
            if '|' in x:
                return [s.strip() for s in x.split('|') if s.strip()]
            if ',' in x:
                return [s.strip() for s in x.split(',') if s.strip()]
            return [x]
        return list(x)
    df['symptoms_list'] = df['symptoms'].apply(parse_sym)
    return df[['age','weight','symptoms_list','label']]

def build_features(df):
    mlb = MultiLabelBinarizer()
    X_sym = mlb.fit_transform(df['symptoms_list'])
    X_num = df[['age','weight']].fillna(0).values
    X = np.hstack([X_num, X_sym])
    le = LabelEncoder()
    y = le.fit_transform(df['label'].astype(str))
    return X, y, mlb, le

def main():
    if not os.path.exists(CSV_FILE):
        print(f"{CSV_FILE} not found. Please export cases from Prolog or create the CSV with columns: age,weight,symptoms,label")
        return

    df = load_dataset(CSV_FILE)
    X, y, mlb, le = build_features(df)

    # Print class distribution
    print("\n📊 Class distribution:")
    counts = Counter(df['label'])
    for cls, cnt in counts.items():
        print(f"  {cls}: {cnt} samples")

    # Safe split with stratification
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        print("⚠️ Dataset too small for stratified split → using full dataset for training.")
        X_train, X_test, y_train, y_test = X, X, y, y

    # Train Decision Tree with regularization to combat overfitting
    model = DecisionTreeClassifier(
        criterion='entropy',
        random_state=42,
        max_depth=3,  # Shallower to reduce complexity
        min_samples_split=10,  # Avoid splits on small groups
        min_samples_leaf=5,  # Ensure leaves have enough samples
        class_weight='balanced',  # Handle slight imbalance
        ccp_alpha=0.05  # Pruning: penalize complex trees (tune 0.001-0.1)
    )
    model.fit(X_train, y_train)

    # Optional: Hyperparameter tuning with GridSearchCV (uncomment for auto-optimization)
    # param_grid = {
    #     'max_depth': [2, 3, 4],
    #     'min_samples_leaf': [5, 10],
    #     'ccp_alpha': [0.001, 0.01, 0.1]
    # }
    # grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    # grid.fit(X_train, y_train)
    # model = grid.best_estimator_
    # print(f"Best params: {grid.best_params_}")

    # Evaluate on train and test
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print("\n✅ Model trained successfully")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    if train_acc - test_acc > 0.1:
        print("⚠️ Potential overfitting detected (train-test gap >10%). Try increasing ccp_alpha or reducing max_depth.")
    elif test_acc < 0.8:
        print("⚠️ Underfitting? Try increasing max_depth or removing pruning.")
    print("Test Classification report:")
    print(classification_report(y_test, y_test_pred, target_names=le.classes_))

    # Cross-validation for robust evaluation (5-fold stratified)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"\n🔄 5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"CV scores per fold: {cv_scores}")

    # Feature importances (for interpretability in DT)
    feature_names = ['age', 'weight'] + list(mlb.classes_)
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    print("\n📈 Top 10 Feature Importances:")
    print(importances.head(10))

    # X is matrix [age, weight, symptoms...]
    X_sym = X[:, 2:].astype(int)     # symptom binary columns

# quick empirical VC estimation (keep max_k small!)
    vc_emp, subset = empirical_vc_shatter(X_sym, max_k=5, max_train_samples=150, verbose=True)

    eps, delta = 0.05, 0.05
    m_vc = pac_sample_size_vc(vc_emp if vc_emp>0 else 1, eps, delta)
    log2H = approximate_log2_hypotheses_from_features(X_sym.shape[1], max_depth=3)
    m_finite = pac_sample_size_finite(log2H, eps, delta)
    print(f"Empirical VC: {vc_emp}, subset indices: {subset}")
    print(f"Suggested sample size (VC-bound): {m_vc}")
    print(f"Suggested sample size (finite-hypothesis approx): {m_finite}")

    # Save artifacts
    joblib.dump(model, MODEL_FILE)
    joblib.dump(mlb, MLB_FILE)
    joblib.dump(le, LE_FILE)
    print("\n💾 Saved:", MODEL_FILE, MLB_FILE, LE_FILE)

    plt.figure(figsize=(20, 10))
    feature_names = ['age', 'weight'] + list(mlb.classes_)
    plot_tree(model, feature_names=feature_names, class_names=le.classes_, filled=True, rounded=True, max_depth=3)
    plt.savefig('decision_tree.png')  # Saves as image
    plt.show()  # Or display in Jupyter
    print("\n🌳 Tree visualization saved as 'decision_tree.png'")

if __name__ == "__main__":
    main()

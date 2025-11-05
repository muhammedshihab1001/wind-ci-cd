import os, json, joblib
from pathlib import Path
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

ART = Path("artifacts")
ART.mkdir(parents=True, exist_ok=True)

def main():
    # 1) Load data
    data = load_wine()
    X, y = data.data, data.target  # 13 features, 3 classes

    # 2) Split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 3) Hyperparams via env for quick experimentation
    n_estimators = int(os.getenv("N_ESTIMATORS", "150"))
    max_depth = os.getenv("MAX_DEPTH")
    max_depth = int(max_depth) if max_depth else None

    # 4) Train
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    clf.fit(Xtr, ytr)

    # 5) Evaluate
    yhat = clf.predict(Xte)
    acc = float(accuracy_score(yte, yhat))
    f1m = float(f1_score(yte, yhat, average="macro"))

    # 6) Save artifacts
    joblib.dump(clf, ART / "model.pkl")
    (ART / "metrics.json").write_text(json.dumps({"accuracy": acc, "f1_macro": f1m}, indent=2))

    print(f"Trained: accuracy={acc:.4f}, f1_macro={f1m:.4f}")
    print("Artifacts saved to artifacts/model.pkl & artifacts/metrics.json")

if __name__ == "__main__":
    main()




#{"features":[14.23,1.71,2.43,15.6,127.0,2.80,3.06,0.28,2.29,5.64,1.04,3.92,1065.0]}
#{"features":[12.37,1.07,2.10,18.5,88.0,3.52,3.75,0.24,1.95,4.50,1.04,2.77,660.0]}
#{"features":[12.29,3.17,2.21,18.0,88.0,2.85,2.99,0.45,2.81,2.30,1.42,2.83,406.0]}

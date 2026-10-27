"""Educational credit scoring classifier using synthetic data only."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, ConfusionMatrixDisplay

from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(HERE / "credit_scoring_synthetic.csv")

print(df.head())
print("Dataset loaded successfully!")

X = df.drop(columns=["creditworthy"])
y = df["creditworthy"]  # 1 = creditworthy, 0 = higher-risk example

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=250, random_state=42, class_weight="balanced", min_samples_leaf=2
    )
}

results = {}
best_name, best_model, best_f1 = None, None, -1
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob))
    }
    results[name] = metrics
    print(f"\n{name}\n{classification_report(y_test, pred, zero_division=0)}")
    print("Metrics:", metrics)
    if metrics["f1_score"] > best_f1:
        best_name, best_model, best_f1 = name, model, metrics["f1_score"]

with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax, cmap="Blues")
ax.set_title(f"Credit Scoring — {best_name}")
fig.tight_layout()
fig.savefig(OUT / "confusion_matrix.png", dpi=200)
plt.close(fig)

if hasattr(best_model, "feature_importances_"):
    importance = best_model.feature_importances_
    names = X.columns
else:
    importance = abs(best_model.named_steps["model"].coef_[0])
    names = X.columns
imp = pd.Series(importance, index=names).sort_values()
fig, ax = plt.subplots(figsize=(8, 5))
imp.plot(kind="barh", ax=ax)
ax.set_title(f"Feature Importance — {best_name}")
ax.set_xlabel("Importance (absolute coefficient for Logistic Regression)")
fig.tight_layout()
fig.savefig(OUT / "feature_importance.png", dpi=200)
plt.close(fig)

print(f"\nBest model by F1 score: {best_name}")
print(f"Saved metrics and plots to: {OUT}")
print("LIMITATION: synthetic data only; do not use this model for real credit decisions.")

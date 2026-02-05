import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline

path = os.path.join(os.path.dirname(__file__), "..")
if path not in sys.path:
    sys.path.insert(0, path)

from data_import import df, target, features
from data_preprocess import preprocess

# Baseline logistic regression model setup
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=2000))
])

# Train-test split 80%/20%
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

#
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)[:, 1]
print(proba)

print("ROC AUC:", roc_auc_score(y_test, proba))
print("PR AUC :", average_precision_score(y_test, proba))
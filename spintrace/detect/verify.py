"""Derivative score: a logistic combination of the signals, weights and threshold fit on the dev split only."""
import json
import math

from .. import config

MODEL_PATH = config.RESULTS / "verifier.json"

# Used only until `spintrace eval` has fit the model on the dev split.
DEFAULT = {
    "features": ["align_coverage", "align_order", "ordered_coverage", "fact_overlap", "rare_cos", "shingle_containment"],
    "weights": [4.0, 1.0, 3.0, 2.0, 2.0, 2.0], "bias": -6.5, "threshold": 0.5, "source": "default",
}


def load():
    if MODEL_PATH.exists():
        return json.loads(MODEL_PATH.read_text())
    return DEFAULT


def score(sig, model):
    z = model["bias"] + sum(w * sig.get(f, 0.0) for f, w in zip(model["features"], model["weights"]))
    return 1 / (1 + math.exp(-z))


def fit(X, y, features, threshold_fn):
    """Fit logistic regression on dev pairs and pick the F1-maximising threshold on the same dev data."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    model = {"features": features, "weights": clf.coef_[0].tolist(), "bias": float(clf.intercept_[0]),
             "threshold": 0.5, "source": "dev split"}
    model["threshold"] = threshold_fn(model)
    return model


def save(model):
    config.RESULTS.mkdir(exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, indent=2))

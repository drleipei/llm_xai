from sklearn.base import BaseEstimator, ClassifierMixin


class ThresholdPredictor(ClassifierMixin, BaseEstimator):
    """
    Frozen binary-classification wrapper using a fixed
    validation-selected decision threshold.

    The fit() method exists only for scikit-learn estimator
    compatibility. Retraining is intentionally prohibited.
    """

    def __init__(self, estimator, threshold=0.5):
        self.estimator = estimator
        self.threshold = float(threshold)

        if hasattr(estimator, "n_features_in_"):
            self.n_features_in_ = estimator.n_features_in_

        if hasattr(estimator, "feature_names_in_"):
            self.feature_names_in_ = estimator.feature_names_in_

        if hasattr(estimator, "classes_"):
            self.classes_ = estimator.classes_

    def fit(self, X, y=None):
        raise RuntimeError("ThresholdPredictor is frozen and cannot be retrained.")

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        probability = self.predict_proba(X)[:, 1]

        return (probability >= self.threshold).astype(int)

""""Model explainability. """

import shap
import lime
import lime.lime_tabular


class ModelExplainability:
    def __init__(self, model, X_train, X_test, feature_names):
        """
        Initialize Model Explainability with trained model and dataset.
        :param model: Trained ML model (either sklearn or deep learning model).
        :param X_train: Training data for explainability.
        :param X_test: Test data to analyze.
        :param feature_names: List of feature names.
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names

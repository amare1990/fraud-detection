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

    ### SHAP EXPLAINABILITY ###
    def shap_explain(self):
        """
        Explain the model using SHAP values.
        """
        print("Generating SHAP explanations...")
        # Use SHAP KernelExplainer for black-box models
        explainer = shap.Explainer(self.model.predict, self.X_train)
        shap_values = explainer(self.X_test)

        # Summary Plot: Shows global feature importance
        shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names)

        # Force Plot: Visualizes individual predictions
        shap.initjs()
        sample_index = 0
        shap.force_plot(explainer.expected_value, shap_values[sample_index], self.X_test.iloc[sample_index, :]\
                        , feature_names=self.feature_names)
        # Dependence Plot: Shows how a single feature affects predictions
        feature_index = 0
        shap.dependence_plot(feature_index, shap_values, self.X_test, feature_names=self.feature_names)

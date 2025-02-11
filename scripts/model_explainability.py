""""Model explainability. """

import shap
import lime
import lime.lime_tabular

import matplotlib.pyplot as plt


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

    # SHAP EXPLAINABILITY
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

        # Save force plot as an image
        shap.save_html("/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/notebooks/plots/explainability/shap_force_plot.html", force_plot)  # Save as HTML
        # Dependence Plot: Shows how a single feature affects predictions
        feature_index = 0
        shap.dependence_plot(feature_index, shap_values[1], self.X_test, feature_names=self.feature_names)
        # Save dependence plot as PNG
        plt.savefig("/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/notebooks/plots/explainability/shap_dependence_plot.png", dpi=300, bbox_inches="tight")

        # Save as PDF (optional)
        plt.savefig("/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/notebooks/plots/explainability/shap_dependence_plot.pdf", dpi=300, bbox_inches="tight")

        # Show and close the plot
        plt.show()
        plt.close()


    # LIME EXPLAINABILITY
    def lime_explain(self, sample_index=0):
        """
        Explain a model's prediction using LIME.
        :param sample_idx: Index of the sample to explain.
        """
        print("Generating LIME explanations...")

        explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.to_numpy(),
            feature_names=self.feature_names,
            class_names=[0, 1],
            mode="classification"
        )

        # Explain a single instance
        sample = self.X_test.iloc[sample_index].to_numpy()
        exp = explainer.explain_instance(sample, self.model.predict_proba, num_features=10)

        # Show feature importance
        exp.show_in_notebook()
        exp.as_pyplot_figure()
        plt.savefig(f"/home/am/Documents/Software Development/10_Academy Training/week_8-9/fraud-detection/notebooks/plots/explainability/lime_explanations.png", dpi=300, bbox_inches="tight")
        plt.show()

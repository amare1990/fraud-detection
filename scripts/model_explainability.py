"""Model explainability using SHAP and LIME. """

import os

import numpy as np
import pandas as pd

import shap
import lime
import lime.lime_tabular

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


class ModelExplainability:
    def __init__(self, model, X_train, X_test, feature_names, base_dir):
        """
        Initialize Model Explainability with trained model and dataset.
        :param model: Trained ML model (either sklearn or deep learning model).
        :param X_train: Training data for explainability.
        :param X_test: Test data to analyze.
        :param feature_names: List of feature names.
        :param base_dir: Base directory for saving plots.
        """
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.base_dir = base_dir

        # Initialize SHAP Explainer
        if isinstance(self.model, (RandomForestClassifier, DecisionTreeClassifier, GradientBoostingClassifier)):
            self.shap_explainer = shap.TreeExplainer(self.model)
        else:
            self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, self.X_train)


        # Initialize LIME Explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.feature_names,
            class_names=[0, 1],
            mode="classification",
            discretize_continuous=False
        )

    # SHAP EXPLAINABILITY
    def shap_explain(self, sample_index=0, feature_index=0):
        print("Generating SHAP explanations...")

        shap_values = self.shap_explainer.shap_values(self.X_test)
        expected_value = self.shap_explainer.expected_value

        print("Expected Value Shape:", np.shape(expected_value))
        print("SHAP Values Shape:", np.shape(shap_values))


        # Handle multi-class output and 3D SHAP values for deep learning models
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Select class 1 for binary classification
            expected_value = expected_value[1] if isinstance(expected_value, list) else expected_value
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3: #For deep learning models
          shap_values = shap_values[:, :, 1]  # Select class 1 and reduce to 2D for binary classification


        # Summary Plot
        shap.summary_plot(
            shap_values,
            self.X_test,
            feature_names=np.array(self.feature_names))

        shap_path = os.path.join(
            self.base_dir,
            "plots",
            "explainability",
            f"shap_summary_plot_{self.model}.png")

        plt.title(f'SHAP Summary Plot for {self.model}')
        plt.savefig(shap_path, dpi=300, bbox_inches="tight")
        plt.show(f"SHAP summary plot saved at {shap_path}")
        plt.close()

        print("Plotting summary plot completed.")

        # Force Plot (Fixed for SHAP v0.20+)
        if isinstance(expected_value, (list, np.ndarray)):  # Support both lists and arrays
            force_plot = shap.plots.force(expected_value[sample_index], shap_values[sample_index, :])
        else:
            force_plot = shap.plots.force(expected_value, shap_values[sample_index])


        shap_individual_path = os.path.join(
            self.base_dir,
            "plots", "explainability",
            f"shap_force_plot_{self.model}_{sample_index}.html")

        shap.save_html(shap_individual_path, force_plot)
        print(f"SHAP Force Plot saved at {shap_individual_path}")



        # Dependence Plot (fix the feature_index and shap_values for deep learning models)
        # Adjust feature_index and shap_values if necessary
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
          shap.dependence_plot(feature_index,
                               shap_values,
                               self.X_test,
                               feature_names=np.array(self.feature_names))
        else:  # This block is only reached if shap_values is not the expected shape.
          pass


        shap_single_path = os.path.join(
            self.base_dir,
            "plots",
            "explainability",
            f"shap_dependence_plot_{self.model}_{feature_index}.png")

        plt.title(f'SHAP Dependence Plot for Feature {feature_index}')

        plt.savefig(shap_single_path, dpi=300, bbox_inches="tight")
        print(f"SHAP dependence Plot for Feature {feature_index} saved at {shap_single_path}")
        plt.show()
        plt.close()

        print("SHAP explanations generated successfully.")


    # LIME EXPLAINABILITY
    def lime_explain(self, sample_index=0, num_features=10):
        print(f"Generating LIME explanation for sample {sample_index}...")

        # Get the sample data and reshape it to a 2D array
        sample = self.X_test.iloc[[sample_index]].values  # Use double brackets to keep 2D shape

        exp = self.lime_explainer.explain_instance(
            sample[0],  # Access the first (and only) row of the 2D array
            self.model.predict_proba,
            num_features=num_features
        )

        # Generate explanation plot
        lime_path = os.path.join(
            self.base_dir,
            "plots",
            "explainability",
            f"lime_explanations_{self.model}_{sample_index}.png")

        exp.show_in_notebook()
        fig = exp.as_pyplot_figure()
        fig.savefig(lime_path, dpi=300, bbox_inches="tight")
        plt.title(f"LIME Explanation for Sample {self.model}_{sample_index}")
        plt.show()
        plt.close(fig)

        print(f"LIME explanation saved for sample {sample_index}.")

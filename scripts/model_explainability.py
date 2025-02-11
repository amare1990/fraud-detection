import os
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


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
        self.base_dir = base_dir  # Added base directory as a parameter

        # Initialize SHAP Explainer
        if isinstance(self.model, (RandomForestClassifier,
                                   DecisionTreeClassifier, GradientBoostingClassifier)):
            self.shap_explainer = shap.TreeExplainer(self.model)
        else:
            # Reduce sample size for efficiency
            background_sample = shap.sample(self.X_train, 10)
            self.shap_explainer = shap.KernelExplainer(
                self.model.predict_proba, background_sample)

        # Initialize LIME Explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.feature_names,
            class_names=["Class 0", "Class 1"],
            mode="classification",
            discretize_continuous=False
        )

    # SHAP EXPLAINABILITY
    def shap_explain(self, sample_index=0, feature_index=0):
        print("Generating SHAP explanations...")

        shap_values = self.shap_explainer.shap_values(self.X_test)
        expected_value = self.shap_explainer.expected_value

        print("SHAP explanations generated.")

        # Summary Plot
        shap.summary_plot(
            shap_values,
            self.X_test,
            feature_names=self.feature_names)
        shap_path = os.path.join(
            self.base_dir,
            "plots",
            "explainability",
            "shap_summary_plot.png")
        plt.savefig(shap_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Force Plot
        shap.initjs()
        force_plot = shap.force_plot(expected_value[1],
                                     shap_values[1][sample_index],
                                     self.X_test.iloc[sample_index,
                                                      :],
                                     feature_names=self.feature_names)
        shap_individual_path = os.path.join(
            self.base_dir,
            "plots",
            "explainability",
            f"shap_force_plot_{sample_index}.html")
        shap.save_html(shap_individual_path, force_plot)

        # Dependence Plot
        shap.dependence_plot(
            feature_index,
            shap_values[1],
            self.X_test,
            feature_names=self.feature_names)
        shap_single_path = os.path.join(
            self.base_dir,
            'plots',
            'explainability',
            f'shap_dependence_plot_{feature_index}.png')
        plt.savefig(shap_single_path, dpi=300, bbox_inches="tight")
        plt.close()

    # LIME EXPLAINABILITY
    def lime_explain(self, sample_index=0, num_features=10):
        print(f"Generating LIME explanation for sample {sample_index}...")

        sample = self.X_test.iloc[sample_index].values
        exp = self.lime_explainer.explain_instance(
            sample, self.model.predict_proba, num_features=num_features)

        # Generate explanation plot
        lime_path = os.path.join(
            self.base_dir,
            "plots",
            "explainability",
            f"lime_explanations_{sample_index}.png")
        plt.savefig(lime_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"LIME explanation saved for sample {sample_index}.")

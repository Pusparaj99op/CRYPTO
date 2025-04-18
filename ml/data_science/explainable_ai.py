"""
Explainable AI Module for Cryptocurrency Trading

This module provides tools for interpreting and explaining machine learning
models used in cryptocurrency trading strategies.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
import shap

class FeatureImportanceExplainer:
    """Class for explaining feature importance in ML models."""
    
    def __init__(self, model: BaseEstimator):
        """
        Initialize feature importance explainer.
        
        Args:
            model: Trained model to explain
        """
        self.model = model
        self.importances = {}
        self.feature_names = None
    
    def explain_feature_importance(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
                               method: str = 'built_in') -> pd.Series:
        """
        Calculate feature importance using various methods.
        
        Args:
            X: Feature data
            y: Target data (required for permutation importance)
            method: Method to use ('built_in', 'permutation', 'shap')
            
        Returns:
            Series with feature importances
        """
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        if method == 'built_in':
            # Use model's built-in feature importance if available
            if hasattr(self.model, 'feature_importances_'):
                importances = pd.Series(
                    self.model.feature_importances_,
                    index=self.feature_names
                )
            elif hasattr(self.model, 'coef_'):
                # For linear models
                importances = pd.Series(
                    np.abs(self.model.coef_),
                    index=self.feature_names
                )
            else:
                raise ValueError("Model doesn't have built-in feature importance")
                
        elif method == 'permutation':
            # Use permutation importance
            if y is None:
                raise ValueError("Target values y required for permutation importance")
                
            perm_importance = permutation_importance(
                self.model, X, y, n_repeats=10, random_state=42
            )
            
            importances = pd.Series(
                perm_importance.importances_mean,
                index=self.feature_names
            )
            
        elif method == 'shap':
            # Use SHAP values
            # Create explainer based on model type
            if hasattr(self.model, 'predict_proba'):
                explainer = shap.Explainer(self.model, X)
            else:
                explainer = shap.Explainer(self.model)
                
            # Calculate SHAP values
            shap_values = explainer(X)
            
            # Get mean absolute SHAP values for each feature
            importances = pd.Series(
                np.abs(shap_values.values).mean(axis=0),
                index=self.feature_names
            )
            
            # Store explainer and values for later use
            self.shap_explainer = explainer
            self.shap_values = shap_values
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalize to sum to 1
        importances = importances / importances.sum()
        
        # Store importances
        self.importances[method] = importances
        
        return importances
    
    def plot_feature_importance(self, method: str = 'built_in', 
                             top_n: int = None, figsize: Tuple = (10, 8)):
        """
        Plot feature importance.
        
        Args:
            method: Method used to calculate importance
            top_n: Number of top features to show
            figsize: Figure size
            
        Returns:
            Matplotlib figure with feature importance plot
        """
        if method not in self.importances:
            raise ValueError(f"Importance for method {method} not calculated. "
                           f"Run explain_feature_importance first.")
        
        importances = self.importances[method]
        
        # Get top N features if specified
        if top_n is not None:
            importances = importances.nlargest(top_n)
        
        # Sort for better visualization
        importances = importances.sort_values()
        
        plt.figure(figsize=figsize)
        plt.barh(importances.index, importances.values)
        plt.xlabel('Importance')
        plt.title(f'Feature Importance ({method})')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_shap_summary(self, figsize: Tuple = (12, 8)):
        """
        Plot SHAP summary plot.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure with SHAP summary plot
        """
        if not hasattr(self, 'shap_values'):
            raise ValueError("SHAP values not calculated. "
                           "Run explain_feature_importance with method='shap' first.")
        
        plt.figure(figsize=figsize)
        shap.summary_plot(self.shap_values, plot_type="bar")
        return plt.gcf()
    
    def get_top_features(self, method: str = 'built_in', top_n: int = 10) -> pd.Series:
        """
        Get top N most important features.
        
        Args:
            method: Method used to calculate importance
            top_n: Number of top features to return
            
        Returns:
            Series with top feature importances
        """
        if method not in self.importances:
            raise ValueError(f"Importance for method {method} not calculated. "
                           f"Run explain_feature_importance first.")
        
        return self.importances[method].nlargest(top_n)


class PartialDependenceExplainer:
    """Class for explaining partial dependence in ML models."""
    
    def __init__(self, model: BaseEstimator):
        """
        Initialize partial dependence explainer.
        
        Args:
            model: Trained model to explain
        """
        self.model = model
        self.pd_results = {}
    
    def calculate_partial_dependence(self, X: pd.DataFrame, features: List[str], 
                                  grid_resolution: int = 20) -> Dict:
        """
        Calculate partial dependence for specified features.
        
        Args:
            X: Feature data
            features: Features to calculate PD for
            grid_resolution: Number of points in grid
            
        Returns:
            Dictionary with partial dependence results
        """
        from sklearn.inspection import partial_dependence
        
        results = {}
        
        for feature in features:
            if feature not in X.columns:
                raise ValueError(f"Feature {feature} not found in data")
            
            # Calculate partial dependence
            pd_result = partial_dependence(
                self.model, X, [feature], 
                grid_resolution=grid_resolution
            )
            
            # Extract values
            grid_values = pd_result["values"][0]
            pd_values = pd_result["average"][0]
            
            # Store results
            results[feature] = {
                'grid_values': grid_values,
                'pd_values': pd_values
            }
        
        self.pd_results = results
        return results
    
    def plot_partial_dependence(self, features: Optional[List[str]] = None, 
                             figsize: Tuple = (12, 4), ncols: int = 3):
        """
        Plot partial dependence for specified features.
        
        Args:
            features: Features to plot (if None, plot all calculated)
            figsize: Base figure size (per subplot)
            ncols: Number of columns in subplot grid
            
        Returns:
            Matplotlib figure with partial dependence plots
        """
        if not self.pd_results:
            raise ValueError("Partial dependence not calculated. "
                           "Run calculate_partial_dependence first.")
        
        # Determine features to plot
        if features is None:
            features = list(self.pd_results.keys())
        else:
            # Check if we have results for all requested features
            for feature in features:
                if feature not in self.pd_results:
                    raise ValueError(f"Partial dependence for {feature} not calculated.")
        
        # Create subplot grid
        n_features = len(features)
        nrows = (n_features + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, 
            figsize=(figsize[0] * ncols, figsize[1] * nrows)
        )
        
        # Make axes iterable if there's only one
        if n_features == 1:
            axes = [axes]
        
        # Flatten axes for easy iteration
        if nrows > 1 and ncols > 1:
            axes_flat = axes.flatten()
        else:
            axes_flat = axes
        
        # Plot each feature
        for i, feature in enumerate(features):
            if i < len(axes_flat):
                ax = axes_flat[i]
                
                # Get values
                grid_values = self.pd_results[feature]['grid_values']
                pd_values = self.pd_results[feature]['pd_values']
                
                # Plot
                ax.plot(grid_values, pd_values)
                ax.set_xlabel(feature)
                ax.set_ylabel('Partial Dependence')
                ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for j in range(i+1, len(axes_flat)):
            axes_flat[j].axis('off')
        
        plt.tight_layout()
        return fig


class LocalExplainer:
    """Class for local explanations of model predictions."""
    
    def __init__(self, model: BaseEstimator):
        """
        Initialize local explainer.
        
        Args:
            model: Trained model to explain
        """
        self.model = model
        self.shap_explainer = None
        self.lime_explainer = None
    
    def explain_prediction_with_shap(self, X: pd.DataFrame, instance_idx: int = 0) -> Dict:
        """
        Explain a prediction using SHAP.
        
        Args:
            X: Feature data
            instance_idx: Index of instance to explain
            
        Returns:
            Dictionary with explanation
        """
        # Create SHAP explainer if not already created
        if self.shap_explainer is None:
            self.shap_explainer = shap.Explainer(self.model, X)
        
        # Get the instance to explain
        if isinstance(X, pd.DataFrame):
            instance = X.iloc[[instance_idx]]
            feature_names = X.columns.tolist()
        else:
            instance = X[[instance_idx]]
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        
        # Calculate SHAP values
        shap_values = self.shap_explainer(instance)
        
        # Get model prediction
        if hasattr(self.model, 'predict_proba'):
            prediction = self.model.predict_proba(instance)[0]
            predicted_class = self.model.predict(instance)[0]
        else:
            prediction = self.model.predict(instance)[0]
            predicted_class = None
        
        # Create explanation dictionary
        explanation = {
            'shap_values': shap_values,
            'feature_values': instance.values[0],
            'feature_names': feature_names,
            'prediction': prediction,
            'predicted_class': predicted_class
        }
        
        return explanation
    
    def explain_prediction_with_lime(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
                                 instance_idx: int = 0, num_features: int = 10) -> Dict:
        """
        Explain a prediction using LIME.
        
        Args:
            X: Feature data
            y: Target data (for classification)
            instance_idx: Index of instance to explain
            num_features: Number of features to include in explanation
            
        Returns:
            Dictionary with explanation
        """
        try:
            import lime
            import lime.lime_tabular
        except ImportError:
            raise ImportError(
                "lime package is required for LIME explanations. "
                "Install it with: pip install lime"
            )
        
        # Check if we're doing classification or regression
        if hasattr(self.model, 'predict_proba'):
            mode = 'classification'
        else:
            mode = 'regression'
        
        # Get feature names
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        
        # Create LIME explainer if not already created
        if self.lime_explainer is None:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X.values,
                feature_names=feature_names,
                class_names=None if y is None else y.unique().tolist(),
                mode=mode
            )
        
        # Get the instance to explain
        if isinstance(X, pd.DataFrame):
            instance = X.iloc[instance_idx].values
        else:
            instance = X[instance_idx]
        
        # Get model prediction function
        if mode == 'classification':
            predict_fn = self.model.predict_proba
        else:
            predict_fn = self.model.predict
        
        # Calculate LIME explanation
        lime_exp = self.lime_explainer.explain_instance(
            instance, predict_fn, num_features=num_features
        )
        
        # Get model prediction
        if mode == 'classification':
            prediction = self.model.predict_proba([instance])[0]
            predicted_class = self.model.predict([instance])[0]
        else:
            prediction = self.model.predict([instance])[0]
            predicted_class = None
        
        # Create explanation dictionary
        explanation = {
            'lime_exp': lime_exp,
            'feature_values': instance,
            'feature_names': feature_names,
            'prediction': prediction,
            'predicted_class': predicted_class
        }
        
        return explanation
    
    def plot_shap_explanation(self, explanation: Dict, figsize: Tuple = (10, 6)):
        """
        Plot SHAP explanation.
        
        Args:
            explanation: Explanation from explain_prediction_with_shap
            figsize: Figure size
            
        Returns:
            Matplotlib figure with SHAP explanation
        """
        plt.figure(figsize=figsize)
        shap.plots.waterfall(explanation['shap_values'][0], show=False)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_lime_explanation(self, explanation: Dict, figsize: Tuple = (10, 6)):
        """
        Plot LIME explanation.
        
        Args:
            explanation: Explanation from explain_prediction_with_lime
            figsize: Figure size
            
        Returns:
            Matplotlib figure with LIME explanation
        """
        if 'lime_exp' not in explanation:
            raise ValueError("Not a valid LIME explanation")
        
        plt.figure(figsize=figsize)
        explanation['lime_exp'].as_pyplot_figure()
        plt.tight_layout()
        return plt.gcf()


class GlobalExplainer:
    """Class for global model explanations."""
    
    def __init__(self, model: BaseEstimator, feature_names: List[str] = None):
        """
        Initialize global explainer.
        
        Args:
            model: Trained model to explain
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names
        self.global_shap_values = None
    
    def explain_model_with_shap(self, X: pd.DataFrame) -> np.ndarray:
        """
        Explain model globally using SHAP.
        
        Args:
            X: Feature data
            
        Returns:
            SHAP values
        """
        # Create SHAP explainer
        explainer = shap.Explainer(self.model, X)
        
        # Calculate SHAP values
        shap_values = explainer(X)
        
        # Store values
        self.global_shap_values = shap_values
        
        # Store feature names if not provided
        if self.feature_names is None and isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        return shap_values
    
    def plot_shap_summary(self, plot_type: str = "bar", figsize: Tuple = (10, 8)):
        """
        Plot SHAP summary.
        
        Args:
            plot_type: Type of plot ('bar', 'dot', 'violin')
            figsize: Figure size
            
        Returns:
            Matplotlib figure with SHAP summary
        """
        if self.global_shap_values is None:
            raise ValueError("SHAP values not calculated. "
                           "Run explain_model_with_shap first.")
        
        plt.figure(figsize=figsize)
        shap.summary_plot(
            self.global_shap_values, 
            plot_type=plot_type,
            feature_names=self.feature_names,
            show=False
        )
        plt.tight_layout()
        return plt.gcf()
    
    def plot_shap_dependence(self, feature_idx: Union[int, str], 
                          interaction_idx: Union[int, str, None] = "auto",
                          figsize: Tuple = (10, 8)):
        """
        Plot SHAP dependence.
        
        Args:
            feature_idx: Index or name of feature to plot
            interaction_idx: Index or name of feature to use for interaction
            figsize: Figure size
            
        Returns:
            Matplotlib figure with SHAP dependence plot
        """
        if self.global_shap_values is None:
            raise ValueError("SHAP values not calculated. "
                           "Run explain_model_with_shap first.")
        
        # Convert feature name to index if needed
        if isinstance(feature_idx, str) and self.feature_names is not None:
            if feature_idx in self.feature_names:
                feature_idx = self.feature_names.index(feature_idx)
            else:
                raise ValueError(f"Feature {feature_idx} not found in feature names")
        
        # Convert interaction feature name to index if needed
        if isinstance(interaction_idx, str) and interaction_idx != "auto" and self.feature_names is not None:
            if interaction_idx in self.feature_names:
                interaction_idx = self.feature_names.index(interaction_idx)
            else:
                raise ValueError(f"Feature {interaction_idx} not found in feature names")
        
        plt.figure(figsize=figsize)
        shap.dependence_plot(
            feature_idx, 
            self.global_shap_values.values, 
            self.global_shap_values.data,
            interaction_index=interaction_idx,
            feature_names=self.feature_names
        )
        plt.tight_layout()
        return plt.gcf()


class RuleExtractor:
    """Class for extracting rules from ML models."""
    
    def __init__(self, model: BaseEstimator):
        """
        Initialize rule extractor.
        
        Args:
            model: Trained model to extract rules from
        """
        self.model = model
        self.rules = None
    
    def extract_rules_from_tree(self, feature_names: List[str] = None, class_names: List[str] = None) -> List[str]:
        """
        Extract rules from tree-based models.
        
        Args:
            feature_names: Names of features
            class_names: Names of classes (for classification)
            
        Returns:
            List of rules as strings
        """
        # Check if model is tree-based
        tree_models = (
            'DecisionTreeClassifier', 'DecisionTreeRegressor',
            'ExtraTreeClassifier', 'ExtraTreeRegressor'
        )
        
        model_type = type(self.model).__name__
        
        if model_type not in tree_models:
            raise ValueError(f"Model {model_type} is not a supported tree model")
        
        # Import tree module
        from sklearn import tree
        
        # Get model's tree
        estimator = self.model
        
        # Extract rules
        rules = []
        
        def tree_to_rules(tree_model, node_id=0, depth=0, condition=""):
            # If leaf node, return the predicted value
            if tree_model.tree_.children_left[node_id] == -1:
                if hasattr(tree_model, 'classes_'):
                    # Classification
                    class_idx = np.argmax(tree_model.tree_.value[node_id])
                    class_val = (
                        class_idx if class_names is None else class_names[class_idx]
                    )
                    rules.append(f"{condition} THEN class = {class_val}")
                else:
                    # Regression
                    value = tree_model.tree_.value[node_id][0, 0]
                    rules.append(f"{condition} THEN value = {value:.4f}")
                return
            
            # Get feature name
            feature_idx = tree_model.tree_.feature[node_id]
            feature = (
                f"feature_{feature_idx}" if feature_names is None 
                else feature_names[feature_idx]
            )
            
            # Get threshold
            threshold = tree_model.tree_.threshold[node_id]
            
            # Process left child (feature <= threshold)
            new_condition = f"{condition} AND {feature} <= {threshold:.4f}" if condition else f"IF {feature} <= {threshold:.4f}"
            tree_to_rules(
                tree_model, 
                tree_model.tree_.children_left[node_id], 
                depth + 1, 
                new_condition
            )
            
            # Process right child (feature > threshold)
            new_condition = f"{condition} AND {feature} > {threshold:.4f}" if condition else f"IF {feature} > {threshold:.4f}"
            tree_to_rules(
                tree_model, 
                tree_model.tree_.children_right[node_id], 
                depth + 1, 
                new_condition
            )
        
        # Generate rules
        tree_to_rules(estimator)
        
        self.rules = rules
        return rules
    
    def extract_rules_from_ensemble(self, max_trees: int = 3, 
                                 feature_names: List[str] = None,
                                 class_names: List[str] = None) -> List[str]:
        """
        Extract rules from ensemble tree-based models.
        
        Args:
            max_trees: Maximum number of trees to extract rules from
            feature_names: Names of features
            class_names: Names of classes (for classification)
            
        Returns:
            List of rules as strings
        """
        # Check if model is ensemble
        ensemble_models = (
            'RandomForestClassifier', 'RandomForestRegressor',
            'GradientBoostingClassifier', 'GradientBoostingRegressor',
            'ExtraTreesClassifier', 'ExtraTreesRegressor'
        )
        
        model_type = type(self.model).__name__
        
        if model_type not in ensemble_models:
            raise ValueError(f"Model {model_type} is not a supported ensemble model")
        
        # Extract rules from each tree in the ensemble
        all_rules = []
        
        # Limit the number of trees
        n_trees = min(len(self.model.estimators_), max_trees)
        
        for i in range(n_trees):
            tree_model = self.model.estimators_[i]
            
            # Create rule extractor for this tree
            tree_extractor = RuleExtractor(tree_model)
            tree_rules = tree_extractor.extract_rules_from_tree(
                feature_names=feature_names,
                class_names=class_names
            )
            
            # Add tree identifier to rules
            tree_rules = [f"Tree {i+1}: {rule}" for rule in tree_rules]
            all_rules.extend(tree_rules)
        
        self.rules = all_rules
        return all_rules
    
    def print_rules(self, max_rules: int = None):
        """
        Print extracted rules.
        
        Args:
            max_rules: Maximum number of rules to print (None for all)
        """
        if self.rules is None:
            raise ValueError("Rules not extracted. Run extract_rules first.")
        
        # Determine how many rules to print
        n_rules = len(self.rules) if max_rules is None else min(len(self.rules), max_rules)
        
        for i, rule in enumerate(self.rules[:n_rules]):
            print(f"Rule {i+1}: {rule}")
        
        if n_rules < len(self.rules):
            print(f"... ({len(self.rules) - n_rules} more rules)")


class CryptoModelExplainer:
    """Specialized class for explaining cryptocurrency trading models."""
    
    def __init__(self, model: BaseEstimator, feature_metadata: Optional[Dict] = None):
        """
        Initialize cryptocurrency model explainer.
        
        Args:
            model: Trained model to explain
            feature_metadata: Dictionary with feature metadata
        """
        self.model = model
        self.feature_metadata = feature_metadata or {}
        
        # Initialize individual explainers
        self.feature_explainer = FeatureImportanceExplainer(model)
        self.local_explainer = LocalExplainer(model)
        self.global_explainer = GlobalExplainer(model)
    
    def explain_trading_signal(self, X: pd.DataFrame, 
                            instance_idx: int = -1,
                            n_features: int = 10) -> Dict:
        """
        Explain trading signal for a specific market instance.
        
        Args:
            X: Feature data
            instance_idx: Index of instance to explain
            n_features: Number of top features to highlight
            
        Returns:
            Dictionary with explanation
        """
        # Get model prediction
        if hasattr(self.model, 'predict_proba'):
            prediction = self.model.predict_proba(X.iloc[[instance_idx]])[0]
            if len(prediction) == 2:  # Binary classification
                signal_strength = prediction[1]  # Probability of positive class
            else:
                signal_strength = prediction  # Multiclass
        else:
            signal_strength = self.model.predict(X.iloc[[instance_idx]])[0]
        
        # Get SHAP explanation
        shap_explanation = self.local_explainer.explain_prediction_with_shap(
            X, instance_idx=instance_idx
        )
        
        # Get instance data
        instance = X.iloc[instance_idx]
        
        # Categorize features by type if metadata available
        categorized_features = {}
        
        if self.feature_metadata:
            for feature in instance.index:
                if feature in self.feature_metadata:
                    category = self.feature_metadata[feature].get('category', 'Other')
                    if category not in categorized_features:
                        categorized_features[category] = []
                    categorized_features[category].append(feature)
        
        # Prepare explanation
        explanation = {
            'signal_strength': signal_strength,
            'instance': instance,
            'shap_explanation': shap_explanation,
            'top_features': pd.Series({
                feature: shap_explanation['shap_values'].values[0, i]
                for i, feature in enumerate(X.columns)
            }).abs().nlargest(n_features),
            'categorized_features': categorized_features
        }
        
        return explanation
    
    def plot_trading_signal_explanation(self, explanation: Dict, figsize: Tuple = (12, 8)):
        """
        Plot trading signal explanation.
        
        Args:
            explanation: Explanation from explain_trading_signal
            figsize: Figure size
            
        Returns:
            Matplotlib figure with explanation
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot feature contributions
        top_features = explanation['top_features']
        colors = ['green' if x > 0 else 'red' for x in top_features.values]
        
        top_features.sort_values().plot(
            kind='barh',
            ax=axes[0],
            color=colors
        )
        
        axes[0].set_title('Top Feature Contributions')
        axes[0].axvline(x=0, color='gray', linestyle='--')
        
        # Add signal strength display
        signal_strength = explanation['signal_strength']
        
        if isinstance(signal_strength, (float, int)):
            # Single value (regression or binary)
            signal_scale = np.linspace(0, 1, 100)
            signal_display = np.zeros_like(signal_scale)
            
            axes[1].plot(signal_scale, signal_display, 'k-', alpha=0.2)
            axes[1].axvline(x=signal_strength, color='blue', linestyle='-', linewidth=3)
            axes[1].set_ylim(-0.1, 0.1)
            axes[1].set_title(f'Signal Strength: {signal_strength:.4f}')
            
            # Add interpretation
            if signal_strength > 0.7:
                label = "Strong Buy Signal"
                color = 'darkgreen'
            elif signal_strength > 0.5:
                label = "Buy Signal"
                color = 'green'
            elif signal_strength > 0.3:
                label = "Weak Buy Signal"
                color = 'lightgreen'
            elif signal_strength > -0.3:
                label = "Neutral Signal"
                color = 'gray'
            elif signal_strength > -0.5:
                label = "Weak Sell Signal"
                color = 'salmon'
            elif signal_strength > -0.7:
                label = "Sell Signal"
                color = 'red'
            else:
                label = "Strong Sell Signal"
                color = 'darkred'
            
            axes[1].annotate(
                label,
                xy=(signal_strength, 0),
                xytext=(signal_strength, 0.05),
                arrowprops=dict(arrowstyle='->'),
                color=color,
                fontsize=12,
                ha='center'
            )
            
        else:
            # Multiple values (multiclass)
            classes = np.arange(len(signal_strength))
            axes[1].bar(classes, signal_strength, color='blue')
            axes[1].set_xticks(classes)
            axes[1].set_title('Class Probabilities')
        
        axes[1].grid(False)
        plt.tight_layout()
        
        return fig
    
    def analyze_feature_contribution_over_time(self, X: pd.DataFrame, 
                                          top_n_features: int = 5,
                                          figsize: Tuple = (12, 10)):
        """
        Analyze how feature contributions change over time.
        
        Args:
            X: Feature data (time-indexed)
            top_n_features: Number of top features to analyze
            figsize: Figure size
            
        Returns:
            Matplotlib figure with analysis
        """
        # Check if data is time-indexed
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("Data must be time-indexed for this analysis")
        
        # Get global feature importance
        self.feature_explainer.explain_feature_importance(X, method='shap')
        top_features = self.feature_explainer.get_top_features(
            method='shap', top_n=top_n_features
        ).index.tolist()
        
        # Calculate SHAP values for each instance
        if not hasattr(self.global_explainer, 'global_shap_values'):
            self.global_explainer.explain_model_with_shap(X)
        
        shap_values = self.global_explainer.global_shap_values
        
        # Extract time series of SHAP values for top features
        shap_time_series = pd.DataFrame(
            shap_values.values,
            index=X.index,
            columns=X.columns
        )[top_features]
        
        # Create plot
        fig, axes = plt.subplots(top_n_features + 1, 1, figsize=figsize, sharex=True)
        
        # Plot model predictions
        if hasattr(self.model, 'predict_proba'):
            predictions = self.model.predict_proba(X)[:, 1]  # Probability of positive class
        else:
            predictions = self.model.predict(X)
        
        pred_series = pd.Series(predictions, index=X.index)
        pred_series.plot(ax=axes[0], color='blue')
        axes[0].set_title('Model Predictions')
        axes[0].grid(alpha=0.3)
        
        # Plot SHAP values for top features
        for i, feature in enumerate(top_features):
            ax = axes[i+1]
            
            # Plot SHAP values
            shap_time_series[feature].plot(ax=ax, color='purple')
            ax.set_title(f'SHAP values: {feature}')
            ax.axhline(y=0, color='gray', linestyle='--')
            ax.grid(alpha=0.3)
            
            # Plot feature values on secondary axis
            if feature in X.columns:
                ax2 = ax.twinx()
                X[feature].plot(ax=ax2, color='green', alpha=0.5)
                ax2.spines['right'].set_visible(True)
                ax2.spines['right'].set_color('green')
                ax2.tick_params(axis='y', colors='green')
        
        plt.tight_layout()
        return fig


# Utility functions for model interpretation

def compare_models_shap(models: Dict[str, BaseEstimator], X: pd.DataFrame, 
                     figsize: Tuple = (15, 5 * len(models))):
    """
    Compare multiple models using SHAP values.
    
    Args:
        models: Dictionary mapping model names to model objects
        X: Feature data
        figsize: Figure size
        
    Returns:
        Matplotlib figure with model comparison
    """
    # Create figure
    fig, axes = plt.subplots(len(models), 1, figsize=figsize)
    
    # Make axes iterable if there's only one
    if len(models) == 1:
        axes = [axes]
    
    # Compare models
    for i, (model_name, model) in enumerate(models.items()):
        # Create explainer
        explainer = shap.Explainer(model, X)
        
        # Calculate SHAP values
        shap_values = explainer(X)
        
        # Plot SHAP summary
        plt.sca(axes[i])
        shap.summary_plot(
            shap_values, 
            plot_type="bar",
            show=False
        )
        plt.title(f"SHAP Feature Importance: {model_name}")
    
    plt.tight_layout()
    return fig

def analyze_prediction_differences(models: Dict[str, BaseEstimator], X: pd.DataFrame, 
                                instance_idx: int = 0, figsize: Tuple = (12, 8)):
    """
    Analyze differences in predictions between models.
    
    Args:
        models: Dictionary mapping model names to model objects
        X: Feature data
        instance_idx: Index of instance to analyze
        figsize: Figure size
        
    Returns:
        Matplotlib figure with analysis
    """
    # Get instance
    instance = X.iloc[[instance_idx]]
    
    # Calculate predictions and explanations
    predictions = {}
    explanations = {}
    
    for model_name, model in models.items():
        # Get prediction
        if hasattr(model, 'predict_proba'):
            pred = model.predict_proba(instance)[0]
            if len(pred) == 2:  # Binary classification
                pred = pred[1]  # Probability of positive class
        else:
            pred = model.predict(instance)[0]
        
        predictions[model_name] = pred
        
        # Get SHAP explanation
        explainer = shap.Explainer(model, X)
        shap_values = explainer(instance)
        
        explanations[model_name] = {
            'shap_values': shap_values,
            'feature_values': instance.values[0],
            'feature_names': X.columns.tolist()
        }
    
    # Create plot
    fig, axes = plt.subplots(len(models), 1, figsize=figsize)
    
    # Make axes iterable if there's only one
    if len(models) == 1:
        axes = [axes]
    
    # Plot explanations
    for i, (model_name, explanation) in enumerate(explanations.items()):
        plt.sca(axes[i])
        
        # Plot waterfall
        shap.plots.waterfall(explanation['shap_values'][0], show=False)
        plt.title(f"{model_name} - Prediction: {predictions[model_name]:.4f}")
    
    plt.tight_layout()
    return fig 
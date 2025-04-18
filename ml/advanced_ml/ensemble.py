"""
Advanced ensemble techniques for cryptocurrency prediction.
This module provides ensemble methods optimized for financial time series,
with a focus on cryptocurrency market prediction.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Optional, Tuple, Any


class RandomForestEnsemble:
    """Enhanced Random Forest Ensemble for cryptocurrency time series."""
    
    def __init__(self, n_estimators=100, max_features='auto', max_depth=None, 
                 min_samples_split=2, bootstrap=True, random_state=42,
                 feature_importance_method='permutation'):
        """
        Initialize enhanced Random Forest ensemble.
        
        Args:
            n_estimators: Number of trees in the forest
            max_features: Number of features to consider for best split
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split
            bootstrap: Whether to use bootstrap samples
            random_state: Random state for reproducibility
            feature_importance_method: Method to compute feature importance
                ('permutation' or 'impurity')
        """
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.feature_importance_method = feature_importance_method
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.feature_names = None
        self.feature_importances_ = None
    
    def fit(self, X, y, feature_names=None):
        """
        Fit the random forest ensemble to the training data.
        
        Args:
            X: Training features
            y: Target values
            feature_names: Names of the features (optional)
            
        Returns:
            Fitted model
        """
        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        self.model.fit(X, y)
        
        # Calculate feature importances
        if self.feature_importance_method == 'permutation':
            self._calculate_permutation_importance(X, y)
        else:
            self.feature_importances_ = self.model.feature_importances_
        
        return self
    
    def _calculate_permutation_importance(self, X, y, n_repeats=10):
        """Calculate permutation-based feature importance."""
        from sklearn.inspection import permutation_importance
        
        # Calculate permutation importance
        result = permutation_importance(
            self.model, X, y, n_repeats=n_repeats, random_state=self.random_state
        )
        self.feature_importances_ = result.importances_mean
    
    def predict(self, X):
        """
        Make predictions with the random forest ensemble.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def predict_with_uncertainty(self, X, percentiles=[5, 95]):
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X: Features for prediction
            percentiles: Percentiles for the uncertainty bounds
            
        Returns:
            Tuple of (mean predictions, lower bounds, upper bounds)
        """
        # Use individual tree predictions to estimate uncertainty
        predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        
        # Calculate mean prediction and percentiles
        mean_pred = np.mean(predictions, axis=0)
        lower_bounds = np.percentile(predictions, percentiles[0], axis=0)
        upper_bounds = np.percentile(predictions, percentiles[1], axis=0)
        
        return mean_pred, lower_bounds, upper_bounds
    
    def plot_feature_importance(self, top_n=10, save_path=None):
        """
        Plot feature importances.
        
        Args:
            top_n: Number of top features to display
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if self.feature_importances_ is None:
            raise ValueError("Model not fitted yet. Call fit() before plotting feature importance.")
        
        # Create a DataFrame for better visualization
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importances_
        })
        
        # Sort by importance and select top N
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return plt.gcf()


class GradientBoostingEnsemble:
    """Advanced Gradient Boosting Ensemble for cryptocurrency forecasting."""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, loss='squared_error', subsample=1.0,
                 random_state=42, early_stopping_rounds=None, validation_fraction=0.1):
        """
        Initialize Gradient Boosting Ensemble.
        
        Args:
            n_estimators: Number of boosting stages
            learning_rate: Learning rate shrinks the contribution of each tree
            max_depth: Maximum depth of the individual regression estimators
            min_samples_split: Minimum samples required to split an internal node
            loss: Loss function to be optimized
            subsample: Fraction of samples to be used for fitting the individual trees
            random_state: Random state for reproducibility
            early_stopping_rounds: Number of validation rounds without improvement to stop
            validation_fraction: Fraction of training data to use as validation set
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.subsample = subsample
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        
        # For sklearn compatibility
        if loss == 'squared_error' and hasattr(GradientBoostingRegressor, 'criterion'):
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                loss=loss,
                subsample=subsample,
                random_state=random_state,
                validation_fraction=validation_fraction if early_stopping_rounds else 0.0,
                n_iter_no_change=early_stopping_rounds if early_stopping_rounds else None,
                tol=0.0001
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                loss=loss,
                subsample=subsample,
                random_state=random_state
            )
        
        self.feature_names = None
        self.train_score = None
        self.val_score = None
        self.feature_importances_ = None
    
    def fit(self, X, y, feature_names=None, eval_set=None, eval_metric='mse'):
        """
        Fit the gradient boosting ensemble.
        
        Args:
            X: Training features
            y: Target values
            feature_names: Names of features
            eval_set: Optional validation set for early stopping
            eval_metric: Metric for evaluation
            
        Returns:
            Fitted model
        """
        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Convert to numpy arrays if DataFrame/Series
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
            
        # If using early stopping with custom validation set
        if self.early_stopping_rounds is not None and eval_set is not None:
            X_val, y_val = eval_set
            if hasattr(X_val, 'values'):
                X_val = X_val.values
            if hasattr(y_val, 'values'):
                y_val = y_val.values
            
            # Initialize lists to store scores
            train_scores = []
            val_scores = []
            
            # Manual early stopping implementation
            best_val_score = float('inf')
            best_iteration = 0
            best_model = None
            
            # Train initial model with a few estimators
            initial_estimators = min(10, self.n_estimators)
            self.model.set_params(n_estimators=initial_estimators)
            self.model.fit(X, y)
            
            # Continue training with early stopping
            for i in range(initial_estimators, self.n_estimators):
                # Add one stage
                self.model.set_params(n_estimators=i+1)
                self.model.fit(X, y, warm_start=True)
                
                # Evaluate
                train_pred = self.model.predict(X)
                val_pred = self.model.predict(X_val)
                
                if eval_metric == 'mse':
                    train_score = mean_squared_error(y, train_pred)
                    val_score = mean_squared_error(y_val, val_pred)
                elif eval_metric == 'mae':
                    train_score = mean_absolute_error(y, train_pred)
                    val_score = mean_absolute_error(y_val, val_pred)
                
                train_scores.append(train_score)
                val_scores.append(val_score)
                
                # Check for improvement
                if val_score < best_val_score:
                    best_val_score = val_score
                    best_iteration = i
                    best_model = clone(self.model)
                
                # Early stopping check
                if i - best_iteration >= self.early_stopping_rounds:
                    print(f"Early stopping at iteration {i}")
                    self.model = best_model
                    break
            
            self.train_score = train_scores
            self.val_score = val_scores
            self.best_iteration = best_iteration
        else:
            # Standard training
            self.model.fit(X, y)
        
        # Feature importance
        self.feature_importances_ = self.model.feature_importances_
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the gradient boosting ensemble.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def plot_training_curves(self, save_path=None):
        """
        Plot training and validation curves if early stopping was used.
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure or None
        """
        if self.train_score is None or self.val_score is None:
            print("No training curves available. Model was not trained with early stopping.")
            return None
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_score, label='Training')
        plt.plot(self.val_score, label='Validation')
        plt.axvline(x=self.best_iteration, color='r', linestyle='--', 
                   label=f'Best iteration: {self.best_iteration}')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title('Training and Validation Curves')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            
        return plt.gcf()
    
    def plot_feature_importance(self, top_n=10, save_path=None):
        """
        Plot feature importances.
        
        Args:
            top_n: Number of top features to display
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if self.feature_importances_ is None:
            raise ValueError("Model not fitted yet. Call fit() before plotting feature importance.")
        
        # Create a DataFrame for better visualization
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importances_
        })
        
        # Sort by importance and select top N
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return plt.gcf()


class StackingEnsemble(BaseEstimator, RegressorMixin):
    """
    Advanced stacking ensemble for cryptocurrency prediction.
    This implementation allows for time-aware cross-validation 
    and multiple meta-learners.
    """
    
    def __init__(self, base_models, meta_model, cv=5, use_features_in_secondary=False,
                 time_aware=True, refit=True):
        """
        Initialize Stacking Ensemble.
        
        Args:
            base_models: List of base models (first layer)
            meta_model: Meta model (second layer)
            cv: Number of cross-validation folds
            use_features_in_secondary: Whether to use original features in meta-model
            time_aware: Whether to use time-aware cross-validation
            refit: Whether to refit base models on the full dataset
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv = cv
        self.use_features_in_secondary = use_features_in_secondary
        self.time_aware = time_aware
        self.refit = refit
        
        self.base_models_ = None
        self.meta_model_ = None
        self.feature_names = None
    
    def fit(self, X, y, feature_names=None):
        """
        Fit the stacking ensemble.
        
        Args:
            X: Training features
            y: Target values
            feature_names: Names of features
            
        Returns:
            Fitted model
        """
        X, y = check_X_y(X, y, accept_sparse=True)
        
        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Clone base models
        self.base_models_ = [clone(model) for model in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        
        # Generate cross-validation folds
        if self.time_aware:
            # Time-aware cross-validation (blocks of data)
            n_samples = X.shape[0]
            fold_size = n_samples // self.cv
            cv_folds = []
            
            for i in range(self.cv):
                # For each fold, use data up to fold i for training and fold i+1 for validation
                if i < self.cv - 1:
                    train_idx = list(range(0, fold_size * (i + 1)))
                    val_idx = list(range(fold_size * (i + 1), fold_size * (i + 2)))
                else:
                    train_idx = list(range(0, fold_size * (i + 1)))
                    val_idx = list(range(fold_size * (i + 1), n_samples))
                
                cv_folds.append((train_idx, val_idx))
        else:
            # Standard K-fold cross-validation
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
            cv_folds = list(kf.split(X))
        
        # Create meta-features
        meta_features = np.zeros((X.shape[0], len(self.base_models_)))
        
        # Train base models and create meta-features
        for i, model in enumerate(self.base_models_):
            # Cross-validation to create meta-features
            for train_idx, val_idx in cv_folds:
                X_train, y_train = X[train_idx], y[train_idx]
                X_val = X[val_idx]
                
                # Train and predict
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                meta_features[val_idx, i] = model_clone.predict(X_val)
        
        # Train the meta-model
        if self.use_features_in_secondary:
            # Combine meta-features with original features
            self.meta_model_.fit(np.hstack((meta_features, X)), y)
        else:
            # Use only meta-features
            self.meta_model_.fit(meta_features, y)
        
        # Refit base models on the entire dataset if required
        if self.refit:
            for i, model in enumerate(self.base_models_):
                self.base_models_[i] = model.fit(X, y)
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the stacking ensemble.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        check_is_fitted(self, ['base_models_', 'meta_model_'])
        X = check_array(X, accept_sparse=True)
        
        # Generate meta-features
        meta_features = np.column_stack([
            model.predict(X) for model in self.base_models_
        ])
        
        # Make final predictions
        if self.use_features_in_secondary:
            return self.meta_model_.predict(np.hstack((meta_features, X)))
        else:
            return self.meta_model_.predict(meta_features)
    
    def get_feature_importance(self):
        """
        Get feature importance from base models.
        
        Returns:
            Dictionary of feature importances by model
        """
        importances = {}
        
        for i, model in enumerate(self.base_models_):
            if hasattr(model, 'feature_importances_'):
                importances[f'model_{i}'] = {
                    'model_name': model.__class__.__name__,
                    'importances': dict(zip(self.feature_names, model.feature_importances_))
                }
        
        return importances


class VotingEnsemble:
    """
    Enhanced voting ensemble for cryptocurrency prediction.
    Supports both hard and soft voting, with optional weighting of models
    based on their performance.
    """
    
    def __init__(self, models, weights=None, voting='soft', performance_weighting=False):
        """
        Initialize Voting Ensemble.
        
        Args:
            models: List of (name, model) tuples
            weights: List of weights for models (optional)
            voting: Type of voting ('hard' or 'soft')
            performance_weighting: Whether to update weights based on performance
        """
        self.models = models
        self.weights = weights
        self.voting = voting
        self.performance_weighting = performance_weighting
        
        # Create scikit-learn VotingRegressor
        self.model = VotingRegressor(
            estimators=models,
            weights=weights,
            n_jobs=-1
        )
        
        self.model_performances = None
    
    def fit(self, X, y, eval_set=None):
        """
        Fit the voting ensemble.
        
        Args:
            X: Training features
            y: Target values
            eval_set: Optional validation set for performance weighting
            
        Returns:
            Fitted model
        """
        # Fit the VotingRegressor
        self.model.fit(X, y)
        
        # If using performance weighting and a validation set is provided
        if self.performance_weighting and eval_set is not None:
            X_val, y_val = eval_set
            
            # Calculate performance for each model
            performances = []
            for name, model in self.models:
                y_pred = model.predict(X_val)
                score = mean_squared_error(y_val, y_pred)
                performances.append((name, score))
            
            # Store performances
            self.model_performances = performances
            
            # Update weights inversely proportional to error
            inverse_errors = [1.0 / (score + 1e-10) for _, score in performances]
            total = sum(inverse_errors)
            self.weights = [error / total for error in inverse_errors]
            
            # Update the VotingRegressor weights
            self.model.weights = self.weights
        
        return self
    
    def predict(self, X):
        """
        Make predictions with the voting ensemble.
        
        Args:
            X: Features for prediction
            
        Returns:
            Predictions
        """
        if self.voting == 'soft' and self.weights is not None:
            # Custom weighted prediction
            predictions = [model.predict(X) for _, model in self.models]
            weighted_predictions = np.zeros_like(predictions[0])
            
            for i, pred in enumerate(predictions):
                weighted_predictions += self.weights[i] * pred
                
            return weighted_predictions
        else:
            # Use scikit-learn implementation
            return self.model.predict(X)
    
    def print_model_weights(self):
        """
        Print the weights of each model in the ensemble.
        
        Returns:
            DataFrame of model weights and performances
        """
        if self.weights is None:
            print("No weights available. Models are weighted equally.")
            return None
        
        result = pd.DataFrame({
            'Model': [name for name, _ in self.models],
            'Weight': self.weights
        })
        
        if self.model_performances is not None:
            result['MSE'] = [score for _, score in self.model_performances]
        
        return result 
"""
Feature Selection Module for Cryptocurrency Trading

This module provides tools for feature importance evaluation and
feature selection to optimize machine learning models.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional
from sklearn.feature_selection import (
    SelectKBest, mutual_info_regression, f_regression,
    RFE, RFECV, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import shap
from boruta import BorutaPy


class FeatureSelector:
    """Class for feature selection and importance analysis for trading models."""
    
    def __init__(self):
        """Initialize the feature selector."""
        self.importance_scores = {}
        self.selected_features = []
        self.feature_masks = {}
    
    def filter_low_variance(self, df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """
        Remove features with variance below threshold.
        
        Args:
            df: DataFrame with features
            threshold: Variance threshold
            
        Returns:
            DataFrame with low-variance features removed
        """
        # Scale data before variance calculation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns)
        
        # Apply variance threshold
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X_scaled_df)
        
        # Get and store selected features
        self.feature_masks['variance'] = selector.get_support()
        selected_cols = df.columns[self.feature_masks['variance']]
        self.selected_features = selected_cols.tolist()
        
        return df[selected_cols]
    
    def remove_collinear_features(self, df: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
        """
        Remove highly correlated features.
        
        Args:
            df: DataFrame with features
            threshold: Correlation threshold for removal
            
        Returns:
            DataFrame with collinear features removed
        """
        # Calculate correlation matrix
        corr_matrix = df.corr().abs()
        
        # Extract upper triangle
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Store dropped features
        self.feature_masks['collinearity'] = ~df.columns.isin(to_drop)
        selected_cols = df.columns[self.feature_masks['collinearity']]
        self.selected_features = selected_cols.tolist()
        
        return df.drop(columns=to_drop)
    
    def select_k_best(self, X: pd.DataFrame, y: pd.Series, k: int = 10, 
                     method: str = 'mutual_info') -> pd.DataFrame:
        """
        Select k best features based on univariate statistical tests.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            k: Number of features to select
            method: Selection method ('mutual_info' or 'f_regression')
            
        Returns:
            DataFrame with k best features
        """
        # Choose scoring function
        if method == 'mutual_info':
            score_func = mutual_info_regression
        elif method == 'f_regression':
            score_func = f_regression
        else:
            raise ValueError("Method must be 'mutual_info' or 'f_regression'")
        
        # Apply selection
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        
        # Store results
        self.feature_masks['k_best'] = selector.get_support()
        self.importance_scores['k_best'] = {X.columns[i]: selector.scores_[i] 
                                          for i in range(len(X.columns))}
        
        # Update selected features
        selected_cols = X.columns[self.feature_masks['k_best']]
        self.selected_features = selected_cols.tolist()
        
        return X[selected_cols]
    
    def select_from_model(self, X: pd.DataFrame, y: pd.Series, 
                         method: str = 'random_forest', max_features: int = None,
                         threshold: str = 'mean') -> pd.DataFrame:
        """
        Select features using model-based importance.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            method: Model type ('random_forest', 'lasso')
            max_features: Maximum number of features to select
            threshold: Importance threshold strategy
            
        Returns:
            DataFrame with selected features
        """
        # Choose model
        if method == 'random_forest':
            # Check if classification or regression problem
            if len(np.unique(y)) < 5:  # Classification if fewer unique values
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif method == 'lasso':
            model = Lasso(alpha=0.01, random_state=42)
        else:
            raise ValueError("Method must be 'random_forest' or 'lasso'")
        
        # Apply selection
        selector = SelectFromModel(estimator=model, threshold=threshold, max_features=max_features)
        selector.fit(X, y)
        
        # Store results
        self.feature_masks['model_based'] = selector.get_support()
        
        if hasattr(selector.estimator_, 'feature_importances_'):
            self.importance_scores['model_based'] = {X.columns[i]: selector.estimator_.feature_importances_[i] 
                                                   for i in range(len(X.columns))}
        elif hasattr(selector.estimator_, 'coef_'):
            self.importance_scores['model_based'] = {X.columns[i]: abs(selector.estimator_.coef_[i]) 
                                                   for i in range(len(X.columns))}
        
        # Update selected features
        selected_cols = X.columns[self.feature_masks['model_based']]
        self.selected_features = selected_cols.tolist()
        
        return X[selected_cols]
    
    def recursive_feature_elimination(self, X: pd.DataFrame, y: pd.Series, 
                                     n_features: int = None, step: int = 1,
                                     use_cv: bool = True, cv: int = 5) -> pd.DataFrame:
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            n_features: Number of features to select
            step: Number of features to remove at each step
            use_cv: Whether to use cross-validation
            cv: Number of cross-validation folds
            
        Returns:
            DataFrame with selected features
        """
        # Initialize estimator (use random forest by default)
        if len(np.unique(y)) < 5:  # Classification
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # Regression
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Apply RFE with or without cross-validation
        if use_cv:
            selector = RFECV(estimator=estimator, step=step, cv=cv, n_jobs=-1)
        else:
            selector = RFE(estimator=estimator, n_features_to_select=n_features, step=step)
        
        selector.fit(X, y)
        
        # Store results
        self.feature_masks['rfe'] = selector.support_
        self.importance_scores['rfe'] = {X.columns[i]: selector.ranking_[i] 
                                       for i in range(len(X.columns))}
        
        # Update selected features
        selected_cols = X.columns[self.feature_masks['rfe']]
        self.selected_features = selected_cols.tolist()
        
        return X[selected_cols]
    
    def boruta_selection(self, X: pd.DataFrame, y: pd.Series, 
                        max_iter: int = 100, alpha: float = 0.05) -> pd.DataFrame:
        """
        Select features using Boruta algorithm.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            max_iter: Maximum number of iterations
            alpha: P-value threshold
            
        Returns:
            DataFrame with selected features
        """
        # Choose estimator based on target type
        if len(np.unique(y)) < 5:  # Classification
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # Regression
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Initialize Boruta
        boruta_selector = BorutaPy(
            estimator=estimator,
            n_estimators='auto',
            max_iter=max_iter,
            verbose=0,
            alpha=alpha,
            random_state=42
        )
        
        # Fit Boruta (requires numpy array)
        boruta_selector.fit(X.values, y.values)
        
        # Store results
        self.feature_masks['boruta'] = boruta_selector.support_
        self.importance_scores['boruta'] = {X.columns[i]: boruta_selector.ranking_[i] 
                                          for i in range(len(X.columns))}
        
        # Update selected features
        selected_cols = X.columns[boruta_selector.support_]
        self.selected_features = selected_cols.tolist()
        
        return X[selected_cols]
    
    def shap_based_selection(self, X: pd.DataFrame, y: pd.Series, 
                           n_features: int = 10, model=None) -> pd.DataFrame:
        """
        Select features based on SHAP values.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            n_features: Number of features to select
            model: Pre-trained model (default: RandomForest)
            
        Returns:
            DataFrame with selected features
        """
        # Create model if none provided
        if model is None:
            if len(np.unique(y)) < 5:  # Classification
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:  # Regression
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X, y)
        
        # Calculate SHAP values
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        
        # Get mean absolute SHAP values for each feature
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        
        # Store importance scores
        self.importance_scores['shap'] = {X.columns[i]: feature_importance[i] 
                                        for i in range(len(X.columns))}
        
        # Select top n features
        top_indices = np.argsort(feature_importance)[-n_features:]
        self.feature_masks['shap'] = np.zeros(len(X.columns), dtype=bool)
        self.feature_masks['shap'][top_indices] = True
        
        # Update selected features
        selected_cols = X.columns[self.feature_masks['shap']]
        self.selected_features = selected_cols.tolist()
        
        return X[selected_cols]
    
    def time_series_cv_selection(self, X: pd.DataFrame, y: pd.Series, 
                               model=None, n_features: int = 10,
                               n_splits: int = 5, test_size: float = 0.2) -> pd.DataFrame:
        """
        Select features using time series cross-validation performance.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            model: Model to use (default: RandomForest)
            n_features: Number of features to select
            n_splits: Number of time series splits
            test_size: Proportion of data in each test split
            
        Returns:
            DataFrame with selected features
        """
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import mean_squared_error, accuracy_score
        
        # Create model if none provided
        if model is None:
            if len(np.unique(y)) < 5:  # Classification
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                score_func = accuracy_score
            else:  # Regression
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                score_func = lambda y, y_pred: -mean_squared_error(y, y_pred)  # Negative MSE
        
        # Set up time series CV
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(len(X) * test_size))
        
        # Evaluate each feature's predictive power
        feature_scores = {}
        
        for feature in X.columns:
            scores = []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx][[feature]], X.iloc[test_idx][[feature]]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = score_func(y_test, y_pred)
                scores.append(score)
            
            feature_scores[feature] = np.mean(scores)
        
        # Store importance scores
        self.importance_scores['ts_cv'] = feature_scores
        
        # Select top n features
        top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:n_features]
        selected_cols = [f[0] for f in top_features]
        
        # Update mask and selected features
        self.feature_masks['ts_cv'] = np.isin(X.columns, selected_cols)
        self.selected_features = selected_cols
        
        return X[selected_cols]
    
    def plot_feature_importance(self, method: str = 'model_based', top_n: int = 20):
        """
        Plot feature importance for the selected method.
        
        Args:
            method: Method used for feature selection
            top_n: Number of top features to display
            
        Returns:
            Matplotlib figure
        """
        if method not in self.importance_scores:
            raise ValueError(f"Method '{method}' not found in importance scores.")
        
        # Get importance scores
        scores = self.importance_scores[method]
        
        # Sort and get top N features
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, values = zip(*sorted_scores)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), values, align='center')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title(f'Top {top_n} Features ({method})')
        plt.tight_layout()
        
        return plt.gcf()
    
    def stability_selection(self, X: pd.DataFrame, y: pd.Series, 
                          method: str = 'random_forest', n_features: int = 10, 
                          n_iterations: int = 50, sample_fraction: float = 0.75) -> pd.DataFrame:
        """
        Select features using stability selection.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            method: Selection method
            n_features: Number of features to select in each iteration
            n_iterations: Number of subsampling iterations
            sample_fraction: Fraction of samples to use in each iteration
            
        Returns:
            DataFrame with stable features
        """
        import random
        
        # Feature selection counts
        feature_counts = {col: 0 for col in X.columns}
        
        # Run iterations
        for i in range(n_iterations):
            # Subsample data
            sample_size = int(len(X) * sample_fraction)
            indices = random.sample(range(len(X)), sample_size)
            X_sample, y_sample = X.iloc[indices], y.iloc[indices]
            
            # Apply selection method
            if method == 'random_forest':
                sub_selector = self.__class__()  # Create new instance
                sub_selector.select_from_model(X_sample, y_sample, method='random_forest', max_features=n_features)
                selected = sub_selector.selected_features
            elif method == 'lasso':
                sub_selector = self.__class__()
                sub_selector.select_from_model(X_sample, y_sample, method='lasso', max_features=n_features)
                selected = sub_selector.selected_features
            elif method == 'mutual_info':
                sub_selector = self.__class__()
                sub_selector.select_k_best(X_sample, y_sample, k=n_features, method='mutual_info')
                selected = sub_selector.selected_features
            else:
                raise ValueError(f"Method '{method}' not supported for stability selection.")
            
            # Update counts
            for feature in selected:
                feature_counts[feature] += 1
        
        # Calculate selection probability
        selection_probability = {f: c / n_iterations for f, c in feature_counts.items()}
        self.importance_scores['stability'] = selection_probability
        
        # Select features that appear in at least 50% of iterations
        stable_features = [f for f, p in selection_probability.items() if p >= 0.5]
        self.feature_masks['stability'] = np.isin(X.columns, stable_features)
        self.selected_features = stable_features
        
        return X[stable_features]
    
    def sequential_feature_selection(self, X: pd.DataFrame, y: pd.Series, 
                                    n_features: int = 10, direction: str = 'forward',
                                    cv: int = 5) -> pd.DataFrame:
        """
        Select features using sequential feature selection.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            n_features: Number of features to select
            direction: Selection direction ('forward' or 'backward')
            cv: Number of cross-validation folds
            
        Returns:
            DataFrame with selected features
        """
        from sklearn.feature_selection import SequentialFeatureSelector
        
        # Initialize estimator
        if len(np.unique(y)) < 5:  # Classification
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # Regression
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Set up sequential feature selection
        sfs = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=n_features,
            direction=direction,
            scoring='neg_mean_squared_error' if len(np.unique(y)) >= 5 else 'accuracy',
            cv=cv,
            n_jobs=-1
        )
        
        # Fit selector
        sfs.fit(X, y)
        
        # Store results
        self.feature_masks['sequential'] = sfs.get_support()
        
        # Update selected features
        selected_cols = X.columns[sfs.get_support()]
        self.selected_features = selected_cols.tolist()
        
        return X[selected_cols]


# Utility functions

def rank_features_by_target_correlation(X: pd.DataFrame, y: pd.Series, 
                                       method: str = 'spearman') -> Dict[str, float]:
    """
    Rank features by correlation with target variable.
    
    Args:
        X: Feature DataFrame
        y: Target variable
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        
    Returns:
        Dictionary of features and correlation coefficients
    """
    correlations = {}
    
    for col in X.columns:
        if method == 'spearman':
            corr, _ = spearmanr(X[col], y)
        else:
            corr = X[col].corr(y, method=method)
        
        correlations[col] = abs(corr)  # Use absolute correlation
    
    # Sort by absolute correlation
    sorted_correlations = {k: v for k, v in sorted(correlations.items(), 
                                                  key=lambda item: item[1], 
                                                  reverse=True)}
    
    return sorted_correlations

def get_feature_importance_from_gbm(X: pd.DataFrame, y: pd.Series, 
                                   n_estimators: int = 100) -> Dict[str, float]:
    """
    Get feature importance from gradient boosting model.
    
    Args:
        X: Feature DataFrame
        y: Target variable
        n_estimators: Number of estimators
        
    Returns:
        Dictionary of features and importance scores
    """
    from lightgbm import LGBMRegressor, LGBMClassifier
    
    # Choose model type based on target
    if len(np.unique(y)) < 5:  # Classification
        model = LGBMClassifier(n_estimators=n_estimators, random_state=42)
    else:  # Regression
        model = LGBMRegressor(n_estimators=n_estimators, random_state=42)
    
    # Fit model
    model.fit(X, y)
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create dictionary
    importance_dict = {X.columns[i]: importance[i] for i in range(len(X.columns))}
    
    # Sort by importance
    sorted_importance = {k: v for k, v in sorted(importance_dict.items(), 
                                               key=lambda item: item[1], 
                                               reverse=True)}
    
    return sorted_importance 
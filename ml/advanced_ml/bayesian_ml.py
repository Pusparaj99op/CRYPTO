"""
Bayesian machine learning methods optimized for cryptocurrency data analysis.

This module provides Bayesian approaches to modeling financial time series data,
including Bayesian Neural Networks, Gaussian Processes, and Probabilistic models
that quantify uncertainty in predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pymc3 as pm
import arviz as az
from scipy import stats

tfd = tfp.distributions
tfb = tfp.bijectors


class BayesianLinearRegression:
    """Bayesian Linear Regression for time series forecasting with uncertainty quantification."""
    
    def __init__(self, n_iter=2000, tune=1000, target_accept=0.95, random_seed=42):
        """
        Initialize Bayesian Linear Regression model.
        
        Parameters:
        -----------
        n_iter : int
            Number of iterations for MCMC sampling
        tune : int
            Number of tuning steps for MCMC
        target_accept : float
            Target acceptance rate for MCMC
        random_seed : int
            Random seed for reproducibility
        """
        self.n_iter = n_iter
        self.tune = tune
        self.target_accept = target_accept
        self.random_seed = random_seed
        self.model = None
        self.trace = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def fit(self, X, y):
        """
        Fit Bayesian Linear Regression model.
        
        Parameters:
        -----------
        X : array-like
            Features matrix of shape (n_samples, n_features)
        y : array-like
            Target vector of shape (n_samples,)
            
        Returns:
        --------
        self : object
            Returns self
        """
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        n_features = X_scaled.shape[1]
        
        with pm.Model() as self.model:
            # Priors
            alpha = pm.HalfCauchy('alpha', beta=5)
            beta = pm.Normal('beta', mu=0, sd=1, shape=n_features)
            sigma = pm.HalfCauchy('sigma', beta=5)
            
            # Linear model
            mu = pm.math.dot(X_scaled, beta)
            
            # Likelihood
            likelihood = pm.Normal('likelihood', mu=mu, sd=sigma, observed=y_scaled)
            
            # Inference
            self.trace = pm.sample(
                draws=self.n_iter, 
                tune=self.tune, 
                target_accept=self.target_accept,
                random_seed=self.random_seed,
                return_inferencedata=True
            )
        
        return self
    
    def predict(self, X_new, return_std=False, credible_interval=0.95):
        """
        Make predictions with uncertainty estimates.
        
        Parameters:
        -----------
        X_new : array-like
            New features matrix of shape (n_samples, n_features)
        return_std : bool
            Whether to return standard deviation of predictions
        credible_interval : float
            Credible interval width (0 to 1)
            
        Returns:
        --------
        y_pred : array
            Mean predictions
        y_std : array, optional
            Standard deviation of predictions (if return_std=True)
        y_ci : tuple of arrays, optional
            Lower and upper bounds of credible interval (if return_std=True)
        """
        X_new_scaled = self.scaler_X.transform(X_new)
        
        beta_samples = self.trace.posterior['beta'].values.reshape(-1, X_new.shape[1])
        
        # Generate predictions
        y_pred_samples = np.dot(X_new_scaled, beta_samples.T)
        
        # Convert back to original scale
        y_pred_samples = self.scaler_y.inverse_transform(y_pred_samples)
        
        # Mean prediction
        y_pred_mean = np.mean(y_pred_samples, axis=1)
        
        if return_std:
            y_pred_std = np.std(y_pred_samples, axis=1)
            lower_ci = np.percentile(y_pred_samples, (1 - credible_interval) / 2 * 100, axis=1)
            upper_ci = np.percentile(y_pred_samples, (1 + credible_interval) / 2 * 100, axis=1)
            return y_pred_mean, y_pred_std, (lower_ci, upper_ci)
        
        return y_pred_mean
    
    def plot_posterior(self, var_names=None, figsize=(12, 8)):
        """
        Plot posterior distributions of parameters.
        
        Parameters:
        -----------
        var_names : list or None
            Variables to plot
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Figure with posterior plots
        """
        if self.trace is None:
            raise ValueError("Model must be fitted before plotting posterior.")
        
        return az.plot_posterior(self.trace, var_names=var_names, figsize=figsize)
    
    def plot_trace(self, var_names=None, figsize=(12, 8)):
        """
        Plot MCMC traces.
        
        Parameters:
        -----------
        var_names : list or None
            Variables to plot
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Figure with trace plots
        """
        if self.trace is None:
            raise ValueError("Model must be fitted before plotting trace.")
        
        return az.plot_trace(self.trace, var_names=var_names, figsize=figsize)
        

class BayesianNeuralNetwork:
    """Bayesian Neural Network for cryptocurrency price prediction with uncertainty quantification."""
    
    def __init__(self, 
                 hidden_units=[64, 32], 
                 activation='relu',
                 learning_rate=0.01,
                 kl_weight=0.1,
                 random_seed=42):
        """
        Initialize Bayesian Neural Network.
        
        Parameters:
        -----------
        hidden_units : list
            Number of units in each hidden layer
        activation : str
            Activation function
        learning_rate : float
            Learning rate for optimization
        kl_weight : float
            Weight for KL divergence loss term
        random_seed : int
            Random seed for reproducibility
        """
        self.hidden_units = hidden_units
        self.activation = activation
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.random_seed = random_seed
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def _build_model(self, input_dim):
        """Build the Bayesian Neural Network model."""
        tf.random.set_seed(self.random_seed)
        
        # Define the prior weight distribution as Normal(0, 1)
        def prior(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            prior_model = keras_model = tf.keras.Sequential([
                tfp.layers.DistributionLambda(
                    lambda t: tfd.Independent(
                        tfd.Normal(loc=tf.zeros(n), scale=1),
                        reinterpreted_batch_ndims=1))
            ])
            return prior_model
        
        # Define variational posterior weight distribution as normal with trainable loc and scale
        def posterior(kernel_size, bias_size, dtype=None):
            n = kernel_size + bias_size
            posterior_model = tf.keras.Sequential([
                tfp.layers.VariableLayer(
                    tfp.layers.IndependentNormal.params_size(n), dtype=dtype),
                tfp.layers.IndependentNormal(n)
            ])
            return posterior_model
        
        # Build the model
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(input_dim,)))
        
        # Add DenseVariational layers
        for units in self.hidden_units:
            model.add(
                tfp.layers.DenseVariational(
                    units=units,
                    make_prior_fn=prior,
                    make_posterior_fn=posterior,
                    activation=self.activation,
                    kl_weight=self.kl_weight / tf.cast(1, dtype=tf.float32)
                )
            )
        
        # Add output layer
        model.add(
            tfp.layers.DenseVariational(
                units=1,
                make_prior_fn=prior,
                make_posterior_fn=posterior,
                kl_weight=self.kl_weight / tf.cast(1, dtype=tf.float32)
            )
        )
        
        # Add distribution lambda layer for heteroscedastic noise modeling
        model.add(
            tfp.layers.DistributionLambda(
                lambda t: tfd.Normal(loc=t[..., :1],
                                    scale=1e-3 + tf.math.softplus(0.01 * t[..., :1]))
            )
        )
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=lambda y, p_y: -p_y.log_prob(y)
        )
        
        return model
    
    def fit(self, X, y, validation_data=None, epochs=200, batch_size=64, verbose=1):
        """
        Fit Bayesian Neural Network model.
        
        Parameters:
        -----------
        X : array-like
            Features matrix of shape (n_samples, n_features)
        y : array-like
            Target vector of shape (n_samples,)
        validation_data : tuple or None
            Tuple (X_val, y_val) for validation
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        verbose : int
            Verbosity mode
            
        Returns:
        --------
        history : dict
            Training history
        """
        # Scale the data
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Create validation data if provided
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            validation_data = (X_val_scaled, y_val_scaled)
        
        # Build the model
        self.model = self._build_model(X_scaled.shape[1])
        
        # Train the model
        history = self.model.fit(
            X_scaled, y_scaled,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X_new, n_samples=100):
        """
        Make predictions with uncertainty estimates.
        
        Parameters:
        -----------
        X_new : array-like
            New features matrix of shape (n_samples, n_features)
        n_samples : int
            Number of samples to draw from posterior
            
        Returns:
        --------
        y_pred_mean : array
            Mean predictions
        y_pred_std : array
            Standard deviation of predictions
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction.")
        
        X_new_scaled = self.scaler_X.transform(X_new)
        
        # Draw multiple samples from posterior
        yhats = [self.model(X_new_scaled) for _ in range(n_samples)]
        yhats = [self.scaler_y.inverse_transform(m.mean().numpy()) for m in yhats]
        yhats = np.stack(yhats)
        
        # Calculate mean and std
        y_pred_mean = np.mean(yhats, axis=0).flatten()
        y_pred_std = np.std(yhats, axis=0).flatten()
        
        return y_pred_mean, y_pred_std
    
    def plot_predictions(self, X, y, n_samples=100, figsize=(12, 6)):
        """
        Plot predictions with uncertainty estimates.
        
        Parameters:
        -----------
        X : array-like
            Features matrix
        y : array-like
            True target values
        n_samples : int
            Number of samples to draw from posterior
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Figure with prediction plot
        """
        mean_pred, std_pred = self.predict(X, n_samples=n_samples)
        
        plt.figure(figsize=figsize)
        plt.plot(y, 'k-', label='True Values')
        plt.plot(mean_pred, 'b-', label='Predictions')
        plt.fill_between(
            np.arange(len(mean_pred)),
            mean_pred - 2 * std_pred,
            mean_pred + 2 * std_pred,
            color='b',
            alpha=0.2,
            label='Uncertainty (2 std)'
        )
        plt.legend()
        plt.title('BNN Predictions with Uncertainty')
        plt.tight_layout()
        
        return plt.gcf()


class GaussianProcessRegressor:
    """Gaussian Process Regression for cryptocurrency time series with custom kernels."""
    
    def __init__(self, kernel=None, alpha=1e-6, n_restarts_optimizer=10, random_seed=42):
        """
        Initialize Gaussian Process Regressor with custom financial kernels.
        
        Parameters:
        -----------
        kernel : sklearn.gaussian_process.kernels or None
            Kernel function. If None, a combination of RBF and Periodic kernels is used
        alpha : float
            Noise level
        n_restarts_optimizer : int
            Number of restarts for optimizer
        random_seed : int
            Random seed for reproducibility
        """
        from sklearn.gaussian_process import GaussianProcessRegressor as GPR
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel, ExpSineSquared
        
        if kernel is None:
            # Combine multiple kernels for financial time series
            # Long-term trend (RBF)
            k1 = ConstantKernel(1.0) * RBF(length_scale=10.0)
            # Short-term irregularities (Matern)
            k2 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5)
            # Periodic patterns (daily, weekly) - common in crypto
            k3 = ConstantKernel(1.0) * ExpSineSquared(length_scale=1.0, periodicity=1.0)
            # Noise
            k4 = WhiteKernel(noise_level=alpha)
            
            kernel = k1 + k2 + k3 + k4
        
        self.model = GPR(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_seed
        )
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def fit(self, X, y):
        """
        Fit Gaussian Process model.
        
        Parameters:
        -----------
        X : array-like
            Features matrix of shape (n_samples, n_features)
        y : array-like
            Target vector of shape (n_samples,)
            
        Returns:
        --------
        self : object
            Returns self
        """
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        self.model.fit(X_scaled, y_scaled)
        
        return self
    
    def predict(self, X_new, return_std=True):
        """
        Make predictions with uncertainty estimates.
        
        Parameters:
        -----------
        X_new : array-like
            New features matrix of shape (n_samples, n_features)
        return_std : bool
            Whether to return standard deviation
            
        Returns:
        --------
        y_pred : array
            Mean predictions
        y_std : array, optional
            Standard deviation of predictions (if return_std=True)
        """
        X_new_scaled = self.scaler_X.transform(X_new)
        
        if return_std:
            y_pred_scaled, y_std_scaled = self.model.predict(X_new_scaled, return_std=True)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            # Scale standard deviation appropriately
            y_std = y_std_scaled * self.scaler_y.scale_[0]
            return y_pred, y_std
        else:
            y_pred_scaled = self.model.predict(X_new_scaled, return_std=False)
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            return y_pred
    
    def sample_posterior(self, X_new, n_samples=100):
        """
        Sample from posterior distribution at given points.
        
        Parameters:
        -----------
        X_new : array-like
            New features matrix of shape (n_samples, n_features)
        n_samples : int
            Number of samples to draw
            
        Returns:
        --------
        samples : array
            Samples from posterior, shape (n_samples, n_points)
        """
        X_new_scaled = self.scaler_X.transform(X_new)
        
        y_samples_scaled = self.model.sample_y(X_new_scaled, n_samples=n_samples)
        # Transform each sample back to original scale
        y_samples = np.zeros_like(y_samples_scaled)
        for i in range(n_samples):
            y_samples[i] = self.scaler_y.inverse_transform(y_samples_scaled[i].reshape(-1, 1)).flatten()
        
        return y_samples
    
    def plot_prediction(self, X_test, y_test, X_train=None, y_train=None, figsize=(12, 6)):
        """
        Plot predictions with confidence intervals.
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            True test values
        X_train : array-like or None
            Training features for plotting
        y_train : array-like or None
            Training targets for plotting
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Figure with prediction plot
        """
        # Get predictions and std
        y_pred, y_std = self.predict(X_test)
        
        # Plot
        plt.figure(figsize=figsize)
        
        # Plot training data if provided
        if X_train is not None and y_train is not None:
            plt.scatter(range(len(y_train)), y_train, c='k', alpha=0.2, label='Training Data')
        
        # Plot test data
        plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_test, 'k-', label='True Values')
        
        # Plot predictions
        x_pred = range(len(y_train), len(y_train) + len(y_test))
        plt.plot(x_pred, y_pred, 'b-', label='GP Prediction')
        plt.fill_between(
            x_pred,
            y_pred - 2 * y_std,
            y_pred + 2 * y_std,
            color='b',
            alpha=0.2,
            label='95% Confidence Interval'
        )
        
        plt.legend()
        plt.title('Gaussian Process Regression Prediction')
        plt.tight_layout()
        
        return plt.gcf()


class BayesianTimeSeriesForecaster:
    """Bayesian structural time series forecasting for crypto assets."""
    
    def __init__(self, n_iter=2000, tune=1000, random_seed=42):
        """
        Initialize Bayesian Time Series Forecaster.
        
        Parameters:
        -----------
        n_iter : int
            Number of iterations for MCMC sampling
        tune : int
            Number of tuning steps for MCMC
        random_seed : int
            Random seed for reproducibility
        """
        self.n_iter = n_iter
        self.tune = tune
        self.random_seed = random_seed
        self.model = None
        self.trace = None
        self.scaler_y = StandardScaler()
        
    def fit(self, y, X=None, seasonality_period=7, trend_flexibility=0.1):
        """
        Fit Bayesian structural time series model.
        
        Parameters:
        -----------
        y : array-like
            Time series data of shape (n_samples,)
        X : array-like or None
            Exogenous features of shape (n_samples, n_features)
        seasonality_period : int
            Period of seasonality (e.g., 7 for weekly, 24 for daily)
        trend_flexibility : float
            Flexibility of trend component (higher = more flexible)
            
        Returns:
        --------
        self : object
            Returns self
        """
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        with pm.Model() as self.model:
            # Level and trend components (local linear trend model)
            level_sigma = pm.HalfCauchy('level_sigma', beta=trend_flexibility)
            trend_sigma = pm.HalfCauchy('trend_sigma', beta=trend_flexibility)
            
            # Initial level and trend
            level_init = pm.Normal('level_init', mu=0, sigma=1)
            trend_init = pm.Normal('trend_init', mu=0, sigma=0.1)
            
            # Seasonality components if requested
            if seasonality_period > 1:
                season_sigma = pm.HalfCauchy('season_sigma', beta=0.1)
                season_init = pm.Normal('season_init', mu=0, sigma=0.1, 
                                       shape=seasonality_period-1)
            
            # Observation noise
            sigma = pm.HalfCauchy('sigma', beta=1)
            
            # Regression component for exogenous features
            if X is not None:
                X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)
                beta = pm.Normal('beta', mu=0, sigma=1, shape=X.shape[1])
                regression = pm.math.dot(X_scaled, beta)
            else:
                regression = 0
            
            # Define the model components: level, trend, seasonality
            level = pm.GaussianRandomWalk('level', 
                                         mu=level_init + trend_init,
                                         sigma=level_sigma,
                                         shape=len(y_scaled))
            
            trend = pm.GaussianRandomWalk('trend',
                                         mu=trend_init,
                                         sigma=trend_sigma,
                                         shape=len(y_scaled))
            
            # Add seasonality if requested
            if seasonality_period > 1:
                season_part = pm.Normal('season_part', mu=0, sigma=season_sigma,
                                      shape=(len(y_scaled), seasonality_period-1))
                
                # Ensure sum-to-zero constraint for seasonality
                season = pm.Deterministic('season', 
                                         tt.concatenate([
                                            season_part,
                                            -tt.sum(season_part, axis=1).dimshuffle(0, 'x')
                                         ], axis=1))
                
                # Extract the appropriate seasonal component for each time point
                t = np.arange(len(y_scaled)) % seasonality_period
                seasonal_effect = season[np.arange(len(y_scaled)), t]
            else:
                seasonal_effect = 0
            
            # Combine components
            mu = level + trend + seasonal_effect + regression
            
            # Likelihood
            pm.Normal('y', mu=mu, sigma=sigma, observed=y_scaled)
            
            # Inference
            self.trace = pm.sample(
                draws=self.n_iter,
                tune=self.tune,
                random_seed=self.random_seed,
                return_inferencedata=True
            )
        
        return self
    
    def predict(self, steps=30, X_future=None, credible_interval=0.95):
        """
        Forecast future values with uncertainty.
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast
        X_future : array-like or None
            Future exogenous features of shape (steps, n_features)
        credible_interval : float
            Credible interval width (0 to 1)
            
        Returns:
        --------
        forecast : array
            Mean forecast
        forecast_ci : tuple of arrays
            Lower and upper bounds of credible interval
        components : dict
            Dictionary of forecast components (trend, seasonality, etc.)
        """
        if self.trace is None:
            raise ValueError("Model must be fitted before forecasting.")
        
        # Extract posterior samples
        level_samples = self.trace.posterior['level'].values
        trend_samples = self.trace.posterior['trend'].values
        
        # Get the last values
        last_level = level_samples[:, :, -1]
        last_trend = trend_samples[:, :, -1]
        
        # Prepare arrays for forecasts
        n_chains = level_samples.shape[0]
        n_samples = level_samples.shape[1]
        forecast_samples = np.zeros((n_chains, n_samples, steps))
        
        # Check if model has seasonality
        has_seasonality = 'season' in self.trace.posterior
        if has_seasonality:
            season_samples = self.trace.posterior['season'].values
            seasonality_period = season_samples.shape[-1]
        
        # Check if model has regression component
        has_regression = 'beta' in self.trace.posterior
        if has_regression and X_future is None:
            raise ValueError("Model was trained with exogenous features but X_future not provided.")
        
        if has_regression:
            beta_samples = self.trace.posterior['beta'].values
            X_future_scaled = (X_future - X_future.mean(axis=0)) / X_future.std(axis=0)
        
        # Generate forecasts
        for c in range(n_chains):
            for s in range(n_samples):
                level = last_level[c, s]
                trend = last_trend[c, s]
                
                for i in range(steps):
                    # Update level and trend
                    level = level + trend
                    
                    # Add seasonality if present
                    seasonal_component = 0
                    if has_seasonality:
                        t = (len(level_samples[0, 0]) + i) % seasonality_period
                        seasonal_component = season_samples[c, s, -seasonality_period + t]
                    
                    # Add regression component if present
                    regression_component = 0
                    if has_regression:
                        regression_component = np.dot(X_future_scaled[i], beta_samples[c, s])
                    
                    # Store forecast
                    forecast_samples[c, s, i] = level + seasonal_component + regression_component
        
        # Flatten chain and sample dimensions
        forecast_samples_flat = forecast_samples.reshape(-1, steps)
        
        # Calculate statistics
        forecast_mean = np.mean(forecast_samples_flat, axis=0)
        forecast_lower = np.percentile(forecast_samples_flat, 
                                     (1 - credible_interval) / 2 * 100, 
                                     axis=0)
        forecast_upper = np.percentile(forecast_samples_flat, 
                                     (1 + credible_interval) / 2 * 100, 
                                     axis=0)
        
        # Transform back to original scale
        forecast_mean = self.scaler_y.inverse_transform(
            forecast_mean.reshape(-1, 1)).flatten()
        forecast_lower = self.scaler_y.inverse_transform(
            forecast_lower.reshape(-1, 1)).flatten()
        forecast_upper = self.scaler_y.inverse_transform(
            forecast_upper.reshape(-1, 1)).flatten()
        
        # Prepare component dictionary
        components = {
            'level': self.scaler_y.inverse_transform(level.reshape(-1, 1)).flatten(),
            'trend': trend
        }
        
        if has_seasonality:
            components['seasonality'] = seasonal_component
        
        if has_regression:
            components['regression'] = regression_component
        
        return forecast_mean, (forecast_lower, forecast_upper), components
    
    def plot_forecast(self, y_history, forecast, forecast_ci, figsize=(14, 7)):
        """
        Plot the forecast with historical data.
        
        Parameters:
        -----------
        y_history : array-like
            Historical time series data
        forecast : array-like
            Mean forecast
        forecast_ci : tuple of arrays
            Lower and upper bounds of credible interval
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : matplotlib figure
            Figure with forecast plot
        """
        lower_ci, upper_ci = forecast_ci
        
        plt.figure(figsize=figsize)
        
        # Plot historical data
        plt.plot(np.arange(len(y_history)), y_history, 'k-', label='Historical Data')
        
        # Plot forecast
        forecast_idx = np.arange(len(y_history), len(y_history) + len(forecast))
        plt.plot(forecast_idx, forecast, 'b-', label='Forecast')
        plt.fill_between(
            forecast_idx,
            lower_ci,
            upper_ci,
            color='b',
            alpha=0.2,
            label='95% Credible Interval'
        )
        
        # Add vertical line to separate history from forecast
        plt.axvline(x=len(y_history)-1, color='r', linestyle='--', alpha=0.5)
        
        plt.legend()
        plt.title('Bayesian Time Series Forecast')
        plt.tight_layout()
        
        return plt.gcf() 
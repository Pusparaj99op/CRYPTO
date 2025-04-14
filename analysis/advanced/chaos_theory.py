import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Union, Optional, Any
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.neighbors import KDTree
import warnings


class ChaosTheory:
    """
    Implementation of chaos theory concepts for cryptocurrency market analysis.
    Includes Lyapunov exponents, phase space reconstruction, recurrence plots,
    and predictability measures.
    """
    
    def __init__(self):
        """Initialize the ChaosTheory class."""
        pass
    
    def phase_space_reconstruction(self, time_series: np.ndarray, 
                                  embedding_dimension: int, 
                                  time_delay: int) -> np.ndarray:
        """
        Reconstruct the phase space from a time series using delay embedding.
        
        Args:
            time_series (np.ndarray): Input time series
            embedding_dimension (int): Embedding dimension
            time_delay (int): Time delay
            
        Returns:
            np.ndarray: Reconstructed phase space with shape (n_points, embedding_dimension)
        """
        n = len(time_series)
        
        # Check if parameters are valid
        if embedding_dimension * time_delay > n:
            raise ValueError("Embedding dimension and time delay are too large for the time series")
        
        # Reconstruct phase space
        n_points = n - (embedding_dimension - 1) * time_delay
        phase_space = np.zeros((n_points, embedding_dimension))
        
        for i in range(n_points):
            for j in range(embedding_dimension):
                phase_space[i, j] = time_series[i + j * time_delay]
        
        return phase_space
    
    def estimate_time_delay(self, time_series: np.ndarray, 
                           max_delay: int = 50, 
                           method: str = 'autocorr') -> int:
        """
        Estimate the optimal time delay for phase space reconstruction.
        
        Args:
            time_series (np.ndarray): Input time series
            max_delay (int): Maximum delay to consider
            method (str): Method to use ('autocorr' or 'mutual_info')
            
        Returns:
            int: Estimated optimal time delay
        """
        # Ensure the time series is a numpy array
        time_series = np.asarray(time_series)
        
        if method == 'autocorr':
            # Calculate autocorrelation function
            n = len(time_series)
            mean = np.mean(time_series)
            var = np.var(time_series)
            
            if var == 0:
                return 1  # Default value for constant series
            
            # Calculate normalized autocorrelation
            autocorr = np.zeros(max_delay)
            for lag in range(max_delay):
                c = np.sum((time_series[:(n-lag)] - mean) * (time_series[lag:] - mean)) / ((n - lag) * var)
                autocorr[lag] = c
            
            # Find first zero crossing or 1/e point
            for i in range(1, max_delay):
                if autocorr[i] <= 0:
                    return i  # First zero crossing
                if autocorr[i] <= 1/np.e:
                    return i  # 1/e point
            
            # If no zero crossing or 1/e point, return max_delay/4
            return max(1, max_delay // 4)
            
        elif method == 'mutual_info':
            # Calculate mutual information
            mi = np.zeros(max_delay)
            for delay in range(1, max_delay):
                x1 = time_series[:-delay].reshape(-1, 1)
                x2 = time_series[delay:].reshape(-1, 1)
                
                # Calculate mutual information using sklearn
                mi[delay] = mutual_info_score(
                    np.digitize(x1, np.linspace(min(x1), max(x1), 10)).ravel(),
                    np.digitize(x2, np.linspace(min(x2), max(x2), 10)).ravel()
                )
            
            # Find first minimum
            for i in range(1, max_delay-1):
                if mi[i-1] > mi[i] and mi[i] < mi[i+1]:
                    return i
            
            # If no minimum, return max_delay/4
            return max(1, max_delay // 4)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def estimate_embedding_dimension(self, time_series: np.ndarray, 
                                    time_delay: int, 
                                    max_dim: int = 10,
                                    method: str = 'fnn') -> int:
        """
        Estimate the optimal embedding dimension using the false nearest neighbor method.
        
        Args:
            time_series (np.ndarray): Input time series
            time_delay (int): Time delay
            max_dim (int): Maximum embedding dimension to consider
            method (str): Method to use ('fnn' for false nearest neighbors)
            
        Returns:
            int: Estimated optimal embedding dimension
        """
        # Ensure the time series is a numpy array
        time_series = np.asarray(time_series)
        
        if method == 'fnn':
            # Calculate percentage of false nearest neighbors for each dimension
            fnn_percentage = np.zeros(max_dim - 1)
            
            # Threshold for considering a neighbor as false
            threshold = 10.0
            
            for dim in range(1, max_dim):
                # Reconstruct phase space for current and next dimension
                ps_current = self.phase_space_reconstruction(time_series, dim, time_delay)
                ps_next = self.phase_space_reconstruction(time_series, dim + 1, time_delay)
                
                # Find nearest neighbors in current dimension
                tree = KDTree(ps_current)
                n_points = ps_current.shape[0]
                
                # Count false nearest neighbors
                false_neighbors = 0
                
                for i in range(n_points):
                    # Find nearest neighbor (excluding the point itself)
                    dist, ind = tree.query([ps_current[i]], k=2)
                    neighbor_idx = ind[0][1]
                    
                    # Calculate distances in the next dimension
                    if i < ps_next.shape[0] and neighbor_idx < ps_next.shape[0]:
                        # Calculate the additional distance in the next dimension
                        dist_current = np.linalg.norm(ps_current[i] - ps_current[neighbor_idx])
                        dist_next = np.linalg.norm(ps_next[i] - ps_next[neighbor_idx])
                        
                        # Check if it's a false neighbor
                        if dist_current > 0 and (dist_next / dist_current) > threshold:
                            false_neighbors += 1
                
                # Calculate percentage of false neighbors
                fnn_percentage[dim - 1] = false_neighbors / n_points if n_points > 0 else 0
            
            # Find the dimension where the percentage drops below a threshold
            for dim in range(1, max_dim - 1):
                if fnn_percentage[dim - 1] < 0.01:  # Less than 1% false neighbors
                    return dim
            
            # If no clear drop, return the dimension with minimum percentage
            return np.argmin(fnn_percentage) + 1
            
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def lyapunov_exponent(self, time_series: np.ndarray, 
                         embedding_dim: int = None, 
                         time_delay: int = None,
                         max_iter: int = 50) -> float:
        """
        Calculate the maximal Lyapunov exponent of a time series.
        
        The Lyapunov exponent measures the rate of separation of infinitesimally close trajectories.
        Positive values indicate chaos, while negative values indicate stability.
        
        Args:
            time_series (np.ndarray): Input time series
            embedding_dim (int, optional): Embedding dimension
            time_delay (int, optional): Time delay
            max_iter (int): Maximum number of iterations
            
        Returns:
            float: Maximal Lyapunov exponent
        """
        # Ensure the time series is a numpy array
        time_series = np.asarray(time_series)
        
        # Automatically determine embedding parameters if not provided
        if time_delay is None:
            time_delay = self.estimate_time_delay(time_series)
        
        if embedding_dim is None:
            embedding_dim = self.estimate_embedding_dimension(time_series, time_delay)
        
        # Reconstruct phase space
        phase_space = self.phase_space_reconstruction(time_series, embedding_dim, time_delay)
        
        # Number of points in the phase space
        n_points = phase_space.shape[0]
        
        # Calculate the maximal Lyapunov exponent using the algorithm by Rosenstein et al.
        # Build KD-tree for nearest neighbor search
        tree = KDTree(phase_space)
        
        # Minimum sequence separation to avoid temporal correlations
        min_sep = int(round(time_delay / 2)) if time_delay > 0 else 1
        
        # Calculate average divergence at each time step
        sum_divergence = np.zeros(max_iter)
        count = np.zeros(max_iter)
        
        for i in range(n_points):
            # Find the nearest neighbor with sufficient temporal separation
            dists, inds = tree.query([phase_space[i]], k=n_points)
            dists, inds = dists[0], inds[0]
            
            # Find the nearest neighbor with temporal separation > min_sep
            j = None
            for k in range(1, n_points):
                if k < len(inds) and abs(inds[k] - i) > min_sep:
                    j = inds[k]
                    break
            
            if j is None:
                continue
            
            # Calculate divergence over time
            for k in range(max_iter):
                i_future = i + k
                j_future = j + k
                
                if i_future < n_points and j_future < n_points:
                    # Calculate Euclidean distance between points
                    dist = np.linalg.norm(phase_space[i_future] - phase_space[j_future])
                    
                    if dist > 0:
                        sum_divergence[k] += np.log(dist)
                        count[k] += 1
        
        # Calculate average divergence
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            avg_divergence = np.divide(sum_divergence, count, where=count > 0)
        
        # Fit a line to the average divergence
        valid = count > 0
        if np.sum(valid) > 1:
            x = np.arange(max_iter)[valid]
            y = avg_divergence[valid]
            
            # Linear regression to find the slope
            slope, _, _, _, _ = stats.linregress(x, y)
            
            return slope
        else:
            return np.nan
    
    def recurrence_plot(self, time_series: np.ndarray, 
                       embedding_dim: int = None, 
                       time_delay: int = None,
                       threshold: float = None,
                       distance_norm: str = 'euclidean') -> np.ndarray:
        """
        Generate a recurrence plot for a time series.
        
        Args:
            time_series (np.ndarray): Input time series
            embedding_dim (int, optional): Embedding dimension
            time_delay (int, optional): Time delay
            threshold (float, optional): Distance threshold
            distance_norm (str): Distance norm ('euclidean', 'manhattan', 'max')
            
        Returns:
            np.ndarray: Recurrence matrix (binary)
        """
        # Ensure the time series is a numpy array
        time_series = np.asarray(time_series)
        
        # Automatically determine embedding parameters if not provided
        if time_delay is None:
            time_delay = self.estimate_time_delay(time_series)
        
        if embedding_dim is None:
            embedding_dim = self.estimate_embedding_dimension(time_series, time_delay)
        
        # Reconstruct phase space
        phase_space = self.phase_space_reconstruction(time_series, embedding_dim, time_delay)
        
        # Number of points in the phase space
        n_points = phase_space.shape[0]
        
        # Create distance matrix
        if distance_norm == 'euclidean':
            # Euclidean distance
            distances = np.zeros((n_points, n_points))
            for i in range(n_points):
                for j in range(i, n_points):
                    dist = np.linalg.norm(phase_space[i] - phase_space[j])
                    distances[i, j] = distances[j, i] = dist
        elif distance_norm == 'manhattan':
            # Manhattan distance
            distances = np.zeros((n_points, n_points))
            for i in range(n_points):
                for j in range(i, n_points):
                    dist = np.sum(np.abs(phase_space[i] - phase_space[j]))
                    distances[i, j] = distances[j, i] = dist
        elif distance_norm == 'max':
            # Maximum distance
            distances = np.zeros((n_points, n_points))
            for i in range(n_points):
                for j in range(i, n_points):
                    dist = np.max(np.abs(phase_space[i] - phase_space[j]))
                    distances[i, j] = distances[j, i] = dist
        else:
            raise ValueError(f"Unknown distance norm: {distance_norm}")
        
        # Determine threshold if not provided
        if threshold is None:
            # Use a percentage of the maximum distance
            threshold = 0.1 * np.max(distances)
        
        # Create recurrence matrix
        recurrence = distances <= threshold
        
        return recurrence
    
    def recurrence_quantification(self, recurrence_matrix: np.ndarray) -> Dict[str, float]:
        """
        Calculate recurrence quantification analysis measures.
        
        Args:
            recurrence_matrix (np.ndarray): Recurrence matrix
            
        Returns:
            Dict: Dictionary with RQA measures
        """
        # Total number of points
        n_points = recurrence_matrix.shape[0]
        
        # Recurrence rate (RR)
        rr = np.sum(recurrence_matrix) / (n_points ** 2)
        
        # Calculate diagonal line measures
        diag_lines = []
        for i in range(-(n_points-1), n_points):
            diag = np.diag(recurrence_matrix, i)
            # Count consecutive ones
            line_lengths = self._count_lines(diag)
            if line_lengths:
                diag_lines.extend(line_lengths)
        
        # Filter out lines of length 1
        diag_lines = [line for line in diag_lines if line > 1]
        
        # Calculate vertical line measures
        vert_lines = []
        for i in range(n_points):
            col = recurrence_matrix[:, i]
            # Count consecutive ones
            line_lengths = self._count_lines(col)
            if line_lengths:
                vert_lines.extend(line_lengths)
        
        # Filter out lines of length 1
        vert_lines = [line for line in vert_lines if line > 1]
        
        # Calculate RQA measures
        results = {'RR': rr}
        
        # Determinism (DET) - percentage of recurrence points forming diagonal lines
        if diag_lines:
            results['DET'] = np.sum(diag_lines) / np.sum(recurrence_matrix)
            results['Lmax'] = np.max(diag_lines)  # Longest diagonal line
            results['Lmean'] = np.mean(diag_lines)  # Average diagonal line length
            
            # Divergence - inverse of longest diagonal line
            results['DIV'] = 1.0 / results['Lmax']
            
            # Entropy of diagonal line lengths
            hist, _ = np.histogram(diag_lines, bins=range(1, max(diag_lines) + 2))
            probs = hist / np.sum(hist)
            results['ENTR'] = -np.sum(probs * np.log(probs + 1e-10))
        else:
            results['DET'] = 0.0
            results['Lmax'] = 0.0
            results['Lmean'] = 0.0
            results['DIV'] = float('inf')
            results['ENTR'] = 0.0
        
        # Laminarity (LAM) - percentage of recurrence points forming vertical lines
        if vert_lines:
            results['LAM'] = np.sum(vert_lines) / np.sum(recurrence_matrix)
            results['Vmax'] = np.max(vert_lines)  # Longest vertical line
            results['Vmean'] = np.mean(vert_lines)  # Average vertical line length
            
            # Trapping time (TT) - average length of vertical lines
            results['TT'] = results['Vmean']
        else:
            results['LAM'] = 0.0
            results['Vmax'] = 0.0
            results['Vmean'] = 0.0
            results['TT'] = 0.0
        
        return results
    
    def _count_lines(self, binary_array: np.ndarray) -> List[int]:
        """Count lengths of consecutive ones in a binary array."""
        if not np.any(binary_array):
            return []
        
        # Convert to string for easier processing
        s = ''.join(map(str, binary_array.astype(int)))
        return [len(ones) for ones in s.split('0') if ones]
    
    def correlation_dimension(self, time_series: np.ndarray, 
                             embedding_dim: int = None, 
                             time_delay: int = None,
                             max_radius: float = None,
                             n_points: int = 1000) -> float:
        """
        Calculate the correlation dimension of a time series.
        
        Args:
            time_series (np.ndarray): Input time series
            embedding_dim (int, optional): Embedding dimension
            time_delay (int, optional): Time delay
            max_radius (float, optional): Maximum radius for correlation sum
            n_points (int): Number of points to sample
            
        Returns:
            float: Correlation dimension
        """
        # Ensure the time series is a numpy array
        time_series = np.asarray(time_series)
        
        # Automatically determine embedding parameters if not provided
        if time_delay is None:
            time_delay = self.estimate_time_delay(time_series)
        
        if embedding_dim is None:
            embedding_dim = self.estimate_embedding_dimension(time_series, time_delay)
        
        # Reconstruct phase space
        phase_space = self.phase_space_reconstruction(time_series, embedding_dim, time_delay)
        
        # Number of points in the phase space
        total_points = phase_space.shape[0]
        
        # Sample points if necessary
        if total_points > n_points:
            indices = np.random.choice(total_points, n_points, replace=False)
            phase_space = phase_space[indices]
        else:
            n_points = total_points
        
        # Calculate pair-wise distances
        distances = np.zeros((n_points, n_points))
        for i in range(n_points):
            for j in range(i+1, n_points):
                dist = np.linalg.norm(phase_space[i] - phase_space[j])
                distances[i, j] = distances[j, i] = dist
        
        # Determine maximum radius if not provided
        if max_radius is None:
            max_radius = np.max(distances) * 0.5
        
        # Calculate correlation sum for different radii
        min_radius = np.min(distances[distances > 0]) * 1.1
        radii = np.logspace(np.log10(min_radius), np.log10(max_radius), 20)
        correlation_sum = np.zeros_like(radii)
        
        for i, radius in enumerate(radii):
            correlation_sum[i] = np.sum(distances < radius) / (n_points * (n_points - 1))
        
        # Calculate correlation dimension as the slope of log(C(r)) vs log(r)
        log_radii = np.log(radii)
        log_correlation = np.log(correlation_sum)
        
        # Linear regression
        slope, _, _, _, _ = stats.linregress(log_radii, log_correlation)
        
        return slope
    
    def kolmogorov_entropy(self, time_series: np.ndarray, 
                          embedding_dim: int = None, 
                          time_delay: int = None) -> float:
        """
        Estimate the Kolmogorov entropy of a time series.
        
        Args:
            time_series (np.ndarray): Input time series
            embedding_dim (int, optional): Embedding dimension
            time_delay (int, optional): Time delay
            
        Returns:
            float: Estimated Kolmogorov entropy
        """
        # This is actually the Kolmogorov-Sinai entropy, approximated by the
        # maximal Lyapunov exponent for chaotic systems
        return max(0, self.lyapunov_exponent(time_series, embedding_dim, time_delay))
    
    def predictability(self, lyapunov: float, time_units: str = 'steps') -> Dict[str, float]:
        """
        Calculate predictability measures from Lyapunov exponent.
        
        Args:
            lyapunov (float): Maximal Lyapunov exponent
            time_units (str): Time units for interpretation
            
        Returns:
            Dict: Dictionary with predictability measures
        """
        if lyapunov <= 0:
            # System is stable or at the edge of chaos
            results = {
                'is_chaotic': False,
                'predictability': 'high',
                'horizon_time': float('inf')
            }
        else:
            # System is chaotic with a finite prediction horizon
            # Horizon time is approximately 1/λ
            horizon = 1.0 / lyapunov
            
            results = {
                'is_chaotic': True,
                'predictability': 'low',
                'horizon_time': horizon,
                'horizon_units': time_units
            }
        
        return results
    
    def plot_phase_space(self, time_series: np.ndarray, 
                        embedding_dim: int = 3, 
                        time_delay: int = None) -> plt.Figure:
        """
        Plot the phase space reconstruction of a time series.
        
        Args:
            time_series (np.ndarray): Input time series
            embedding_dim (int): Embedding dimension
            time_delay (int, optional): Time delay
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        # Ensure the time series is a numpy array
        time_series = np.asarray(time_series)
        
        # Determine time delay if not provided
        if time_delay is None:
            time_delay = self.estimate_time_delay(time_series)
        
        # Check if we can plot in 2D or 3D
        if embedding_dim < 2 or embedding_dim > 3:
            # Force 3D for plotting
            embedding_dim = min(3, len(time_series) // time_delay)
        
        # Reconstruct phase space
        phase_space = self.phase_space_reconstruction(time_series, embedding_dim, time_delay)
        
        # Create figure
        fig = plt.figure(figsize=(10, 8))
        
        if embedding_dim == 2:
            # 2D phase space
            plt.scatter(phase_space[:, 0], phase_space[:, 1], s=5, alpha=0.5)
            plt.plot(phase_space[:, 0], phase_space[:, 1], alpha=0.3)
            plt.xlabel(f'x(t)')
            plt.ylabel(f'x(t+τ), τ={time_delay}')
            plt.title(f'2D Phase Space Reconstruction (τ={time_delay})')
            plt.grid(True, alpha=0.3)
        else:
            # 3D phase space
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(phase_space[:, 0], phase_space[:, 1], phase_space[:, 2], s=5, alpha=0.5)
            ax.plot(phase_space[:, 0], phase_space[:, 1], phase_space[:, 2], alpha=0.3)
            ax.set_xlabel(f'x(t)')
            ax.set_ylabel(f'x(t+τ), τ={time_delay}')
            ax.set_zlabel(f'x(t+2τ), τ={time_delay}')
            ax.set_title(f'3D Phase Space Reconstruction (τ={time_delay})')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def plot_recurrence(self, recurrence_matrix: np.ndarray, 
                       title: str = "Recurrence Plot") -> plt.Figure:
        """
        Plot a recurrence matrix.
        
        Args:
            recurrence_matrix (np.ndarray): Recurrence matrix
            title (str): Plot title
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot recurrence matrix as an image
        im = ax.imshow(recurrence_matrix, cmap='binary', origin='lower', 
                      interpolation='none', aspect='equal')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Recurrence')
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('Time Index')
        ax.set_ylabel('Time Index')
        
        plt.tight_layout()
        
        return fig
    
    def analyze_time_series(self, time_series: np.ndarray, title: str = None) -> Dict[str, Any]:
        """
        Perform a comprehensive chaos analysis on a time series.
        
        Args:
            time_series (np.ndarray): Input time series
            title (str, optional): Title for plots
            
        Returns:
            Dict: Dictionary with analysis results
        """
        # Ensure the time series is a numpy array
        time_series = np.asarray(time_series)
        
        # Use provided title or default
        if title is None:
            title = "Chaos Analysis"
        
        # Calculate time delay and embedding dimension
        time_delay = self.estimate_time_delay(time_series)
        embedding_dim = self.estimate_embedding_dimension(time_series, time_delay)
        
        # Calculate Lyapunov exponent
        lyapunov = self.lyapunov_exponent(time_series, embedding_dim, time_delay)
        
        # Calculate correlation dimension
        corr_dim = self.correlation_dimension(time_series, embedding_dim, time_delay)
        
        # Calculate Kolmogorov entropy
        kolmogorov = self.kolmogorov_entropy(time_series, embedding_dim, time_delay)
        
        # Generate recurrence plot
        recurrence = self.recurrence_plot(time_series, embedding_dim, time_delay)
        
        # Calculate recurrence quantification measures
        rqa = self.recurrence_quantification(recurrence)
        
        # Assess predictability
        pred = self.predictability(lyapunov)
        
        # Create result dictionary
        results = {
            'time_delay': time_delay,
            'embedding_dimension': embedding_dim,
            'lyapunov_exponent': lyapunov,
            'correlation_dimension': corr_dim,
            'kolmogorov_entropy': kolmogorov,
            'predictability': pred,
            'rqa': rqa
        }
        
        # Generate plots
        results['plots'] = {
            'phase_space': self.plot_phase_space(time_series, min(3, embedding_dim), time_delay),
            'recurrence': self.plot_recurrence(recurrence, f"Recurrence Plot ({title})")
        }
        
        return results


# Common chaotic systems for testing and simulation

def logistic_map(n_points: int = 1000, r: float = 3.9, x0: float = 0.5) -> np.ndarray:
    """
    Generate a time series from the logistic map.
    
    Args:
        n_points (int): Number of points to generate
        r (float): Control parameter (chaotic for r > 3.57)
        x0 (float): Initial value
        
    Returns:
        np.ndarray: Time series from the logistic map
    """
    x = np.zeros(n_points)
    x[0] = x0
    
    for i in range(1, n_points):
        x[i] = r * x[i-1] * (1 - x[i-1])
    
    return x

def henon_map(n_points: int = 1000, a: float = 1.4, b: float = 0.3, 
             x0: float = 0.0, y0: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a time series from the Henon map.
    
    Args:
        n_points (int): Number of points to generate
        a (float): Control parameter (chaotic for a=1.4, b=0.3)
        b (float): Control parameter
        x0 (float): Initial x value
        y0 (float): Initial y value
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y time series from the Henon map
    """
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    
    x[0] = x0
    y[0] = y0
    
    for i in range(1, n_points):
        x[i] = 1 - a * x[i-1]**2 + y[i-1]
        y[i] = b * x[i-1]
    
    return x, y

def lorenz_system(n_points: int = 10000, dt: float = 0.01, 
                 sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0,
                 x0: float = 0.1, y0: float = 0.1, z0: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a time series from the Lorenz system using Euler integration.
    
    Args:
        n_points (int): Number of points to generate
        dt (float): Time step
        sigma (float): Control parameter
        rho (float): Control parameter
        beta (float): Control parameter
        x0 (float): Initial x value
        y0 (float): Initial y value
        z0 (float): Initial z value
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: x, y, and z time series from the Lorenz system
    """
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    z = np.zeros(n_points)
    
    x[0] = x0
    y[0] = y0
    z[0] = z0
    
    for i in range(1, n_points):
        x_dot = sigma * (y[i-1] - x[i-1])
        y_dot = x[i-1] * (rho - z[i-1]) - y[i-1]
        z_dot = x[i-1] * y[i-1] - beta * z[i-1]
        
        x[i] = x[i-1] + x_dot * dt
        y[i] = y[i-1] + y_dot * dt
        z[i] = z[i-1] + z_dot * dt
    
    return x, y, z 
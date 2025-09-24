# Data Generation Module
# CRISP-DM: Data Preparation Phase

import numpy as np
import pandas as pd
from typing import Tuple, Optional

def generate_linear_data(
    a: float = 2.0,
    b: float = 1.0,
    noise_level: float = 0.1,
    n_points: int = 100,
    x_range: Tuple[float, float] = (0, 10),
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic linear data using y = ax + b + noise
    
    Parameters:
    -----------
    a : float
        Slope of the linear relationship
    b : float
        Intercept of the linear relationship
    noise_level : float
        Standard deviation of Gaussian noise added to y values
    n_points : int
        Number of data points to generate
    x_range : tuple
        Range of x values (min, max)
    random_seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    tuple : (x, y) arrays
        Generated x and y data
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate x values uniformly distributed within the specified range
    x = np.linspace(x_range[0], x_range[1], n_points)
    
    # Add some randomness to x values to make it more realistic
    x_noise = np.random.normal(0, (x_range[1] - x_range[0]) * 0.01, n_points)
    x = x + x_noise
    
    # Generate y values using linear relationship with noise
    y_true = a * x + b
    noise = np.random.normal(0, noise_level, n_points)
    y = y_true + noise
    
    return x, y

def create_dataframe(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """
    Create a pandas DataFrame from x and y arrays
    
    Parameters:
    -----------
    x : np.ndarray
        X values
    y : np.ndarray
        Y values
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns 'x' and 'y'
    """
    return pd.DataFrame({'x': x, 'y': y})

def validate_parameters(a: float, b: float, noise_level: float, n_points: int) -> bool:
    """
    Validate input parameters for data generation
    
    Parameters:
    -----------
    a : float
        Slope parameter
    b : float
        Intercept parameter
    noise_level : float
        Noise level parameter
    n_points : int
        Number of points parameter
    
    Returns:
    --------
    bool
        True if parameters are valid, raises ValueError otherwise
    """
    if not isinstance(n_points, int) or n_points <= 0:
        raise ValueError("n_points must be a positive integer")
    
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative")
    
    if not all(isinstance(param, (int, float)) for param in [a, b, noise_level]):
        raise ValueError("a, b, and noise_level must be numeric")
    
    return True
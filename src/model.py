# Linear Regression Model Module
# CRISP-DM: Modeling Phase

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Tuple, Dict

class LinearRegressionModel:
    """
    A wrapper class for linear regression modeling following CRISP-DM methodology
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the linear regression model
        
        Parameters:
        -----------
        random_state : int
            Random state for reproducibility
        """
        self.model = LinearRegression()
        self.random_state = random_state
        self.is_trained = False
        self.train_metrics = {}
        self.test_metrics = {}
        
    def prepare_data(self, x: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple:
        """
        Prepare data for training by splitting into train/test sets
        
        Parameters:
        -----------
        x : np.ndarray
            Feature array
        y : np.ndarray
            Target array
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        # Reshape x if it's 1D
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            
        return train_test_split(x, y, test_size=test_size, random_state=self.random_state)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the linear regression model
        
        Parameters:
        -----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        """
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        self.train_metrics = self._calculate_metrics(y_train, y_train_pred)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data
        
        Parameters:
        -----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets
            
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
            
        y_test_pred = self.model.predict(X_test)
        self.test_metrics = self._calculate_metrics(y_test, y_test_pred)
        return self.test_metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Parameters:
        -----------
        X : np.ndarray
            Features to predict on
            
        Returns:
        --------
        np.ndarray
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
            
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        return self.model.predict(X)
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Get the model parameters (slope and intercept)
        
        Returns:
        --------
        dict
            Dictionary containing 'slope' and 'intercept'
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before accessing parameters")
            
        return {
            'slope': self.model.coef_[0] if hasattr(self.model.coef_, '__len__') else self.model.coef_,
            'intercept': self.model.intercept_
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Parameters:
        -----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
            
        Returns:
        --------
        dict
            Dictionary containing various metrics
        """
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def get_model_summary(self) -> Dict:
        """
        Get a comprehensive summary of the model
        
        Returns:
        --------
        dict
            Dictionary containing model parameters and metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting summary")
            
        summary = {
            'parameters': self.get_parameters(),
            'train_metrics': self.train_metrics,
            'test_metrics': self.test_metrics,
            'is_trained': self.is_trained
        }
        
        return summary
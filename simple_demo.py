# Simple Linear Regression Demo
# Simplified version that works with basic Python installation

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def generate_linear_data(a=2.0, b=1.0, noise_level=0.1, n_points=100, random_seed=42):
    """Generate synthetic linear data using y = ax + b + noise"""
    np.random.seed(random_seed)
    
    x = np.linspace(0, 10, n_points)
    x_noise = np.random.normal(0, 0.1, n_points)
    x = x + x_noise
    
    y_true = a * x + b
    noise = np.random.normal(0, noise_level, n_points)
    y = y_true + noise
    
    return x, y, y_true

def train_and_evaluate(x, y, test_size=0.2, random_state=42):
    """Train linear regression model and return results"""
    # Prepare data
    X = x.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    results = {
        'model': model,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_mse': mean_squared_error(y_train, y_pred_train),
        'test_mse': mean_squared_error(y_test, y_pred_test),
        'slope': model.coef_[0],
        'intercept': model.intercept_
    }
    
    return results

def plot_results(x, y, y_true, results, a_true, b_true):
    """Create visualization plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Data and regression lines
    x_plot = np.linspace(x.min(), x.max(), 100)
    y_pred_plot = results['model'].predict(x_plot.reshape(-1, 1))
    y_true_plot = a_true * x_plot + b_true
    
    ax1.scatter(results['X_train'].flatten(), results['y_train'], alpha=0.6, color='blue', label='Training Data')
    ax1.scatter(results['X_test'].flatten(), results['y_test'], alpha=0.6, color='orange', label='Test Data')
    ax1.plot(x_plot, y_pred_plot, 'r-', linewidth=2, label=f'Fitted: y = {results["slope"]:.2f}x + {results["intercept"]:.2f}')
    ax1.plot(x_plot, y_true_plot, 'g--', linewidth=2, label=f'True: y = {a_true}x + {b_true}')
    ax1.set_xlabel('X values')
    ax1.set_ylabel('Y values')
    ax1.set_title('Linear Regression Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    residuals = results['y_test'] - results['y_pred_test']
    ax2.scatter(results['y_pred_test'], residuals, alpha=0.7)
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Actual vs Predicted
    ax3.scatter(results['y_test'], results['y_pred_test'], alpha=0.7)
    min_val = min(results['y_test'].min(), results['y_pred_test'].min())
    max_val = max(results['y_test'].max(), results['y_pred_test'].max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax3.set_xlabel('Actual Values')
    ax3.set_ylabel('Predicted Values')
    ax3.set_title(f'Actual vs Predicted (R² = {results["test_r2"]:.4f})')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Metrics comparison
    metrics = ['R²', 'MSE']
    train_vals = [results['train_r2'], results['train_mse']]
    test_vals = [results['test_r2'], results['test_mse']]
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x_pos - width/2, train_vals, width, label='Training', alpha=0.7)
    ax4.bar(x_pos + width/2, test_vals, width, label='Test', alpha=0.7)
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Values')
    ax4.set_title('Performance Metrics')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the linear regression demo"""
    print("🚀 AutoDeployLR: Linear Regression Demo")
    print("="*50)
    
    # Set parameters
    a_true = 2.5
    b_true = 1.0
    noise_level = 0.3
    n_points = 200
    
    print(f"📊 Generating data with parameters:")
    print(f"   True slope (a): {a_true}")
    print(f"   True intercept (b): {b_true}")
    print(f"   Noise level: {noise_level}")
    print(f"   Number of points: {n_points}")
    
    # Generate data
    x, y, y_true = generate_linear_data(a_true, b_true, noise_level, n_points)
    
    # Train and evaluate model
    results = train_and_evaluate(x, y)
    
    # Print results
    print(f"\n🎯 Model Results:")
    print(f"   Estimated slope: {results['slope']:.4f} (True: {a_true})")
    print(f"   Estimated intercept: {results['intercept']:.4f} (True: {b_true})")
    print(f"   Training R²: {results['train_r2']:.4f}")
    print(f"   Test R²: {results['test_r2']:.4f}")
    print(f"   Training MSE: {results['train_mse']:.4f}")
    print(f"   Test MSE: {results['test_mse']:.4f}")
    
    # Create plots
    plot_results(x, y, y_true, results, a_true, b_true)
    
    print("\n✅ Analysis completed successfully!")

if __name__ == "__main__":
    main()
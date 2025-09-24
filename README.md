# AutoDeployLR: Automated Linear Regression Deployment

## 🌐 Live Demo
**Try the interactive web application:** [https://oliverlai-aiot-hw1.streamlit.app/](https://oliverlai-aiot-hw1.streamlit.app/)

## Project Overview
This project implements an automated linear regression system following the CRISP-DM framework. It generates synthetic data using the formula `y = ax + b + noise`, trains a linear regression model, and deploys it through a web interface for interactive use.

## CRISP-DM Framework Implementation

### 1. Business Understanding
The goal is to create an automated system for linear regression analysis that allows users to:
- Generate synthetic data with customizable parameters
- Train and evaluate linear regression models
- Visualize results through an interactive web interface
- Track the development process through prompt logging

### 2. Data Understanding
The system generates synthetic linear data with the following characteristics:
- Linear relationship: `y = ax + b + noise`
- Adjustable slope (a) and intercept (b)
- Configurable noise level for realistic data simulation
- Variable dataset size for different analysis needs

### 3. Data Preparation
Data preparation includes:
- Synthetic data generation with specified parameters
- Data validation and quality checks
- Feature scaling if needed
- Train-test split for model evaluation

### 4. Modeling
The modeling phase involves:
- Simple linear regression using scikit-learn
- Model training with generated synthetic data
- Parameter estimation and model fitting
- Model performance evaluation

### 5. Evaluation
Model evaluation includes:
- R² score calculation
- Mean Squared Error (MSE) analysis
- Residual analysis
- Visual assessment of model fit

### 6. Deployment
The deployment phase features:
- Streamlit web application for user interaction
- Real-time parameter adjustment
- Interactive data visualization
- Model results display

## Project Structure
```
├── README.md
├── requirements.txt
├── src/
│   ├── data_generation.py
│   ├── model.py
│   ├── app.py
│   └── utils.py
├── logs/
│   └── prompts.log
└── notebooks/
    └── exploration.ipynb
```

## Usage Instructions

### Option 1: Full Web Application (Recommended)
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
2. **Run the Streamlit web application:**
   ```bash
   streamlit run src/app.py
   ```
   
3. **Use the web interface:**
   - Adjust parameters in the sidebar (slope, intercept, noise, etc.)
   - Click "Generate Data & Train Model" to see results
   - View interactive visualizations and metrics

### Option 2: Simple Demo (If dependencies fail)
If you encounter import errors, run the simplified demo:
```bash
python simple_demo.py
```

### Option 3: Jupyter Notebook
Explore the complete analysis in the notebook:
```bash
jupyter notebook notebooks/exploration.ipynb
```

## Troubleshooting

### Import Errors
If you get import errors when running the Streamlit app:

1. **Make sure you're running from the project root directory**
2. **Check Python environment:** Ensure all packages are installed in the correct environment
3. **Try the simple demo:** Run `python simple_demo.py` for a basic version
4. **Manual installation:** Install packages individually:
   ```bash
   pip install streamlit
   pip install scikit-learn
   pip install matplotlib
   pip install numpy pandas plotly
   ```

### Python Environment Issues
- If using conda: `conda install streamlit scikit-learn matplotlib numpy pandas plotly`
- If using virtual environment: Activate it first, then install packages
- Check Python version: Requires Python 3.7+

## Features
- Interactive parameter adjustment for data generation
- Real-time model training and evaluation
- Visual results display with plots
- Comprehensive logging of development prompts
- CRISP-DM compliant project structure

## Development
This project tracks all development prompts in `logs/prompts.log` for reproducibility and documentation purposes.
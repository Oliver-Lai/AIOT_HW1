# AutoDeployLR: Automated Linear Regression Deployment

## 🌐 Live Demo
**Try the interactive web application:** [https://oliverlai-aiot-hw1.streamlit.app/](https://oliverlai-aiot-hw1.streamlit.app/)

## Project Overview
This project implements an automated linear regression system following the CRISP-DM framework. It generates synthetic data using the formula `y = ax + b + noise`, trains a linear regression model, and deploys it through a web interface for interactive use.

## Development Log
* Project initialization started - AutoDeployLR following CRISP-DM framework
* Step 1 completed: Project structure created with README.md, requirements.txt, src/, logs/
* Data generation module implemented with adjustable parameters for y=ax+b+noise
* Linear regression model implemented with sklearn, including train and evaluate functions
* Streamlit web application created with interactive parameter controls and visualization
* Utility functions implemented including prompt logging system
* Comprehensive exploration notebook created with CRISP-DM methodology
* Import error encountered: Fixed relative imports in app.py for Streamlit deployment
* Python environment configuration issues detected and addressed
* Added comprehensive troubleshooting section to README.md
* User requested GitHub repository creation and upload - AIOT_HW1
* Prepared .gitignore file for Python project
* Ready to initialize Git repository and push to GitHub
* Git repository initialized and files committed locally
* Ready to add remote origin and push to GitHub AIOT_HW1 repository
* Successfully uploaded project to GitHub repository: Oliver-Lai/AIOT_HW1
* Added Demo Site link to README.md: https://oliverlai-aiot-hw1.streamlit.app/
* Preparing to push updated README with demo link to GitHub

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
│   └── 產生步驟.log
│   └── prompts.md
```
## Features
- Interactive parameter adjustment for data generation
- Real-time model training and evaluation
- Visual results display with plots
- Comprehensive logging of development prompts
- CRISP-DM compliant project structure

## Development
This project tracks all development prompts in `logs/產生步驟.log` for reproducibility and documentation purposes.
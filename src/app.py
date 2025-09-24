# Streamlit Web Application
# CRISP-DM: Deployment Phase

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sys
import os

# Add the project root to the path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import the modules
try:
    from src.data_generation import generate_linear_data, create_dataframe, validate_parameters
    from src.model import LinearRegressionModel
    from src.utils import log_prompt
except ImportError:
    # Fallback for direct imports if running from src directory
    from data_generation import generate_linear_data, create_dataframe, validate_parameters
    from model import LinearRegressionModel
    from utils import log_prompt

# Configure Streamlit page
st.set_page_config(
    page_title="AutoDeployLR: Linear Regression Analysis",
    page_icon="📈",
    layout="wide"
)

def main():
    """Main application function"""
    st.title("📈 AutoDeployLR: Automated Linear Regression Deployment")
    st.markdown("### Following the CRISP-DM Framework")
    
    # Log the session start
    log_prompt("Streamlit application started")
    
    # Sidebar for parameters
    st.sidebar.header("🔧 Data Generation Parameters")
    st.sidebar.markdown("**CRISP-DM: Data Preparation**")
    
    # Parameter inputs
    a = st.sidebar.slider("Slope (a)", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
    b = st.sidebar.slider("Intercept (b)", min_value=-10.0, max_value=10.0, value=1.0, step=0.1)
    noise_level = st.sidebar.slider("Noise Level", min_value=0.0, max_value=2.0, value=0.1, step=0.01)
    n_points = st.sidebar.slider("Number of Points", min_value=50, max_value=1000, value=100, step=10)
    
    # Advanced parameters
    with st.sidebar.expander("Advanced Parameters"):
        x_min = st.number_input("X Range Min", value=0.0)
        x_max = st.number_input("X Range Max", value=10.0)
        test_size = st.slider("Test Size Ratio", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_seed = st.number_input("Random Seed", value=42, step=1)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Business Understanding")
        st.markdown("""
        This application demonstrates automated linear regression following the CRISP-DM methodology:
        - Generate synthetic data with adjustable parameters
        - Train and evaluate linear regression models
        - Visualize results interactively
        """)
        
        # Data generation button
        if st.button("🔄 Generate Data & Train Model", type="primary"):
            try:
                # Validate parameters
                validate_parameters(a, b, noise_level, n_points)
                
                # Log the data generation
                log_prompt(f"Data generation requested with parameters: a={a}, b={b}, noise={noise_level}, n_points={n_points}")
                
                # Generate data
                with st.spinner("Generating data..."):
                    x, y = generate_linear_data(
                        a=a, b=b, noise_level=noise_level, n_points=n_points,
                        x_range=(x_min, x_max), random_seed=random_seed
                    )
                
                # Store data in session state
                st.session_state['x'] = x
                st.session_state['y'] = y
                st.session_state['parameters'] = {'a': a, 'b': b, 'noise': noise_level, 'n_points': n_points}
                
                st.success("✅ Data generated successfully!")
                
                # Train model
                with st.spinner("Training model..."):
                    model = LinearRegressionModel(random_state=random_seed)
                    X_train, X_test, y_train, y_test = model.prepare_data(x, y, test_size=test_size)
                    model.train(X_train, y_train)
                    test_metrics = model.evaluate(X_test, y_test)
                    
                    # Store model results
                    st.session_state['model'] = model
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                
                st.success("✅ Model trained successfully!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                log_prompt(f"Error occurred: {str(e)}")
    
    with col2:
        st.header("📊 Current Parameters")
        st.markdown(f"""
        **True Model**: y = {a}x + {b}
        
        - **Slope (a)**: {a}
        - **Intercept (b)**: {b}
        - **Noise Level**: {noise_level}
        - **Data Points**: {n_points}
        - **X Range**: [{x_min}, {x_max}]
        - **Test Size**: {test_size}
        """)
    
    # Display results if data exists
    if 'x' in st.session_state and 'model' in st.session_state:
        display_results()

def display_results():
    """Display the analysis results"""
    st.header("📈 Analysis Results")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Visualization", "🔍 Model Evaluation", "📋 Data Understanding", "🚀 Model Summary"])
    
    with tab1:
        st.subheader("Data Visualization")
        
        # Create interactive plot
        x = st.session_state['x']
        y = st.session_state['y']
        model = st.session_state['model']
        
        # Generate predictions for plotting
        x_plot = np.linspace(x.min(), x.max(), 100)
        y_pred_plot = model.predict(x_plot.reshape(-1, 1))
        
        # Create plotly figure
        fig = go.Figure()
        
        # Add scatter plot of actual data
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            name='Generated Data',
            marker=dict(color='blue', size=6, opacity=0.6)
        ))
        
        # Add regression line
        fig.add_trace(go.Scatter(
            x=x_plot, y=y_pred_plot,
            mode='lines',
            name='Fitted Line',
            line=dict(color='red', width=3)
        ))
        
        # Add true line
        params = st.session_state['parameters']
        y_true_plot = params['a'] * x_plot + params['b']
        fig.add_trace(go.Scatter(
            x=x_plot, y=y_true_plot,
            mode='lines',
            name='True Line',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Linear Regression Analysis",
            xaxis_title="X values",
            yaxis_title="Y values",
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Model Evaluation")
        
        model = st.session_state['model']
        model_params = model.get_parameters()
        true_params = st.session_state['parameters']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Model Performance**")
            train_metrics = model.train_metrics
            test_metrics = model.test_metrics
            
            metrics_df = pd.DataFrame({
                'Metric': ['R² Score', 'MSE', 'RMSE', 'MAE'],
                'Train': [
                    f"{train_metrics['r2']:.4f}",
                    f"{train_metrics['mse']:.4f}",
                    f"{train_metrics['rmse']:.4f}",
                    f"{train_metrics['mae']:.4f}"
                ],
                'Test': [
                    f"{test_metrics['r2']:.4f}",
                    f"{test_metrics['mse']:.4f}",
                    f"{test_metrics['rmse']:.4f}",
                    f"{test_metrics['mae']:.4f}"
                ]
            })
            
            st.dataframe(metrics_df, use_container_width=True)
        
        with col2:
            st.markdown("**🎯 Parameter Comparison**")
            comparison_df = pd.DataFrame({
                'Parameter': ['Slope (a)', 'Intercept (b)'],
                'True Value': [true_params['a'], true_params['b']],
                'Estimated': [model_params['slope'], model_params['intercept']],
                'Difference': [
                    abs(model_params['slope'] - true_params['a']),
                    abs(model_params['intercept'] - true_params['b'])
                ]
            })
            
            st.dataframe(comparison_df, use_container_width=True)
    
    with tab3:
        st.subheader("Data Understanding")
        
        x = st.session_state['x']
        y = st.session_state['y']
        df = create_dataframe(x, y)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Data Statistics**")
            st.dataframe(df.describe(), use_container_width=True)
        
        with col2:
            st.markdown("**🔍 Data Sample**")
            st.dataframe(df.head(10), use_container_width=True)
    
    with tab4:
        st.subheader("Model Summary")
        
        model = st.session_state['model']
        summary = model.get_model_summary()
        
        st.markdown("**🤖 Complete Model Summary**")
        st.json(summary)

if __name__ == "__main__":
    main()
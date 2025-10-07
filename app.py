# car_price_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Page Configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Enhanced Custom CSS with car theme
st.markdown("""
    <style>
    /* Main theme */
    .main {
        background: linear-gradient(135deg, #1a1a1a 0%, #2c3e50 100%);
        color: white;
    }
    
    /* Dashboard container */
    .dashboard-container {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Car themed cards */
    .specs-card {
        background: rgba(52, 152, 219, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(52, 152, 219, 0.3);
        transition: all 0.3s ease;
    }
    
    .specs-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(52, 152, 219, 0.2);
    }
    
    /* Animated elements */
    .car-icon {
        animation: drive 2s infinite linear;
    }
    
    @keyframes drive {
        0% { transform: translateX(-20px); }
        50% { transform: translateX(20px); }
        100% { transform: translateX(-20px); }
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
    }
    
    /* Prediction box */
    .prediction-box {
        background: linear-gradient(45deg, #2ecc71, #27ae60);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        animation: glow 2s infinite alternate;
    }
    
    @keyframes glow {
        from { box-shadow: 0 0 10px rgba(46, 204, 113, 0.5); }
        to { box-shadow: 0 0 20px rgba(46, 204, 113, 0.8); }
    }
    
    /* Gauge animation */
    .gauge-container {
        transition: all 0.5s ease;
    }
    
    /* Input fields */
    .stNumberInput, .stSelectbox {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 5px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 10px;
        color: white;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(52, 152, 219, 0.2);
    }
    
    /* Progress bar animation */
    @keyframes progress {
        0% { width: 0; }
        100% { width: 100%; }
    }
    
    .progress-bar {
        height: 4px;
        background: #3498db;
        animation: progress 2s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Add dynamic header with animated car
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>
            <span class='car-icon'>🚗</span> 
            Smart Car Value Estimator
            <span class='car-icon' style='transform: scaleX(-1);'>🚗</span>
        </h1>
    </div>
""", unsafe_allow_html=True)

# Add dynamic dashboard container
st.markdown("<div class='dashboard-container'>", unsafe_allow_html=True)

# ===============================
# Load Dataset
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv('car_data/car data.csv')
    df['Car_Age'] = 2025 - df['Year']
    
    # Create brand column from Car_Name
    df['Brand'] = df['Car_Name'].apply(lambda x: x.split()[0])
    
    # Ensure Selling_Price is always less than Present_Price
    df = df[df['Selling_Price'] < df['Present_Price']]
    
    # Remove outliers
    df = df[df['Selling_Price'] > df['Present_Price'] * 0.2]
    
    return df

df = load_data()
original_df = df.copy()

# Feature engineering
df['Brand_encoded'] = pd.factorize(df['Brand'])[0]
brand_mapping = dict(zip(df['Brand'].unique(), df['Brand_encoded'].unique()))

df.drop(['Year', 'Car_Name', 'Brand'], axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)

# Train-Test Split
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# ===============================
# Streamlit UI
# ===============================
st.title("🚘 Smart Car Value Estimator")
st.markdown("""
    <div style='background-color: #2c3e50; padding: 1rem; border-radius: 10px; color: white;'>
        <h3>🎯 Get an Accurate Estimate for Your Car's Value</h3>
        <p>Using advanced machine learning to analyze market trends and car specifications</p>
    </div>
""", unsafe_allow_html=True)

# Create three columns for better layout
col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.markdown("<div class='specs-card'>", unsafe_allow_html=True)
    st.subheader("🚗 Basic Details")
    
    # Add car brand selection
    unique_brands = list(brand_mapping.keys())
    car_brand = st.selectbox("Car Brand", 
                            unique_brands,
                            help="Select your car's manufacturer")
    
    brand_encoded = brand_mapping[car_brand]
    
    present_price = st.number_input("Market Price (Lakhs)", 
                                  min_value=0.0, step=0.1)
    kms_driven = st.number_input("Odometer Reading (km)",
                                min_value=0, step=100)
    car_age = st.slider("Vehicle Age",
                       min_value=0, max_value=30, step=1)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='specs-card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Technical Specs")
    fuel_type = st.selectbox("Fuel Type",
                            ["Petrol ⛽", "Diesel 🛢️", "CNG 🌱"])
    transmission = st.selectbox("Transmission",
                              ["Manual 🔧", "Automatic ⚡"])
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='specs-card'>", unsafe_allow_html=True)
    st.subheader("📋 Additional Info")
    owner = st.selectbox("Previous Owners",
                        ["First Owner 🥇", "Second Owner 🥈", "Third Owner 🥉", "Fourth+ Owner"])
    seller_type = st.selectbox("Seller Category",
                              ["Dealer 🏪", "Individual 👤"])
    st.markdown("</div>", unsafe_allow_html=True)

# Clean up the input data
fuel_type = fuel_type.split()[0]
transmission = transmission.split()[0]
owner = {"First Owner 🥇": 0, "Second Owner 🥈": 1, "Third Owner 🥉": 2, "Fourth+ Owner": 3}[owner]
seller_type = seller_type.split()[0]

# Define binary variables for model input
fuel_diesel = 1 if fuel_type == "Diesel" else 0
fuel_petrol = 1 if fuel_type == "Petrol" else 0
seller_individual = 1 if seller_type == "Individual" else 0
trans_manual = 1 if transmission == "Manual" else 0

# Prepare input for prediction
user_input = np.array([[present_price, kms_driven, owner, car_age, brand_encoded,
                       fuel_diesel, fuel_petrol, seller_individual, trans_manual]])
# user_input_poly = poly.transform(user_input)

# Update the validate_price function to handle negative values
def validate_price(present_price, predicted_price, car_age):
    # Set reasonable depreciation rates based on car age
    MIN_DEPRECIATION = 0.10  # Minimum 10% depreciation
    YEARLY_DEPRECIATION = 0.08  # 8% depreciation per year
    
    # Handle negative predictions
    if predicted_price <= 0:
        return -1  # Special flag for negative values
    
    # Calculate maximum allowed price based on age and depreciation
    max_allowed_price = present_price * (1 - max(MIN_DEPRECIATION, car_age * YEARLY_DEPRECIATION))
    
    # Ensure predicted price doesn't exceed present price
    if predicted_price >= present_price:
        return max_allowed_price
    
    # Set minimum price (20% of present price)
    min_allowed_price = present_price * 0.20
    
    # Clamp the prediction between min and max allowed values
    if predicted_price < min_allowed_price:
        return min_allowed_price
    elif predicted_price > max_allowed_price:
        return max_allowed_price
        
    return predicted_price

# In your prediction code:
if st.button("🔍 Calculate Value", help="Click to get the predicted price"):
    with st.spinner('Analyzing market data...'):
        # Get raw prediction
        predicted_price = model.predict(user_input)[0]
        
        # Validate and adjust prediction
        validated_price = validate_price(present_price, predicted_price, car_age)
        
        if validated_price == -1:
            # Display recommendation for negative predictions
            st.markdown("""
                <div class='prediction-box' style='background: linear-gradient(45deg, #e74c3c, #c0392b);'>
                    <h2>⚠️ Low Value Alert</h2>
                    <div style='margin: 20px 0;'>
                        <h3>Recommendations:</h3>
                        <ul style='list-style-type: none; padding: 0;'>
                            <li style='margin: 10px 0;'>🔄 Consider selling to a scrap dealer</li>
                            <li style='margin: 10px 0;'>🛠️ The car might have more value in parts</li>
                            <li style='margin: 10px 0;'>⚠️ Not recommended for resale market</li>
                        </ul>
                    </div>
                    <p style='margin-top: 15px;'>Based on the car's age and condition, the predicted market value is extremely low or negative.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Calculate depreciation for valid predictions
            depreciation = ((present_price - validated_price) / present_price) * 100
            
            # Add warning if prediction seems unrealistic
            if validated_price >= present_price:
                st.warning("⚠️ Warning: Prediction adjusted due to unrealistic value")
            
            # Display normal prediction box
            st.markdown(f"""
                <div class='prediction-box'>
                    <h2>🎯 Predicted Value</h2>
                    <h1 style='font-size: 3em; margin: 10px 0;'>₹ {validated_price:.2f} Lakhs</h1>
                    <div style='display: flex; justify-content: space-between; margin-top: 15px;'>
                        <div>
                            <p>Depreciation</p>
                            <h3>{depreciation:.1f}%</h3>
                        </div>
                        <div>
                            <p>Recommendation</p>
                            <h3>{'Consider Selling' if depreciation < 70 else 'Consider Scrapping'}</h3>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Train multiple models
        def train_models(X_train, X_test, y_train, y_test):
            models = {
                'Linear Regression': LinearRegression(),
                'SVR': SVR(kernel='rbf'),
                'Decision Tree': DecisionTreeRegressor(random_state=42)
            }
            
            results = {}
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for name, model in models.items():
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                
                # Perform k-fold cross validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='r2')
                
                results[name] = {
                    'model': model,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'cv_scores': cv_scores,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
            
            return results

        # Train all models
        model_results = train_models(X_train, X_test, y_train, y_test)

        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["📊 Prediction Analysis", "📈 Market Trends", "🤖 Model Comparison"])
        
        with tab1:
            # Create gauge chart
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=predicted_price,
                    title={'text': "Predicted Price (Lakhs)"},
                    gauge={
                        'axis': {'range': [0, present_price*1.2]},
                        'bar': {'color': "#3498db"},
                        'steps': [
                            {'range': [0, present_price*0.6], 'color': "lightgray"},
                            {'range': [present_price*0.6, present_price], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': present_price
                        }
                    }
                )
            )
            fig_gauge.update_layout(height=400)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with tab2:
            # Create linear trend analysis
            fig_trend = go.Figure()
            
            # Add trend line
            x_range = np.linspace(0, kms_driven*1.2, 100).reshape(-1, 1)
            input_array = np.column_stack([
                [present_price]*100, x_range.flatten(),
                [owner]*100, [car_age]*100, [brand_encoded]*100,
                [fuel_diesel]*100, [fuel_petrol]*100,
                [seller_individual]*100, [trans_manual]*100
            ])
            y_pred = model.predict(input_array)
            
            fig_trend.add_trace(
                go.Scatter(x=x_range.flatten(), y=y_pred,
                          mode='lines', name='Price Trend',
                          line=dict(color='#3498db'))
            )
            
            # Add current car point
            fig_trend.add_trace(
                go.Scatter(x=[kms_driven], y=[predicted_price],
                          mode='markers', name='Your Car',
                          marker=dict(size=15, color='red'))
            )
            
            fig_trend.update_layout(
                title="Price vs. Kilometers Driven",
                xaxis_title="Kilometers Driven",
                yaxis_title="Price (Lakhs)",
                height=400,
                showlegend=True
            )
            st.plotly_chart(fig_trend, use_container_width=True)
            
        with tab2:
            # Price distribution
            fig2 = px.histogram(original_df, x='Selling_Price',
                              title='Price Distribution',
                              labels={'Selling_Price': 'Selling Price (Lakhs)'},
                              opacity=0.7)
            fig2.add_vline(x=predicted_price, line_dash="dash",
                          line_color="red", annotation_text="Your Car")
            st.plotly_chart(fig2)

        with tab3:
            st.subheader("Model Performance Comparison")
            
            # Create comparison dataframe
            comparison_data = {
                'Model': [],
                'Cross-Val Score': [],
                'Standard Deviation': [],
                'Train RMSE': [],
                'Test RMSE': []
            }
            
            for name, results in model_results.items():
                comparison_data['Model'].append(name)
                comparison_data['Cross-Val Score'].append(f"{results['cv_mean']:.3f}")
                comparison_data['Standard Deviation'].append(f"±{results['cv_std']:.3f}")
                comparison_data['Train RMSE'].append(f"{results['train_rmse']:.3f}")
                comparison_data['Test RMSE'].append(f"{results['test_rmse']:.3f}")
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)
            
            # Plot cross-validation scores
            fig_cv = go.Figure()
            for name, results in model_results.items():
                fig_cv.add_trace(go.Box(
                    y=results['cv_scores'],
                    name=name,
                    boxpoints='all',
                    jitter=0.3,
                    pointpos=-1.8
                ))
            
            fig_cv.update_layout(
                title="Cross-Validation Scores Distribution",
                yaxis_title="R² Score",
                showlegend=True
            )
            st.plotly_chart(fig_cv)
            
            # Add model selection
            selected_model = st.selectbox(
                "Select Model for Prediction",
                list(model_results.keys())
            )
            
            # Update prediction to use selected model
            model = model_results[selected_model]['model']

            # Add feature importance plot for Decision Tree
            if selected_model == 'Decision Tree':
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model_results['Decision Tree']['model'].feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig_importance = px.bar(
                    feature_importance,
                    x='Feature',
                    y='Importance',
                    title='Feature Importance (Decision Tree)'
                )
                st.plotly_chart(fig_importance)

# Close dashboard container
st.markdown("</div>", unsafe_allow_html=True)

# Enhanced footer with car theme
st.markdown("""
    <footer style='text-align: center; padding: 20px; margin-top: 50px; background: rgba(0,0,0,0.2); border-radius: 15px;'>
        <div style='margin-bottom: 20px;'>
            <span style='font-size: 24px;'>🚗 🚙 🏎️</span>
        </div>
        <p>Powered by Advanced Machine Learning</p>
        <p style='font-size: 12px; color: #7f8c8d;'>© 2024 Smart Car Value Estimator</p>
    </footer>
""", unsafe_allow_html=True)

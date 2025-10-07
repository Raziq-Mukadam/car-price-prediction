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
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# Global CSS (bluish theme)
st.markdown("""
<style>
/* Main theme - bluish gradient */
.main { background: linear-gradient(135deg, #071b3f 0%, #0b3d91 100%); color: #e6f0ff; }
.dashboard-container { background: rgba(10, 40, 80, 0.12); backdrop-filter: blur(8px); border-radius: 15px; padding: 20px; margin: 10px 0; border: 1px solid rgba(255,255,255,0.03); }
.specs-card { background: rgba(35, 105, 200, 0.06); border-radius: 12px; padding: 18px; margin: 10px 0; border: 1px solid rgba(75, 145, 220, 0.12); transition: all 0.25s ease; }
.specs-card:hover { transform: translateY(-4px); box-shadow: 0 8px 20px rgba(15, 63, 145, 0.2); }
.car-icon { animation: drive 2s infinite linear; }
@keyframes drive { 0% { transform: translateX(-20px); } 50% { transform: translateX(20px); } 100% { transform: translateX(-20px); } }
.stButton>button { background: linear-gradient(45deg, #1f6fb8, #0b3d91); color: white; border: none; padding: 10px 20px; border-radius: 12px; font-weight: 600; transition: all 0.2s ease; }
.stButton>button:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(11,61,145,0.28); }
.prediction-box { background: linear-gradient(135deg, rgba(52,152,219,0.12), rgba(26,115,232,0.18)); color: #eaf6ff; padding: 22px; border-radius: 12px; margin: 18px 0; border: 1px solid rgba(80,160,240,0.08); }
@keyframes glow { from { box-shadow: 0 0 10px rgba(26,115,232,0.08); } to { box-shadow: 0 0 24px rgba(26,115,232,0.12); } }
.gauge-container { transition: all 0.5s ease; }
.stNumberInput, .stSelectbox { background: rgba(255,255,255,0.03); border-radius: 8px; padding: 6px; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: rgba(255,255,255,0.02); border-radius: 12px; padding: 6px; }
.stTabs [data-baseweb="tab"] { background-color: transparent; border-radius: 8px; color: #dbeefe; transition: all 0.2s ease; }
.stTabs [data-baseweb="tab"]:hover { background-color: rgba(31,111,184,0.12); }
@keyframes progress { 0% { width: 0; } 100% { width: 100%; } }
.progress-bar { height: 4px; background: #1f6fb8; animation: progress 2s ease; }
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
    # Try multiple likely dataset locations to make the app more robust
    import os

    possible_paths = [
        os.path.join('car_data', 'car data.csv'),
        os.path.join('car_data', 'car data .csv'),
        'car data.csv',
        'Car details v4.csv',
        'Car details v3.csv',
        'CAR DETAILS FROM CAR DEKHO.csv',
        'car details v4.csv',
        'car details v3.csv'
    ]

    found = None
    for p in possible_paths:
        if os.path.isfile(p):
            found = p
            break

    if found is None:
        # Provide a helpful error that lists files in the repo root and car_data folder
        root_files = []
        try:
            root_files = os.listdir('.')
        except Exception:
            root_files = []

        car_data_files = []
        try:
            car_data_files = os.listdir('car_data')
        except Exception:
            car_data_files = []

        raise FileNotFoundError(
            "Dataset not found. Tried paths: {}.\nFiles in project root: {}\nFiles in ./car_data/: {}\n\nPlease place your CSV in one of the tried locations or update the path in load_data()."
            .format(possible_paths, root_files, car_data_files)
        )

    df = pd.read_csv(found)
    # Simple sanity check and feature creation
    if 'Year' in df.columns:
        df['Car_Age'] = 2025 - df['Year']

    # Create brand column from Car_Name if present
    if 'Car_Name' in df.columns:
        df['Brand'] = df['Car_Name'].apply(lambda x: str(x).split()[0])

    # Ensure Selling_Price is always less than Present_Price when those columns exist
    if 'Selling_Price' in df.columns and 'Present_Price' in df.columns:
        df = df[df['Selling_Price'] < df['Present_Price']]
        # Remove extreme outliers
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
# Polished header
st.markdown("""
<div style="background: linear-gradient(90deg,#0f1724,#152238); padding: 2rem; border-radius: 12px; color: white;">
    <div style="display:flex; align-items:center; gap: 1rem;">
        <div style="font-size: 48px;">🚘</div>
        <div>
            <h1 style="margin:0; font-family: 'Segoe UI', Roboto, Arial;">Smart Car Value Estimator</h1>
            <p style="margin:0; color:#cbd5e1;">A simple ML demo to estimate used car prices with explainable outputs.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main inputs and dataset metrics laid out side-by-side
left_col, right_col = st.columns([1, 1.2])

with left_col:
        st.markdown("<div class='specs-card'>", unsafe_allow_html=True)
        st.subheader("🚗 Your Car Details")
        unique_brands = list(brand_mapping.keys()) if 'brand_mapping' in globals() else ['Unknown']
        car_brand = st.selectbox("Car Brand", unique_brands)
        brand_encoded = brand_mapping.get(car_brand, 0) if 'brand_mapping' in globals() else 0

        present_price = st.number_input("Market Price (Lakhs)", min_value=0.0, step=0.1, value=float(X.mean().get('Present_Price', 5) if 'Present_Price' in X.columns else 5))
        kms_driven = st.number_input("Odometer (km)", min_value=0, step=100, value=10000)
        car_age = st.slider("Vehicle Age (years)", min_value=0, max_value=30, step=1, value=5)

        st.markdown("<hr />", unsafe_allow_html=True)
        st.subheader("⚙️ Technical Specs")
        fuel_type = st.selectbox("Fuel Type", ["Petrol ⛽", "Diesel 🛢️", "CNG 🌱"])
        transmission = st.selectbox("Transmission", ["Manual 🔧", "Automatic ⚡"])

        st.markdown("<hr />", unsafe_allow_html=True)
        st.subheader("📋 Additional Info")
        owner = st.selectbox("Previous Owners", ["First Owner 🥇", "Second Owner 🥈", "Third Owner 🥉", "Fourth+ Owner"])
        seller_type = st.selectbox("Seller Category", ["Dealer 🏪", "Individual 👤"])
        st.markdown("</div>", unsafe_allow_html=True)

with right_col:
        st.markdown("""
        <div style='display:flex; gap:12px; margin-top:8px;'>
            <div style='flex:1; background:linear-gradient(135deg,#1f2a44,#233a66); padding:16px; border-radius:10px; color:white;'>
                <h4 style='margin:0;'>Dataset Count</h4>
                <h2 style='margin:0;'>{count}</h2>
            </div>
            <div style='flex:1; background:linear-gradient(135deg,#173a2b,#1f6f45); padding:16px; border-radius:10px; color:white;'>
                <h4 style='margin:0;'>Median Price (Lakhs)</h4>
                <h2 style='margin:0;'>{median:.2f}</h2>
            </div>
            <div style='flex:1; background:linear-gradient(135deg,#4b2b6b,#6a3ea0); padding:16px; border-radius:10px; color:white;'>
                <h4 style='margin:0;'>Mean Price (Lakhs)</h4>
                <h2 style='margin:0;'>{mean:.2f}</h2>
            </div>
        </div>
        """.format(count=len(original_df) if 'original_df' in globals() else 0,
                             median=float(original_df['Selling_Price'].median()) if 'original_df' in globals() and 'Selling_Price' in original_df.columns else 0.0,
                             mean=float(original_df['Selling_Price'].mean()) if 'original_df' in globals() and 'Selling_Price' in original_df.columns else 0.0), unsafe_allow_html=True)

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

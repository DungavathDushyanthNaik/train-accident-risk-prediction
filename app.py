import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set page configuration for a wider layout
st.set_page_config(layout="wide")

# Constants for risk analysis based on synthetic data generation
RISK_THRESHOLD = 30.0  # Threshold for moderate/high risk in %

# Use st.cache_resource to ensure the model is trained only once per session
@st.cache_resource
def get_trained_model():
    """Trains a simulated model and returns the model and test data."""
    
    # 1. Generate a dummy dataset for demonstration
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'temp_celsius': np.random.uniform(-20, 70, n_samples),
        'humidity_percent': np.random.uniform(30, 100, n_samples),
        'wind_speed_kmph': np.random.uniform(0, 50, n_samples),
        'train_speed_kmph': np.random.uniform(40, 160, n_samples),
        'track_age_years': np.random.uniform(1, 50, n_samples),
        'last_inspection_days': np.random.uniform(1, 365, n_samples),
        'is_winter': np.random.randint(0, 2, n_samples),
        'is_night': np.random.randint(0, 2, n_samples),
    })
    
    # Simulate a target variable (accident probability) based on features
    # Improved risk calculation for better model performance
    df['accident_risk_score'] = (
        (df['temp_celsius'] < 0) * 0.3 + 
        (df['track_age_years'] > 30) * 0.4 + 
        (df['last_inspection_days'] > 90) * 0.3 + 
        (df['wind_speed_kmph'] > 30) * 0.2 + 
        (df['train_speed_kmph'] > 160) * 0.3 +
        (df['is_winter'] == 1) * 0.1 +
        (df['is_night'] == 1) * 0.1
    )
    
    # Convert to binary classification (accident vs no accident)
    df['accident'] = (df['accident_risk_score'] + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    
    # 2. Split data and train a simple RandomForestClassifier
    X = df.drop(['accident', 'accident_risk_score'], axis=1)
    y = df['accident']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

# --- Main App Logic ---

# Get the trained model and data
model, X_test, y_test = get_trained_model()

# --- Dashboard Layout ---

st.title(" Train Accident Risk Prediction Dashboard")

# Create two columns for layout: input form and a plot
st.sidebar.header("User Input")

# Create a form to get user input for prediction
with st.sidebar.form("prediction_form"):
    st.subheader("Predict New Accident Risk")
    
    # Input fields
    temp = st.slider("Temperature (°C)", -30.0, 100.0, 15.0)
    humidity = st.slider("Humidity (%)", 0, 100, 70)
    wind_speed = st.slider("Wind Speed (km/h)", 0.0, 100.0, 15.0)
    train_speed = st.slider("Train Speed (km/h)", 0.0, 200.0, 100.0)
    track_age = st.slider("Track Age (years)", 0, 100, 20)
    last_inspection = st.slider("Days since last inspection", 0, 365, 30)
    is_winter = st.selectbox("Is it winter?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    is_night = st.selectbox("Is it night?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    submitted = st.form_submit_button("Predict Risk")

# --- Create input_data here so it is available to the entire script on submission ---
input_data = pd.DataFrame([[
    temp, humidity, wind_speed, train_speed, track_age, last_inspection, is_winter, is_night
]], columns=[
    'temp_celsius', 'humidity_percent', 'wind_speed_kmph', 'train_speed_kmph', 'track_age_years', 
    'last_inspection_days', 'is_winter', 'is_night'
])

# --- Prediction and Results Section ---
if submitted:
    
    # 1. Make primary prediction using user's input speed
    prediction_proba = model.predict_proba(input_data)[0][1] * 100

    # Calculate the baseline risk if the train were stopped (0 km/h)
    baseline_input = input_data.copy()
    baseline_input['train_speed_kmph'] = 40
    baseline_risk_at_zero = model.predict_proba(baseline_input)[0][1] * 100
    
    st.subheader(" Prediction Result")
    
    # Display the result in a metric card
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric(label="Predicted Accident Probability", value=f"{prediction_proba:.2f}%")
        with col2:
            if prediction_proba > 60:
                st.error(" High Risk! Immediate action recommended.")
            elif prediction_proba > RISK_THRESHOLD:
                st.warning("Moderate Risk. Monitor conditions closely.")
            else:
                st.success(" Low Risk. All systems appear normal.")
            
            st.write(f"""
            This prediction is based on the following input features:
            - Input Train Speed: **{train_speed} km/h**
            - Track Age: **{track_age} years**
            - Temperature: **{temp}°C**
            - Days since last inspection: **{last_inspection} days**
            """)

    st.markdown("---")
    
    # 2. Estimate safe max train speed
    st.subheader(f" Recommended Max Train Speed (Risk < {RISK_THRESHOLD:.0f}%)")
    
    safe_speed = 0.1
    speed_risks = []
    
    # Test speeds starting from 0 km/h, up to 200 km/h
    for test_speed in np.arange(0, 201, 5):  
        test_input = input_data.copy() 
        test_input['train_speed_kmph'] = test_speed
        prob = model.predict_proba(test_input)[0][1] * 100
        speed_risks.append((test_speed, prob))
        
        if prob <= RISK_THRESHOLD:
            safe_speed = test_speed
        else:
            break
    
    st.metric(label="Recommended Safe Limit", value=f"{safe_speed:.0f}  km/h")
    
    if safe_speed == 0:
        st.error(f"**Critical Situation:** Even at 0 km/h, the risk is {baseline_risk_at_zero:.1f}%. Do not operate!")
    elif safe_speed < train_speed:
        st.warning(f"**Recommendation:** Reduce speed from {train_speed} km/h to {safe_speed} km/h for safer operation.")
    else:
        st.success(f"**Current speed ({train_speed} km/h) is within safe limits.** Maximum safe speed: {safe_speed} km/h")

    # Create speed vs risk chart
    speeds, risks = zip(*speed_risks)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(speeds, risks, 'b-', linewidth=2, label='Accident Risk')
    ax.axhline(y=RISK_THRESHOLD, color='r', linestyle='--', label=f'Safety Threshold ({RISK_THRESHOLD}%)')
    ax.axvline(x=safe_speed, color='g', linestyle='--', label=f'Safe Speed Limit ({safe_speed} km/h)')
    ax.axvline(x=train_speed, color='orange', linestyle='--', label=f'Current Speed ({train_speed} km/h)')
    ax.fill_between(speeds, risks, RISK_THRESHOLD, where=(np.array(risks)<=RISK_THRESHOLD), alpha=0.3, color='green', label='Safe Zone')
    ax.fill_between(speeds, risks, RISK_THRESHOLD, where=(np.array(risks)>RISK_THRESHOLD), alpha=0.3, color='red', label='Danger Zone')
    ax.set_xlabel('Train Speed (km/h)')
    ax.set_ylabel('Accident Risk (%)')
    ax.set_title('Speed vs Accident Risk Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    st.markdown("---")
    
    # 3. Dynamic Explanation Panel
    st.header(" Prediction Breakdown")
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.markdown("#### **1. Risk Factor Analysis**")
        
        # Calculate risk factors
        risk_factors = []
        if track_age > 30:
            risk_factors.append(f"Track Age ({track_age} years > 30 years)")
        if temp < 0:
            risk_factors.append(f"Low Temperature ({temp}°C < 0°C)")
        if last_inspection > 90:
            risk_factors.append(f"Old Inspection ({last_inspection} days > 90 days)")
        if wind_speed > 30:
            risk_factors.append(f"High Wind Speed ({wind_speed} km/h > 30 km/h)")
        if train_speed > 160:
            risk_factors.append(f"High Train Speed ({train_speed} km/h > 160 km/h)")
        if is_winter == 1:
            risk_factors.append("Winter Conditions")
        if is_night == 1:
            risk_factors.append("Night Operation")
            
        if risk_factors:
            st.warning("**Active Risk Factors:**")
            for factor in risk_factors:
                st.markdown(f" {factor}")
        else:
            st.success("No major risk factors detected.")
            
        st.info(f"Baseline risk (at 40 km/h): **{baseline_risk_at_zero:.1f}%**")

    with col_exp2:
        st.markdown("#### **2. Safety Recommendation Logic**")
        
        if baseline_risk_at_zero >= RISK_THRESHOLD:
            st.error("**Critical Issue:** Baseline risk exceeds safety threshold!")
            st.markdown(f"""
            The risk when completely stopped ({baseline_risk_at_zero:.1f}%) is already above our safety limit of {RISK_THRESHOLD}%.
            
            **Immediate Actions Required:**
            - Halt all operations
            - Conduct emergency track inspection
            - Address environmental hazards
            - Reassess when conditions improve
            """)
        else:
            st.markdown(f"""
            **Safety Analysis Process:**
            1. **Baseline Assessment:** Risk at 40 km/h = {baseline_risk_at_zero:.1f}%
            2. **Speed Testing:** Analyzed risk from 0-200 km/h in 5 km/h increments
            3. **Threshold Check:** Identified where risk crosses {RISK_THRESHOLD}%
            4. **Recommendation:** Maximum safe speed = **{safe_speed} 50 km/h**
            
            **Current Status:** Your input speed of {train_speed} km/h is {'**ABOVE**' if train_speed > safe_speed else '**WITHIN**'} the safe limit.
            """)

# --- Model Performance & Feature Importance Section ---
st.header(" Model Insights")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Model Performance")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.format({'precision': '{:.2f}', 'recall': '{:.2f}', 'f1-score': '{:.2f}', 'support': '{:.0f}'}), 
                 use_container_width=True)

with col2:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

st.markdown("---")

# Feature Importance Plot
st.subheader("Feature Importance")
st.write("This chart shows which features had the most impact on the model's predictions.")

feature_importances = model.feature_importances_
features = X_test.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

fig_importance, ax_importance = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax_importance)
ax_importance.set_title('Feature Importance in Accident Prediction')
ax_importance.set_xlabel('Importance Score')
plt.tight_layout()
st.pyplot(fig_importance)


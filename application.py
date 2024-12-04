import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Sample DataFrame for demonstration
@st.cache_data
def load_data():
    # Replace this with your actual data loading logic
    data = pd.DataFrame({
        'battery_power': np.random.randint(500, 2000, 1000),
        'blue': np.random.randint(0, 2, 1000),
        'clock_speed': np.random.uniform(0.5, 3.0, 1000),
        'dual_sim': np.random.randint(0, 2, 1000),
        'fc': np.random.randint(0, 20, 1000),
        'four_g': np.random.randint(0, 2, 1000),
        'int_memory': np.random.randint(8, 128, 1000),
        'm_dep': np.random.uniform(0.1, 1.5, 1000),
        'mobile_wt': np.random.randint(80, 250, 1000),
        'n_cores': np.random.randint(1, 8, 1000),
        'pc': np.random.randint(0, 20, 1000),
        'px_height': np.random.randint(0, 2000, 1000),
        'px_width': np.random.randint(0, 2000, 1000),
        'ram': np.random.randint(256, 8192, 1000),
        'sc_h': np.random.randint(5, 20, 1000),
        'sc_w': np.random.randint(0, 15, 1000),
        'talk_time': np.random.randint(2, 20, 1000),
        'three_g': np.random.randint(0, 2, 1000),
        'touch_screen': np.random.randint(0, 2, 1000),
        'wifi': np.random.randint(0, 2, 1000),
        'price_range': np.random.randint(0, 4, 1000)
    })
    return data

# Load and preprocess data
data = load_data()
X = data.drop('price_range', axis=1)
y = data['price_range']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Train models
dt_model = DecisionTreeClassifier(random_state=0)
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(random_state=0)
rf_model.fit(X_train, y_train)

svm_model = SVC(random_state=0)
svm_model.fit(X_train, y_train)

# Streamlit UI
st.title("Mobile Price Prediction App")

# User selects the model
model_choice = st.selectbox(
    "Select a Model",
    ["Decision Tree", "Random Forest", "Support Vector Machine (SVM)"]
)

# Input fields for features
st.write("Enter the feature values below:")
features = {
    'battery_power': st.number_input("Battery Power", min_value=500, max_value=2000, step=10),
    'blue': st.selectbox("Bluetooth (0 or 1)", [0, 1]),
    'clock_speed': st.number_input("Clock Speed (GHz)", min_value=0.5, max_value=3.0, step=0.1),
    'dual_sim': st.selectbox("Dual SIM (0 or 1)", [0, 1]),
    'fc': st.number_input("Front Camera (MP)", min_value=0, max_value=20, step=1),
    'four_g': st.selectbox("4G Support (0 or 1)", [0, 1]),
    'int_memory': st.number_input("Internal Memory (GB)", min_value=8, max_value=128, step=1),
    'm_dep': st.number_input("Mobile Depth (cm)", min_value=0.1, max_value=1.5, step=0.1),
    'mobile_wt': st.number_input("Mobile Weight (g)", min_value=80, max_value=250, step=1),
    'n_cores': st.number_input("Number of Cores", min_value=1, max_value=8, step=1),
    'pc': st.number_input("Primary Camera (MP)", min_value=0, max_value=20, step=1),
    'px_height': st.number_input("Pixel Height", min_value=0, max_value=2000, step=10),
    'px_width': st.number_input("Pixel Width", min_value=0, max_value=2000, step=10),
    'ram': st.number_input("RAM (MB)", min_value=256, max_value=8192, step=128),
    'sc_h': st.number_input("Screen Height (cm)", min_value=5, max_value=20, step=1),
    'sc_w': st.number_input("Screen Width (cm)", min_value=0, max_value=15, step=1),
    'talk_time': st.number_input("Talk Time (hours)", min_value=2, max_value=20, step=1),
    'three_g': st.selectbox("3G Support (0 or 1)", [0, 1]),
    'touch_screen': st.selectbox("Touch Screen (0 or 1)", [0, 1]),
    'wifi': st.selectbox("WiFi Support (0 or 1)", [0, 1]),
}

# Prediction button
if st.button("Predict"):
    # Convert input to DataFrame
    input_data = pd.DataFrame([features])
    
    # Scale input data
    scaled_input = scaler.transform(input_data)
    
    # Predict using selected model
    if model_choice == "Decision Tree":
        prediction = dt_model.predict(scaled_input)
    elif model_choice == "Random Forest":
        prediction = rf_model.predict(scaled_input)
    else:
        prediction = svm_model.predict(scaled_input)
    
    # Display result
    st.success(f"The predicted price range is: {int(prediction[0])}")

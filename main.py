import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import folium
from streamlit_folium import folium_static

# Initialize session state variables if they don't exist
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'train_columns' not in st.session_state:
    st.session_state.train_columns = None


# Function to calculate WQI
def calculate_WQI(df):
    # Check if required columns exist in the DataFrame
    required_columns = ['measured values', 'ideal values', 'standard values']
    if not all(col in df.columns for col in required_columns):
        return "Required columns are missing in the uploaded file."
    
    measured_values = df['measured values'].values
    ideal_values = df['ideal values'].values
    standard_values = df['standard values'].values

    qn_values = [((v_en - v_io) / (s_n - v_io)) * 100 if (s_n - v_io) != 0 else 0 for v_en, v_io, s_n in zip(measured_values, ideal_values, standard_values)]
    sum_sn = sum(standard_values)
    k = 1 / sum_sn if sum_sn != 0 else 0
    w_values = [k / s_n if s_n != 0 else 0 for s_n in standard_values]
    
    WQI = sum(qn * w for qn, w in zip(qn_values, w_values)) / sum(w_values) if sum(w_values) != 0 else 0
    return WQI

# Title and introduction
st.title("WQI Prediction")
st.write("This tool estimates the Water Quality Index (WQI) using machine learning algorithms.")

# Read the actual data
df_wqi = pd.read_csv("preprocessed_hydro_data.csv")
df_quality_params = pd.read_csv("preprocessed_water_quality_parameters.csv")

# Calculate WQI for each record in df_wqi
df_wqi['WQI'] = [calculate_WQI(df_quality_params) for _ in range(len(df_wqi))]

# Create DataFrame for map
df_map = df_wqi.copy()
df_map['lat'] = 16.9310
df_map['lon'] = 80.1000

# Option to select ML algorithm
algorithm = st.selectbox("Select the machine learning algorithm:", ["Gradient Boosting", "Decision Tree"])

# Option to select performance metric
metric = st.selectbox("Select the performance metric:", ["RMSE", "MAE", "MSE"])

# Button to update map
if st.button("Update Map"):
    m = folium.Map(location=[16.9310, 80.1000], zoom_start=10)
    for idx, row in df_map.iterrows():
        tooltip_text = "WQI: {}".format(row.get('WQI', 'Not calculated'))
        folium.Marker([row['lat'], row['lon']], tooltip=tooltip_text).add_to(m)
    folium_static(m)

# Button to train the model
if st.button("Train Model"):
    # Drop or encode non-numerical columns here, for example:
    if 'Month' in df_wqi.columns:
        df_wqi = pd.get_dummies(df_wqi, columns=['Month'], drop_first=True)

    X = df_wqi.drop(columns=['WQI'])
    y = df_wqi['WQI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = None  # Initialize model to None
    if algorithm == "Gradient Boosting":
        model = GradientBoostingRegressor()
    elif algorithm == "Decision Tree":
        model = DecisionTreeRegressor()

    if model is not None:  # Only proceed if a valid model is selected
        model.fit(X_train, y_train)
        st.session_state.trained_model = model  # Store the trained model
        st.session_state.train_columns = X_train.columns  # Store the training columns

        y_pred = model.predict(X_test)
        performance = None  # Initialize performance to None
        if metric == "RMSE":
            performance = np.sqrt(mean_squared_error(y_test, y_pred))
        elif metric == "MAE":
            performance = mean_absolute_error(y_test, y_pred)
        elif metric == "MSE":
            performance = mean_squared_error(y_test, y_pred)

        if performance is not None:  # Only proceed if performance is calculated
            st.write(f"The performance of the model based on {metric} is: {performance}")

# File uploader for sample CSV
uploaded_file = st.file_uploader("Upload your sample CSV file containing parameters", type=["csv"], key="unique_key_1")


# Button to upload sample.csv and predict WQI
if st.button("Upload and Predict"):
    if 'trained_model' in st.session_state and 'train_columns' in st.session_state:
        if uploaded_file is not None:
            df_sample = pd.read_csv(uploaded_file)
            
            # Check if encoding (like one-hot encoding) was done during training and do the same here.
            if 'Month' in df_sample.columns:
                df_sample = pd.get_dummies(df_sample, columns=['Month'], drop_first=True)
            
            # Make sure the sample data has the same columns as the training data
            missing_cols = set(st.session_state.train_columns) - set(df_sample.columns)
            for c in missing_cols:
                df_sample[c] = 0  # Add missing columns and set them to 0
            
            # Ensure the order of columns in the sample data is the same as in the training data
            df_sample = df_sample[st.session_state.train_columns]  # replaced X_train.columns
            
            sample_pred = st.session_state.trained_model.predict(df_sample)
            st.write(f"Predicted WQI for the uploaded sample is: {sample_pred}")
        else:
            st.write("Please upload a CSV file.")
    else:
        st.write("Please train the model first by clicking 'Train Model'.")


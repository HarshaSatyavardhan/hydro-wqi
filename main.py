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

# Title and introduction
st.title("WQI Prediction")
st.write("This tool estimates the Water Quality Index (WQI) using machine learning algorithms.")

# Dummy data for WQI
np.random.seed(0)
n = 100
ph = np.random.uniform(6.5, 8.5, n)
conductivity = np.random.uniform(100, 2000, n)
bod = np.random.uniform(1, 50, n)
nitrate = np.random.uniform(0.1, 10, n)
turbidity = np.random.uniform(1, 100, n)
TDS = np.random.uniform(50, 1500, n)
wqi = np.random.uniform(0, 100, n)

# Dummy data for map
lat = np.array([16.9310] * n)
lon = np.array([80.1000] * n)

# Create DataFrame for WQI
df_wqi = pd.DataFrame({
    'ph': ph,
    'conductivity': conductivity,
    'bod': bod,
    'nitrate': nitrate,
    'turbidity': turbidity,
    'TDS': TDS,
    'WQI': wqi
})

# Create DataFrame for map
df_map = df_wqi.copy()
df_map['lat'] = lat
df_map['lon'] = lon

# Option to select ML algorithm
algorithm = st.selectbox("Select the machine learning algorithm:", ["Gradient Boosting", "Decision Tree"])

# Option to select performance metric
metric = st.selectbox("Select the performance metric:", ["RMSE", "MAE", "MSE"])

# Button to update map
if st.button("Update Map"):
    m = folium.Map(location=[16.9310, 80.1000], zoom_start=10)
    for idx, row in df_map.iterrows():
        tooltip_text = f"WQI: {row['WQI']}<br>ph: {row['ph']}<br>Conductivity: {row['conductivity']}<br>BOD: {row['bod']}<br>Nitrate: {row['nitrate']}<br>Turbidity: {row['turbidity']}<br>TDS: {row['TDS']}"
        folium.Marker([row['lat'], row['lon']], tooltip=tooltip_text).add_to(m)
    folium_static(m)

# Button to calculate WQI
if st.button("Calculate WQI"):
    X = df_wqi.drop(columns=['WQI'])
    y = df_wqi['WQI']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if algorithm == "Gradient Boosting":
        model = GradientBoostingRegressor()
    elif algorithm == "Decision Tree":
        model = DecisionTreeRegressor()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if metric == "RMSE":
        performance = np.sqrt(mean_squared_error(y_test, y_pred))
    elif metric == "MAE":
        performance = mean_absolute_error(y_test, y_pred)
    else:
        performance = mean_squared_error(y_test, y_pred)

    st.write(f"The performance of the model ({algorithm}) based on {metric} is: {performance}")

    # Option to download the output
    output_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    st.download_button("Download Output File", data=pd.DataFrame.to_csv(output_df), file_name="wqi_output.csv", mime="text/csv")




# streamlit run main.py
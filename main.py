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

# Function to calculate WQI
def calculate_WQI(df):
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

# Option for user to select input source
option = st.selectbox("Select the source for observed WQI values:", ["Upload File", "Use Calculated WQI"])

# Read the actual data
df_wqi = pd.read_csv("preprocessed_hydro_data.csv")
df_quality_params = pd.read_csv("preprocessed_water_quality_parameters.csv")

# Calculate WQI for each record in df_wqi
if option == "Use Calculated WQI":
    df_wqi['WQI'] = [calculate_WQI(df_quality_params) for _ in range(len(df_wqi))]
elif option == "Upload File":
    # Add file upload logic here
    pass

# Create DataFrame for map
df_map = df_wqi.copy()
df_map['lat'] = 16.9310
df_map['lon'] = 80.1000

# Option to select ML algorithm
algorithm = st.selectbox("Select the machine learning algorithm:", ["Gradient Boosting", "Decision Tree", "Convolutional Neural Network"])

# Option to select performance metric
metric = st.selectbox("Select the performance metric:", ["RMSE", "MAE", "MSE"])

# Button to update map
if st.button("Update Map"):
    m = folium.Map(location=[16.9310, 80.1000], zoom_start=10)
    for idx, row in df_map.iterrows():
        tooltip_text = "WQI: {}".format(row.get('WQI', 'Not calculated'))
        for param in ['pH', 'Conductivity (uS/cm)', 'BOD', 'Nitrates', 'Turbidity', 'TDS']:
            tooltip_text += f"<br><b>{param}:</b> {row.get(param, 'N/A')}"  # Using get() to avoid KeyError
        folium.Marker([row['lat'], row['lon']], tooltip=tooltip_text).add_to(m)
    folium_static(m)

# Button to calculate WQI and train the model
if st.button("Submit"):
    X = df_wqi.drop(columns=['WQI', 'Year', 'Month'])  # Drop columns not used in the model
    y = df_wqi['WQI']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if algorithm == "Gradient Boosting":
        model = GradientBoostingRegressor()
    elif algorithm == "Decision Tree":
        model = DecisionTreeRegressor()
    elif algorithm == "Convolutional Neural Network":
        # Add CNN model code here
        pass

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if metric == "RMSE":
        performance = np.sqrt(mean_squared_error(y_test, y_pred))
    elif metric == "MAE":
        performance = mean_absolute_error(y_test, y_pred)
    else:
        performance = mean_squared_error(y_test, y_pred)

    st.write(f"The performance of the model ({algorithm}) based on {metric} is: {performance}")

    # Convert NumPy array to DataFrame
    output_df = pd.DataFrame({"Predicted WQI": y_pred})

    # Convert DataFrame to CSV format
    csv_data = output_df.to_csv(index=False)

    # Download button
    st.download_button("Download Output File", data=csv_data, file_name="predicted_wqi_output.csv", mime="text/csv")

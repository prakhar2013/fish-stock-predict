import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import calendar
import seaborn as sns

# --- Page Configuration and Styling ---
st.set_page_config(
    page_title="Fish Stock Prediction Dashboard",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# A custom CSS to enhance the professional look with better fonts and spacing
st.markdown("""
<style>
/* ===== Dark Theme Global ===== */
body, .stApp {
    background-color: #121212 !important;
    color: #e0e0e0 !important;
    font-family: "Segoe UI", sans-serif !important;
    margin: 0 !important; /* Ensure no default body margin */
}

/* Remove default padding from Streamlit main container */
.stApp {
    padding-top: 0 !important;
}

/* Adjust margins on the main title for better spacing */
.main-header {
    font-size: 4rem !important; /* Larger font size */
    font-weight: 900 !important;
    color: #00bcd4 !important;
    text-align: center !important;
    margin-top: 0 !important; /* Remove top margin */
    margin-bottom: 10px !important; /* Adjust bottom margin */
    text-shadow: 0px 0px 10px rgba(0, 188, 212, 0.8);
}

.sub-header {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #90caf9 !important;
    margin-top: 25px !important;
    margin-bottom: 5px !important;
}

h3, h4, .stMarkdown h3, .stMarkdown h4 {
    color: #bbdefb !important;
    font-weight: 600 !important;
}

.section-divider {
    border: 0;
    height: 2px;
    background-image: linear-gradient(to right, #cccccc, #004d40, #cccccc);
    margin: 2rem 0;
}

.metric-card {
    background: #1e1e1e !important;
    color: #f5f5f5 !important;
    border-radius: 12px !important;
    padding: 18px !important;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.6) !important;
    text-align: center;
}

.element-container, .css-1ht1j8u {
    background-color: #1e1e1e !important;
    border-radius: 8px !important;
    padding: 12px !important;
}

.stAlert {
    font-weight: 600 !important;
    border-radius: 8px !important;
    padding: 14px !important;
}

.stAlert p, .stAlert div, .stAlert span {
    color: #f5f5f5 !important;
}

.stAlert[data-baseweb="notification"] {
    background-color: #0d47a1 !important;
    border-left: 6px solid #64b5f6 !important;
}

.stAlert.success {
    background-color: #1b5e20 !important;
    border-left: 6px solid #81c784 !important;
}

.stAlert.warning {
    background-color: #f57f17 !important;
    border-left: 6px solid #ffeb3b !important;
    color: #000 !important;
}

.stAlert.error {
    background-color: #b71c1c !important;
    border-left: 6px solid #ef9a9a !important;
}

.app-header-bar {
    background-color: #1e1e1e !important;
    padding: 10px 20px !important; /* Adjusted padding */
    border-bottom: 2px solid #004d40;
    text-align: right;
    font-size: 1rem;
    color: #004d40;
    margin-bottom: 20px !important;
}
.header-link {
    margin-left: 15px !important;
    color: #00bcd4 !important; /* Corrected link color for visibility */
    text-decoration: none;
}

footer {
    text-align: center !important;
    font-size: 14px !important;
    color: #bdbdbd !important;
}
</style>
""", unsafe_allow_html=True)
# --- Model and Data Loading with Caching ---
@st.cache_data
def load_data(path):
    try:
        df = pd.read_csv(path)
        df_filtered = df[df['Species'] == 'Mixed_Small_Pelagics'].copy()
        df_filtered.dropna(inplace=True)
        return df_filtered
    except FileNotFoundError:
        st.error(f"Error: The file '{path}' was not found.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading data: {e}")
        st.stop()

@st.cache_resource
def load_model(path):
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        return None

df_viz = load_data('Bay_of_Bengal_Augmented.csv')
model = load_model('xgb_model.joblib')

if model is None:
    st.error("Error: The model file ('xgb_model.joblib') was not found.")
    st.info("Please run the 'fish_stock_model.py' script first to train and save the model.")
    st.stop()

# --- Custom Header ---
st.markdown("""
<div class="app-header-bar">
    <a href="#" class="header-link">Home</a>
    <a href="https://www.yourcompany.com" target="_blank" class="header-link">About Us</a>
    <a href="mailto:info@yourcompany.com" class="header-link">Contact</a>
</div>
""", unsafe_allow_html=True)

# --- Title Section ---
st.markdown("<p class='main-header'>AI-Powered Fisheries Prediction</p>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #555;'>This dashboard provides predictive insights into fish stocks using historical data and an XGBoost model. Adjust parameters to see real-time predictions.</p>",
    unsafe_allow_html=True
)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# --- Dashboard Section ---
st.markdown("<h2 class='sub-header'>📊 At-a-Glance Dashboard</h2>", unsafe_allow_html=True)

# KPI Metrics with enhanced styling
total_catch = df_viz['Fisheries_Catch_tonnes'].sum()
avg_effort = df_viz['Effort_boat_days'].mean()
start_year, end_year = int(df_viz['Year'].min()), int(df_viz['Year'].max())
avg_annual_catch = df_viz.groupby('Year')['Fisheries_Catch_tonnes'].sum().mean()

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.markdown(f"<div class='metric-card'><h4>Total Historical Catch</h4><p style='font-size: 1.5rem; font-weight: bold;'>{total_catch/1e6:,.2f}M tonnes</p></div>", unsafe_allow_html=True)
with kpi_cols[1]:
    st.markdown(f"<div class='metric-card'><h4>Avg. Annual Catch</h4><p style='font-size: 1.5rem; font-weight: bold;'>{avg_annual_catch/1e3:,.2f}K tonnes</p></div>", unsafe_allow_html=True)
with kpi_cols[2]:
    st.markdown(f"<div class='metric-card'><h4>Avg. Fishing Effort</h4><p style='font-size: 1.5rem; font-weight: bold;'>{avg_effort:,.0f} days</p></div>", unsafe_allow_html=True)
with kpi_cols[3]:
    st.markdown(f"<div class='metric-card'><h4>Data Span</h4><p style='font-size: 1.5rem; font-weight: bold;'>{start_year} - {end_year}</p></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Charts section
chart_cols = st.columns(2)
with chart_cols[0]:
    st.subheader("Fisheries Catch Over Time")
    time_series_df = df_viz.groupby('Year')['Fisheries_Catch_tonnes'].sum()
    st.line_chart(time_series_df, color="#FF4B4B")
    st.info("This chart shows the total annual catch of 'Mixed Small Pelagics' from the training dataset.")

with chart_cols[1]:
    st.subheader("Monthly Catch Distribution")
    monthly_catch = df_viz.groupby('Month')['Fisheries_Catch_tonnes'].sum()
    
    # Use calendar.month_name to get month names
    month_names = [calendar.month_name[i] for i in monthly_catch.index]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    # Corrected method for setting colors from a colormap
    colors = plt.get_cmap('Spectral')(np.linspace(0.1, 0.9, len(monthly_catch)))
    ax.pie(monthly_catch, labels=month_names, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    st.pyplot(fig)
    st.info("This pie chart illustrates the percentage of the total historical catch that occurred in each month.")

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# --- Sidebar for User Inputs ---
st.sidebar.header("Input Parameters")
st.sidebar.info("Adjust the sliders to provide input for the model's prediction.")

def user_input_features():
    # --- MODIFIED: Month input uses selectbox for names ---
    month_names_list = [calendar.month_name[i] for i in range(1, 13)]
    selected_month_name = st.sidebar.selectbox('Month of the Year', month_names_list, index=5)
    month = month_names_list.index(selected_month_name) + 1 # Convert name back to number
    
    effort_boat_days = st.sidebar.slider('Fishing Effort (Boat Days)', 100, 1000, 587)
    sst_c = st.sidebar.slider('Sea Surface Temperature (°C)', 25.0, 30.0, 27.5, 0.1)
    sst_anomaly = st.sidebar.slider('SST Anomaly', -2.0, 2.0, 0.0, 0.1)
    chloro_mg_m3 = st.sidebar.slider('Chlorophyll (mg/m³)', 0.1, 3.0, 1.5, 0.1)
    chl_anomaly = st.sidebar.slider('Chlorophyll Anomaly', -1.5, 1.5, 0.0, 0.1)
    enso_index = st.sidebar.slider('ENSO Index', -2.5, 2.5, 0.0, 0.1)
    iod_index = st.sidebar.slider('IOD Index', -2.0, 2.0, 0.0, 0.1)
    bathymetry_m = st.sidebar.slider('Ocean Depth (Bathymetry in m)', 20, 150, 75)
    dist_to_coast_km = st.sidebar.slider('Distance to Coast (km)', 10, 100, 45)

    sin_month = np.sin(2 * np.pi * month / 12)
    cos_month = np.cos(2 * np.pi * month / 12)

    data = {
        'Effort_boat_days': effort_boat_days,
        'Sea_Surface_Temperature_C': sst_c,
        'SST_anomaly': sst_anomaly,
        'Chlorophyll_mg_m3': chloro_mg_m3,
        'Chl_anomaly': chl_anomaly,
        'ENSO_index': enso_index,
        'IOD_index': iod_index,
        'Bathymetry_m': bathymetry_m,
        'Distance_to_Coast_km': dist_to_coast_km,
        'sin_month': sin_month,
        'cos_month': cos_month
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Prediction and Detailed Exploration Section ---
st.markdown("<h2 class='sub-header'>🔬 Model Prediction & Deep Dive</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.subheader('Your Selected Parameters')
    
    # Create a display DataFrame with month name for better UI
    display_df = input_df.T.rename(columns={0: 'Value'})
    # This code assumes df_viz is not empty and has a 'Month' column
    if 'Month' in df_viz.columns and not df_viz.empty:
        # Get the month number from the sin_month and cos_month values
        angle = np.arctan2(input_df.loc[0, 'sin_month'], input_df.loc[0, 'cos_month'])
        month_num = int(round((angle / (2 * np.pi) * 12 + 12) % 12 + 1))
        # Now, you can use month_num
        display_df.loc['Month of the Year'] = calendar.month_name[month_num]
    
    st.dataframe(display_df, width=500)
    
    st.subheader("Feature Relationships to Catch")
    feature_options = [
        'Effort_boat_days', 'Sea_Surface_Temperature_C', 'Chlorophyll_mg_m3', 'ENSO_index', 'IOD_index'
    ]
    selected_feature = st.selectbox("Choose a feature to analyze:", feature_options)

    st.scatter_chart(
        df_viz,
        x=selected_feature,
        y='Fisheries_Catch_tonnes',
        color="#0068C9"
    )
    st.info(f"This chart illustrates how the **{selected_feature}** relates to the historical **Fisheries Catch**.")


# --- Prediction Logic and Visualizations ---
with col2:
    st.subheader('🤖 AI Model Prediction')
    
    if st.button('Predict Fish Catch', type="primary", use_container_width=True):
        prediction = model.predict(input_df)
        pred_value = prediction[0]

        st.success("Prediction Complete!")
        st.metric(label="Predicted Fisheries Catch for 'Mixed Small Pelagics'", value=f"{pred_value:,.2f} tonnes")

        st.markdown("---")
        st.subheader("Prediction in Context")
        
        hist_min = df_viz['Fisheries_Catch_tonnes'].min()
        hist_max = df_viz['Fisheries_Catch_tonnes'].max()
        hist_avg = df_viz['Fisheries_Catch_tonnes'].mean()

        context_df = pd.DataFrame({
            'Metric': ['Historical Minimum', 'Historical Average', 'Your Prediction', 'Historical Maximum'],
            'Value': [hist_min, hist_avg, pred_value, hist_max]
        })

        st.bar_chart(context_df.set_index('Metric'), color="#0083B8")
        st.info("This chart shows where your prediction falls relative to the historical range of fish catches.")
        
        st.markdown("---")
        st.subheader("📈 How the Model Made its Prediction")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(shap_values, input_df, show=False, color_bar=True, plot_size=[10, 6], title='Feature Importance (SHAP Values)')
        st.pyplot(fig)
        
        st.info("This plot ranks features by their importance in driving the prediction. A positive SHAP value (red) indicates the feature increased the predicted catch, while a negative value (blue) decreased it.")

# --- Custom Footer ---

st.markdown("<br>", unsafe_allow_html=True)

st.subheader("Correlation Heatmap of Features")
# Select only the numerical columns for the heatmap
numerical_df = df_viz[['Effort_boat_days', 'Sea_Surface_Temperature_C',
                       'SST_anomaly', 'Chlorophyll_mg_m3', 'Chl_anomaly',
                       'ENSO_index', 'IOD_index', 'Bathymetry_m',
                       'Distance_to_Coast_km', 'Fisheries_Catch_tonnes']]

# Calculate the correlation matrix
corr_matrix = numerical_df.corr()

# Create the heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, ax=ax)
ax.set_title("Feature Correlation Matrix")
st.pyplot(fig)
st.info("This heatmap shows the correlation between different environmental factors and the historical fish catch. A value close to 1 or -1 indicates a strong relationship.")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="app-footer">
    <p>Powered by Bluenetics | &copy; 2025</p>
</div>
""", unsafe_allow_html=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Streamlit page setup
st.set_page_config(page_title="Eagle Ford EUR Simulator", layout="wide")
st.title("🛢️ Eagle Ford Shale Reservoir Simulation App")
st.markdown("This app simulates horizontal well performance using synthetic geology, completion, and production data.")

# --- Configuration ---
RANDOM_SEED = 42
NUMBER_OF_WELLS = 300
np.random.seed(RANDOM_SEED)

# --- Synthetic Data Generation ---
well_data = pd.DataFrame({
    'Well_ID': [f'EF_{i + 1}' for i in range(NUMBER_OF_WELLS)],
    'Lateral_Length_ft': np.random.choice([5000, 7500, 10000], size=NUMBER_OF_WELLS, p=[0.3, 0.4, 0.3]),
    'Completion_Generation': np.random.choice(['Gen1', 'Gen2', 'Gen3', 'Gen4'], size=NUMBER_OF_WELLS, p=[0.2, 0.3, 0.3, 0.2]),
    'Porosity_pct': np.random.normal(10, 1.5, size=NUMBER_OF_WELLS).clip(5, 15),
    'Permeability_uD': np.random.lognormal(0.2, 0.6, size=NUMBER_OF_WELLS).clip(0.01, 10),
    'Thickness_ft': np.random.uniform(100, 300, size=NUMBER_OF_WELLS),
    'GOR_scf_bbl': np.random.normal(800, 150, size=NUMBER_OF_WELLS).clip(300, 1500),
    'Pressure_psi': np.random.normal(6500, 500, size=NUMBER_OF_WELLS).clip(5000, 8000),
    'Temperature_F': np.random.normal(225, 10, size=NUMBER_OF_WELLS),
    'Water_Cut_pct': np.random.normal(40, 10, size=NUMBER_OF_WELLS).clip(10, 80)
})

# --- Add Frac Stages ---
def estimate_frac_stages(length):
    if length == 5000:
        return np.random.randint(20, 41)
    elif length == 7500:
        return np.random.randint(40, 61)
    elif length == 10000:
        return np.random.randint(60, 81)
    else:
        return np.nan

well_data['Frac_Stages'] = well_data['Lateral_Length_ft'].apply(estimate_frac_stages)

# --- EUR Simulation (Simple Model) ---
def simulate_eur(row):
    base = row['Porosity_pct'] * row['Permeability_uD'] * row['Thickness_ft']
    mult = (1 + 0.1 * int(row['Completion_Generation'][-1]))  # Gen1=1.1x, Gen4=1.4x
    return base * mult / 100  # scale factor for MBO

well_data['EUR_MBO'] = well_data.apply(simulate_eur, axis=1).clip(200, 3000)

# --- Monthly Production Simulation ---
def simulate_monthly_production(eur, b=0.4, di=0.4, months=60):
    qi = eur * (1 - b) / (1 - b * (1 + di * np.arange(months))**(-1 / b))
    return qi.tolist()

well_data['Production_Profile'] = well_data['EUR_MBO'].apply(
    lambda eur: simulate_monthly_production(eur, b=0.4, di=0.4, months=60)
)

# --- Show Columns in DataFrame
st.write("🔍 Available Columns:", well_data.columns.tolist())

# --- Data Preview ---
st.subheader("📋 Sample of Synthetic Well Data")
st.dataframe(well_data.head())

# --- Production Profile Plot (Safely) ---
if 'Production_Profile' in well_data.columns:
    st.subheader("📈 Simulated Production Profile (First Well)")
    st.line_chart(well_data['Production_Profile'].iloc[0])
else:
    st.warning("⚠️ Production_Profile column not found.")

# --- Optional Histograms ---
st.subheader("🔎 Histograms of Key Numerical Features")
num_cols = well_data.select_dtypes(include=np.number).columns.tolist()
fig, ax = plt.subplots(figsize=(15, 10))
well_data[num_cols].hist(bins=20, figsize=(15, 10), ax=ax)
st.pyplot(fig)

# --- Correlation Heatmap ---
st.subheader("📊 Correlation Heatmap")
corr = well_data[num_cols].corr()
fig2, ax2 = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)


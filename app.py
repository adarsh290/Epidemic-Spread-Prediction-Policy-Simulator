import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.express as px

# 1. Page Config
st.set_page_config(page_title="Epidemic Spread Simulator", layout="wide")
st.title("🌍 Global Epidemic Risk Simulator")
st.markdown("Adjust the policy levers on the left to simulate interventions and predict hotspot probabilities 14 days into the future.")

# 2. Load Model & Data
@st.cache_resource
def load_system():
    model = xgb.XGBClassifier()
    model.load_model("xgb_hotspot_model.json")
    df = pd.read_csv('ml_ready_dataset.csv')
    return model, df

model, df = load_system()

# Get the most recent data snapshot for all countries
latest_date = df['Date'].max()
current_data = df[df['Date'] == latest_date].copy()

# 3. Sidebar - The Policy Simulator
st.sidebar.header("Policy Interventions (14-21 Day Lags)")

vax_sim = st.sidebar.slider("Simulate Vaccination Rate (%)", 0.0, 100.0, 50.0)
retail_sim = st.sidebar.slider("Simulate Retail Mobility (% Change)", -100, 50, 0)
transit_sim = st.sidebar.slider("Simulate Transit Mobility (% Change)", -100, 50, 0)
work_sim = st.sidebar.slider("Simulate Workplace Mobility (% Change)", -100, 50, 0)

# 4. Apply Simulations to Data
simulated_data = current_data.copy()
simulated_data['Vax_Lag_21d'] = vax_sim
simulated_data['retail_and_recreation_percent_change_from_baseline_Lag_14d'] = retail_sim
simulated_data['transit_stations_percent_change_from_baseline_Lag_14d'] = transit_sim
simulated_data['workplaces_percent_change_from_baseline_Lag_14d'] = work_sim

features = [
    'retail_and_recreation_percent_change_from_baseline_Lag_14d',
    'transit_stations_percent_change_from_baseline_Lag_14d',
    'workplaces_percent_change_from_baseline_Lag_14d',
    'Vax_Lag_21d'
]

# 5. Generate Predictions
probabilities = model.predict_proba(simulated_data[features])[:, 1]
simulated_data['Hotspot_Probability'] = probabilities * 100

# 6. Render the Risk Map
fig = px.choropleth(
    simulated_data,
    locations="Country",
    locationmode='country names',
    color="Hotspot_Probability",
    hover_name="Country",
    color_continuous_scale=px.colors.sequential.Reds,
    range_color=[0, 100],
    title=f"Predicted Hotspot Probability Map (Simulated)"
)
fig.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'))

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("**Biological Interpretation:** Model optimized for false-negative reduction. Note that vaccination rates heavily interact with local testing infrastructure limits.")

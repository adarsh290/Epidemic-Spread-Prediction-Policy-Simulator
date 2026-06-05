import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def download_data():
    logger.info("1. Downloading & Processing JHU COVID-19 Data...")
    jhu_url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
    jhu_df = pd.read_csv(jhu_url)
    jhu_long = jhu_df.melt(
        id_vars=["Province/State", "Country/Region", "Lat", "Long"],
        var_name="Date",
        value_name="Cases",
    )
    jhu_long = jhu_long.rename(columns={"Country/Region": "Country"})
    jhu_long = jhu_long[["Date", "Country", "Cases"]]
    jhu_long["Cases"] = pd.to_numeric(jhu_long["Cases"], errors="coerce").fillna(0)
    jhu_long = jhu_long.groupby(["Country", "Date"], as_index=False)["Cases"].sum()
    jhu_long["Date"] = pd.to_datetime(
        jhu_long["Date"], format="%m/%d/%y", errors="coerce"
    )
    jhu_long = (
        jhu_long.dropna(subset=["Date"])
        .sort_values(by=["Country", "Date"])
        .reset_index(drop=True)
    )
    jhu_long.to_csv("jhu_cleaned.csv", index=False)

    logger.info("2. Downloading OWID Vaccination Data...")
    owid_url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
    owid_df = pd.read_csv(owid_url)
    owid_df.to_csv("owid_raw.csv", index=False)

    logger.info("3. Downloading Google Mobility Data...")
    google_url = "https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
    google_df = pd.read_csv(google_url, low_memory=False)
    google_df.to_csv("google_raw.csv", index=False)
    logger.info("ALL DOWNLOADS COMPLETE.")


def process_and_merge():
    logger.info("1. Loading the datasets...")
    jhu = pd.read_csv("jhu_cleaned.csv")
    owid = pd.read_csv("owid_raw.csv")
    google = pd.read_csv("google_raw.csv", low_memory=False)

    logger.info("2. Standardizing Columns and Granularity...")
    if "location" in owid.columns:
        owid = owid.rename(columns={"location": "Country", "date": "Date"})
    if "country_region" in google.columns:
        google = google.rename(columns={"country_region": "Country", "date": "Date"})

    google = google[google["sub_region_1"].isna()].copy()

    country_mapping = {
        "US": "United States",
        "Korea, South": "South Korea",
        "Taiwan*": "Taiwan",
        "Czechia": "Czech Republic",
        "Russian Federation": "Russia",
    }
    for df_temp in [jhu, owid, google]:
        df_temp["Country"] = df_temp["Country"].replace(country_mapping)
        df_temp["Date"] = pd.to_datetime(df_temp["Date"])

    logger.info("3. Executing the Merge...")
    owid_cols = ["Country", "Date", "people_fully_vaccinated_per_hundred"]
    google_cols = [
        "Country",
        "Date",
        "retail_and_recreation_percent_change_from_baseline",
        "transit_stations_percent_change_from_baseline",
        "workplaces_percent_change_from_baseline",
    ]

    master_df = pd.merge(jhu, owid[owid_cols], on=["Country", "Date"], how="left")
    master_df = pd.merge(
        master_df, google[google_cols], on=["Country", "Date"], how="left"
    )
    master_df = master_df.sort_values(["Country", "Date"]).reset_index(drop=True)
    master_df.to_csv("master_merged_data.csv", index=False)
    logger.info(f"MERGE COMPLETE. Final dataset size: {master_df.shape}")


def feature_engineering():
    logger.info("1. Calculating Daily New Cases and Target...")
    df = pd.read_csv("master_merged_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Country", "Date"]).reset_index(drop=True)

    df["Daily_New_Cases"] = df.groupby("Country")["Cases"].diff().clip(lower=0)
    df["Cases_7d_avg"] = df.groupby("Country")["Daily_New_Cases"].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    df["Cases_7d_avg_last_week"] = df.groupby("Country")["Cases_7d_avg"].shift(7)
    df["WoW_Growth"] = np.where(
        df["Cases_7d_avg_last_week"] > 0,
        df["Cases_7d_avg"] / df["Cases_7d_avg_last_week"],
        0,
    )
    df["Hotspot"] = np.where((df["WoW_Growth"] > 1.5) & (df["Cases_7d_avg"] > 50), 1, 0)

    logger.info("2. Engineering Biological Lags...")
    mobility_cols = [col for col in df.columns if "percent_change" in col]
    for col in mobility_cols:
        df[f"{col}_Lag_14d"] = df.groupby("Country")[col].shift(14)
    if "people_fully_vaccinated_per_hundred" in df.columns:
        df["Vax_Lag_21d"] = df.groupby("Country")[
            "people_fully_vaccinated_per_hundred"
        ].shift(21)

    df.to_csv("ml_ready_dataset.csv", index=False)
    logger.info("ML-Ready Dataset Saved.")


def train_model():
    logger.info("1. Loading and Cleaning Data for ML...")
    df = pd.read_csv("ml_ready_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    if "Vax_Lag_21d" in df.columns:
        df["Vax_Lag_21d"] = df["Vax_Lag_21d"].fillna(0)

    features = [
        "retail_and_recreation_percent_change_from_baseline_Lag_14d",
        "transit_stations_percent_change_from_baseline_Lag_14d",
        "workplaces_percent_change_from_baseline_Lag_14d",
        "Vax_Lag_21d",
    ]
    target = "Hotspot"
    ml_df = df.dropna(subset=features + [target]).copy().sort_values("Date")

    split_idx = int(len(ml_df) * 0.8)
    train, test = ml_df.iloc[:split_idx], ml_df.iloc[split_idx:]
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    logger.info("2. Training XGBoost Classifier...")
    scale_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric="auc",
    )
    model.fit(X_train, y_train)

    logger.info("3. Evaluating Model...")
    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    logger.info(f"ROC-AUC Score: {roc_auc:.4f}")

    model.save_model("xgb_hotspot_model.json")
    logger.info("Model saved as 'xgb_hotspot_model.json'")


def generate_shap_summary():
    logger.info("Generating SHAP summary...")
    model = xgb.XGBClassifier()
    model.load_model("xgb_hotspot_model.json")
    df = pd.read_csv("ml_ready_dataset.csv")
    features = [
        "retail_and_recreation_percent_change_from_baseline_Lag_14d",
        "transit_stations_percent_change_from_baseline_Lag_14d",
        "workplaces_percent_change_from_baseline_Lag_14d",
        "Vax_Lag_21d",
    ]
    ml_df = df.dropna(subset=features + ["Hotspot"]).copy().sort_values("Date")
    X_train = ml_df.iloc[: int(len(ml_df) * 0.8)][features]

    X_sample = shap.utils.sample(X_train, 5000)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig("shap_summary.png", bbox_inches="tight")
    logger.info("SHAP summary saved as 'shap_summary.png'")


if __name__ == "__main__":
    # download_data() # Commented out by default to avoid accidental large downloads
    # process_and_merge()
    # feature_engineering()
    # train_model()
    # generate_shap_summary()
    logger.info(
        "Pipeline script ready. Uncomment functions in __main__ to run specific steps."
    )

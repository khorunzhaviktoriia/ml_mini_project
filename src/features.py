import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.data_prep import (
    DATA_PATH, TARGET, FEATURES, NUMERIC_COLUMNS, CATEGORICAL_FEATURES,
    UNAVAILABLE_AT_9AM, clean
)

if set(FEATURES) & set(UNAVAILABLE_AT_9AM):
    raise ValueError("Update src/data_prep.py to the 09:00 feature list first")

DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


LAG_COLS = [
    "Rainfall", "RainToday", "Humidity9am", "Pressure9am",
    "Humidity3pm", "Pressure3pm", "Cloud3pm", "Temp3pm",
    "WindGustSpeed", "Sunshine", "MaxTemp",
]
LAGS = (1, 2, 3)
WINDOWS = (3, 7, 14, 30)

COORDS = {
    "Adelaide": (-34.93, 138.60), "Albany": (-35.02, 117.88), "Albury": (-36.08, 146.92),
    "AliceSprings": (-23.70, 133.88), "BadgerysCreek": (-33.88, 150.75), "Ballarat": (-37.56, 143.85),
    "Bendigo": (-36.76, 144.28), "Brisbane": (-27.47, 153.03), "Cairns": (-16.92, 145.77),
    "Canberra": (-35.28, 149.13), "Cobar": (-31.50, 145.84), "CoffsHarbour": (-30.30, 153.11),
    "Dartmoor": (-37.92, 141.27), "Darwin": (-12.46, 130.84), "GoldCoast": (-28.02, 153.40),
    "Hobart": (-42.88, 147.33), "Katherine": (-14.47, 132.26), "Launceston": (-41.43, 147.14),
    "Melbourne": (-37.81, 144.96), "MelbourneAirport": (-37.67, 144.84), "Mildura": (-34.19, 142.16),
    "Moree": (-29.47, 149.84), "MountGambier": (-37.83, 140.78), "MountGinini": (-35.53, 148.77),
    "Newcastle": (-32.93, 151.78), "Nhil": (-36.33, 141.65), "NorahHead": (-33.28, 151.57),
    "NorfolkIsland": (-29.04, 167.95), "Nuriootpa": (-34.47, 139.00), "PearceRAAF": (-31.67, 116.02),
    "Penrith": (-33.75, 150.69), "Perth": (-31.95, 115.86), "PerthAirport": (-31.94, 115.97),
    "Portland": (-38.34, 141.60), "Richmond": (-33.60, 150.75), "Sale": (-38.10, 147.07),
    "SalmonGums": (-32.98, 121.64), "Sydney": (-33.87, 151.21), "SydneyAirport": (-33.94, 151.18),
    "Townsville": (-19.26, 146.82), "Tuggeranong": (-35.42, 149.09), "Uluru": (-25.34, 131.04),
    "WaggaWagga": (-35.12, 147.37), "Walpole": (-34.98, 116.73), "Watsonia": (-37.71, 145.08),
    "Williamtown": (-32.80, 151.84), "Witchcliffe": (-34.03, 115.10), "Wollongong": (-34.43, 150.89),
    "Woomera": (-31.20, 136.82),
}

DERIVED_FEATURES = (
    ["DoySin", "DoyCos", "Lat", "Lon",
     "WindDir9amU", "WindDir9amV", "TempHumidityInteraction9am",
     "PressureTrend18h", "HumidityTrend18h", "TempTrend18h"]
    + [f"{col}_lag{k}" for col in LAG_COLS for k in LAGS]
    + [f"{col}_diff1" for col in ("Rainfall", "Humidity9am", "Pressure9am")]
    + [f"Rain_sum{w}" for w in WINDOWS]
    + [f"RainDays{w}" for w in WINDOWS]
    + [f"Hum3pm_mean{w}_prev" for w in WINDOWS]
    + [f"Press3pm_mean{w}_prev" for w in WINDOWS]
    + ["DryStreak"]
)


def add_features(df):
    df = df.sort_values(["Location", "Date"]).reset_index(drop=True).copy()
    doy = df["Date"].dt.dayofyear
    df["DoySin"] = np.sin(2 * np.pi * doy / 365.25)
    df["DoyCos"] = np.cos(2 * np.pi * doy / 365.25)
    df["Lat"] = df["Location"].map({k: v[0] for k, v in COORDS.items()})
    df["Lon"] = df["Location"].map({k: v[1] for k, v in COORDS.items()})

    angle = df["WindDir9am"].map({d: i * np.pi / 8 for i, d in enumerate(DIRS)})
    df["WindDir9amU"] = df["WindSpeed9am"] * np.sin(angle)
    df["WindDir9amV"] = df["WindSpeed9am"] * np.cos(angle)
    df["TempHumidityInteraction9am"] = df["Temp9am"] * df["Humidity9am"]

    full = (df.set_index(["Location", "Date"])
              .groupby(level=0, group_keys=False)
              .apply(lambda part: part.droplevel(0).asfreq("D").assign(Location=part.index[0][0])))
    full = full.reset_index().set_index(["Location", "Date"])
    g = full.groupby(level=0)
    generated = {}

    for col in LAG_COLS:
        for k in LAGS:
            generated[f"{col}_lag{k}"] = g[col].shift(k)

    for col in ("Rainfall", "Humidity9am", "Pressure9am"):
        generated[f"{col}_diff1"] = full[col] - generated[f"{col}_lag1"]

    generated["PressureTrend18h"] = full["Pressure9am"] - generated["Pressure3pm_lag1"]
    generated["HumidityTrend18h"] = full["Humidity9am"] - generated["Humidity3pm_lag1"]
    generated["TempTrend18h"] = full["Temp9am"] - generated["Temp3pm_lag1"]

    for w in WINDOWS:
        generated[f"Rain_sum{w}"] = g["Rainfall"].transform(
            lambda s: s.rolling(w, min_periods=1).sum())
        generated[f"RainDays{w}"] = g["RainToday"].transform(
            lambda s: s.rolling(w, min_periods=1).mean())


        for col, stem in (("Humidity3pm", "Hum3pm"),
                          ("Pressure3pm", "Press3pm")):
            generated[f"{stem}_mean{w}_prev"] = g[col].transform(
                lambda s: s.shift(1).rolling(w, min_periods=1).mean())


    is_dry = full["RainToday"].eq(0)
    generated["DryStreak"] = is_dry.groupby(level=0).transform(
        lambda s: s.groupby((~s).cumsum()).cumsum()
    ).where(full["RainToday"].notna())
    df = df.join(pd.DataFrame(generated, index=full.index),
                 on=["Location", "Date"])
    return df.sort_values(["Date", "Location"]).reset_index(drop=True)


def feature_list(df=None):
    return list(FEATURES) + DERIVED_FEATURES


def get_feature_x_y(part):
    return part[feature_list(part)], part[TARGET]


def build_feature_preprocessor():
    return ColumnTransformer([
        ("numeric_base", SimpleImputer(strategy="median", add_indicator=True),
         list(NUMERIC_COLUMNS)),
        ("numeric_derived", SimpleImputer(strategy="median"),
         DERIVED_FEATURES),
        ("rain_today", SimpleImputer(strategy="constant", fill_value=0,
                                     add_indicator=True), ["RainToday"]),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL_FEATURES),
    ], sparse_threshold=0)


def load_features(path=DATA_PATH):
    raw = pd.read_csv(path)
    raw["HasTarget"] = raw[TARGET].notna()
    raw[TARGET] = raw[TARGET].fillna("No")
    df = add_features(clean(raw))
    return df.loc[df["HasTarget"]].drop(columns="HasTarget").reset_index(drop=True)

import numpy as np
import pandas as pd

from src.data_prep import DATA_PATH, TARGET, clean

DIRS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
LAG_COLS = ["Rainfall", "Humidity3pm", "Pressure3pm", "Cloud3pm", "Temp3pm", "WindGustSpeed", "Sunshine", "RainToday"]
NATIONAL_COLS = ["Humidity3pm", "Pressure3pm", "Rainfall", "Cloud3pm", "WindGustSpeed", "RainToday"]


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
NEIGHBOR_COLS = [
    "Rainfall", "RainToday", "Humidity3pm", "Pressure3pm", "Cloud3pm", "WindGustSpeed", "PressureDelta",
    "Rainfall_lag1", "Humidity3pm_lag1", "Pressure3pm_diff1", "Rain_sum3", "Hum_anom7", "Press_anom7",
    "Sunshine", "Temp3pm",
]


def neighbor_weights(locations):
    lat = np.radians([COORDS[l][0] for l in locations])
    lon = np.radians([COORDS[l][1] for l in locations])
    a = (np.sin((lat[:, None] - lat[None]) / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None]) * np.sin((lon[:, None] - lon[None]) / 2) ** 2)
    dist = 2 * 6371 * np.arcsin(np.sqrt(a))
    near = np.exp(-(dist / 300) ** 2)
    west = np.exp(-(dist / 700) ** 2) * (lon[None] < lon[:, None] - 0.01)
    np.fill_diagonal(near, 0)
    return near, west


def add_neighbor_features(df):
    locations = sorted(df["Location"].unique())
    near, west = neighbor_weights(locations)
    new = []
    for col in NEIGHBOR_COLS:
        wide = df.pivot(index="Date", columns="Location", values=col).reindex(columns=locations)
        vals, mask = np.nan_to_num(wide.to_numpy()), wide.notna().to_numpy().astype(float)
        for name, w in (("Near", near), ("West", west)):
            num, den = vals @ w.T, mask @ w.T
            out = pd.DataFrame(np.where(den > 1e-3, num / np.maximum(den, 1e-9), np.nan),
                               index=wide.index, columns=locations)
            new.append(out.stack(future_stack=True).rename(f"{name}_{col}"))
    new = pd.concat(new, axis=1)
    new.index.names = ["Date", "Location"]
    df = df.join(new, on=["Date", "Location"])
    df["Lat"] = df["Location"].map(lambda l: COORDS[l][0])
    df["Lon"] = df["Location"].map(lambda l: COORDS[l][1])
    df["Hum_vsNear"] = df["Humidity3pm"] - df["Near_Humidity3pm"]
    df["Press_vsNear"] = df["Pressure3pm"] - df["Near_Pressure3pm"]
    return df


def add_features(df):
    df = df.sort_values(["Location", "Date"]).reset_index(drop=True)
    doy = df["Date"].dt.dayofyear
    df["DoySin"] = np.sin(2 * np.pi * doy / 365.25)
    df["DoyCos"] = np.cos(2 * np.pi * doy / 365.25)
    df["Year"] = df["Date"].dt.year

    df["PressureDelta"] = df["Pressure3pm"] - df["Pressure9am"]
    df["HumidityDelta"] = df["Humidity3pm"] - df["Humidity9am"]
    df["TempDelta"] = df["Temp3pm"] - df["Temp9am"]
    df["TempRange"] = df["MaxTemp"] - df["MinTemp"]
    df["CloudDelta"] = df["Cloud3pm"] - df["Cloud9am"]
    df["WindDelta"] = df["WindSpeed3pm"] - df["WindSpeed9am"]
    df["DewGap3pm"] = df["Temp3pm"] * (100 - df["Humidity3pm"]) / 100

    for col in ["WindGustDir", "WindDir9am", "WindDir3pm"]:
        angle = df[col].map({d: i * np.pi / 8 for i, d in enumerate(DIRS)})
        df[col + "Sin"] = np.sin(angle)
        df[col + "Cos"] = np.cos(angle)
        speed = df["WindGustSpeed"] if col == "WindGustDir" else df["WindSpeed" + col[-3:]]
        df[col + "U"] = speed * np.sin(angle)
        df[col + "V"] = speed * np.cos(angle)

    full = (df.set_index(["Location", "Date"])
              .groupby(level=0, group_keys=False)
              .apply(lambda g: g.droplevel(0).asfreq("D").assign(Location=g.index[0][0])))
    full = full.reset_index().set_index(["Location", "Date"])
    g = full.groupby(level=0)
    new = {}
    for col in LAG_COLS:
        for lag in (1, 2, 3):
            new[f"{col}_lag{lag}"] = g[col].shift(lag)
        new[f"{col}_diff1"] = full[col] - new[f"{col}_lag1"]
    for w in (3, 7, 14, 30):
        new[f"Rain_sum{w}"] = g["Rainfall"].transform(lambda s: s.rolling(w, min_periods=1).sum())
        new[f"RainDays{w}"] = g["RainToday"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        new[f"Hum_mean{w}"] = g["Humidity3pm"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        new[f"Press_mean{w}"] = g["Pressure3pm"].transform(lambda s: s.rolling(w, min_periods=1).mean())
    dry = full["RainToday"].fillna(0).eq(0)
    run = dry.groupby(level=0).transform(lambda s: s.groupby((~s).cumsum()).cumsum())
    new["DryStreak"] = run
    new = pd.DataFrame(new, index=full.index)
    new["Hum_anom7"] = full["Humidity3pm"] - new["Hum_mean7"]
    new["Press_anom7"] = full["Pressure3pm"] - new["Press_mean7"]
    df = df.join(new, on=["Location", "Date"])

    nat = df.groupby("Date")[NATIONAL_COLS].agg(["mean", "std"])
    nat.columns = [f"Nat_{c}_{s}" for c, s in nat.columns]
    df = df.join(nat, on="Date")
    for c in ["Humidity3pm", "Pressure3pm"]:
        df[f"{c}_vsNat"] = df[c] - df[f"Nat_{c}_mean"]

    df = add_neighbor_features(df)
    return df.sort_values(["Date", "Location"]).reset_index(drop=True)


def feature_list(df):
    drop = {"Date", "RainTomorrow", "Year", "HasTarget"}
    return [c for c in df.columns if c not in drop]


def load_features(path=DATA_PATH):
    raw = pd.read_csv(path)
    raw["HasTarget"] = raw[TARGET].notna()
    raw[TARGET] = raw[TARGET].fillna("No")
    df = add_features(clean(raw))
    return df[df["HasTarget"]].reset_index(drop=True)

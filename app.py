"""Dataset loading, schema auto-detection, and a clearly-labelled synthetic demo dataset."""
import os, re, io, zipfile
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# role -> (exact normalised names, substrings)
ROLES = {
    "rainfall": (["rainfall", "rain", "precipitation", "precip", "rainfallmm"], ["rainfall", "precip", "rain"]),
    "temperature": (["temperature", "temp", "tempc"], ["temp"]),
    "humidity": (["humidity", "rh"], ["humid"]),
    "wind": (["windspeed", "wind"], ["wind"]),
    "pressure": (["pressure", "atmosphericpressure", "pres"], ["pressure"]),
    "water_level": (["waterlevel", "riverlevel", "river", "gauge"], ["waterlevel", "riverlevel", "river", "gauge"]),
    "soil_moisture": (["soilmoisture", "soil"], ["soilmoist", "soil"]),
    "lat": (["latitude", "lat"], []),
    "lon": (["longitude", "lon", "lng", "long"], []),
    "elevation": (["elevation", "altitude", "elev"], ["elevation", "altitude"]),
    "magnitude": (["magnitude", "mag"], ["magnitude"]),
    "depth": (["depth", "depthkm"], ["depth"]),
    "date": (["date", "datetime", "time", "timestamp", "year"], ["date", "time"]),
}
FRIENDLY = {"rainfall": "Rainfall", "temperature": "Temperature", "humidity": "Humidity",
            "wind": "Wind Speed", "pressure": "Pressure", "water_level": "Water Level",
            "soil_moisture": "Soil Moisture", "elevation": "Elevation"}


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def read_file(name, raw):
    low = name.lower()
    if low.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw), low_memory=False)
    if low.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    return None


def load_all_tables(data_dir=DATA_DIR):
    """Return {filename: DataFrame} for every csv/xlsx in data/ (also inside .zip files)."""
    tables = {}
    if not os.path.isdir(data_dir):
        return tables
    for fn in sorted(os.listdir(data_dir)):
        path = os.path.join(data_dir, fn)
        try:
            if fn.lower().endswith(".zip"):
                with zipfile.ZipFile(path) as z:
                    for inner in z.namelist():
                        if inner.endswith("/") or "__MACOSX" in inner:
                            continue
                        df = read_file(inner, z.read(inner))
                        if df is not None:
                            tables[f"{fn}::{inner}"] = df
            elif fn.lower().endswith((".csv", ".xlsx", ".xls")):
                with open(path, "rb") as f:
                    df = read_file(fn, f.read())
                if df is not None:
                    tables[fn] = df
        except Exception:
            continue
    return tables


def detect_schema(df):
    """Map semantic roles to actual column names (only if they exist)."""
    cols = {_norm(c): c for c in df.columns}
    schema, used = {}, set()
    for role, (exact, subs) in ROLES.items():
        hit = next((cols[e] for e in exact if e in cols and cols[e] not in used), None)
        if hit is None:
            hit = next((c for n, c in cols.items() if c not in used and any(s in n for s in subs)), None)
        if hit is not None:
            schema[role] = hit
            used.add(hit)
    return schema


def detect_flood_target(df):
    cands = [c for c in df.columns if "flood" in _norm(c)]
    return cands[0] if cands else None


def to_binary_target(s):
    """Convert 0/1, yes/no, Low/High, probability 0-1 or continuous labels to 0/1."""
    if s.dtype == object or str(s.dtype).startswith("category"):
        pos = {"1", "yes", "true", "high", "extreme", "severe", "flood", "y", "very high"}
        return s.astype(str).str.strip().str.lower().isin(pos).astype(int)
    s = pd.to_numeric(s, errors="coerce")
    u = s.dropna().unique()
    if len(u) <= 2:
        return (s == s.max()).astype(int)
    thr = 0.5 if (s.min() >= 0 and s.max() <= 1) else s.median()
    return (s > thr).astype(int)


def describe_dataset(df):
    num = df.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in df.columns if c not in num]
    miss = df.isna().sum()
    return {"rows": len(df), "cols": df.shape[1], "numeric": num, "categorical": cat,
            "missing": miss[miss > 0].sort_values(ascending=False)}


def make_demo_datasets(n=3000, seed=42):
    """SYNTHETIC data for demonstration only. Flood label comes from a made-up rule + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    doy = dates.dayofyear.values
    season = np.sin(2 * np.pi * (doy - 150) / 365)
    rain = np.clip(rng.gamma(1.5, 35, n) * (1.2 + season), 0, 400)
    temp = 27 + 6 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 2, n)
    hum = np.clip(55 + 0.12 * rain + rng.normal(0, 8, n), 20, 100)
    wind = np.clip(rng.gamma(2, 6, n), 0, 90)
    pres = 1010 - 0.03 * rain + rng.normal(0, 4, n)
    soil = np.clip(30 + 0.18 * rain + rng.normal(0, 8, n), 5, 100)
    water = np.clip(1.5 + 0.018 * rain + 0.02 * soil + rng.normal(0, 0.6, n), 0.2, 12)
    elev = rng.uniform(5, 400, n)
    lat = rng.uniform(18, 28, n)
    lon = rng.uniform(72, 84, n)
    z = 0.025 * (rain - 110) + 1.1 * (water - 3.5) + 0.04 * (soil - 55) - 0.004 * (elev - 150)
    flood = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    flood_df = pd.DataFrame({"Date": dates, "Rainfall": rain.round(1), "Temperature": temp.round(1),
                             "Humidity": hum.round(1), "WindSpeed": wind.round(1), "Pressure": pres.round(1),
                             "WaterLevel": water.round(2), "SoilMoisture": soil.round(1),
                             "Elevation": elev.round(0), "Latitude": lat.round(3), "Longitude": lon.round(3),
                             "Flood": flood})
    centers = np.array([[23.5, 70.5], [30.5, 79.0], [26.0, 92.0], [20.0, 74.0]])
    k = rng.choice(len(centers), 1200, p=[0.3, 0.3, 0.3, 0.1])
    qlat = centers[k, 0] + rng.normal(0, 1.2, 1200)
    qlon = centers[k, 1] + rng.normal(0, 1.2, 1200)
    mag = np.clip(rng.exponential(0.8, 1200) + 3.0, 3, 8)
    depth = np.clip(rng.gamma(2, 12, 1200), 2, 200)
    qdate = pd.to_datetime("2000-01-01") + pd.to_timedelta(rng.integers(0, 365 * 25, 1200), unit="D")
    quake_df = pd.DataFrame({"Date": qdate, "Latitude": qlat.round(3), "Longitude": qlon.round(3),
                             "Magnitude": mag.round(1), "Depth": depth.round(1)}).sort_values("Date")
    return flood_df, quake_df


def pick_tables(tables):
    """Pick flood table (has a flood-like target) and quake catalogue (magnitude+lat+lon)."""
    flood_name = quake_name = None
    best = -1
    for name, df in tables.items():
        if detect_flood_target(df) is not None and df.select_dtypes(include=np.number).shape[1] >= 3:
            if len(df) > best:
                flood_name, best = name, len(df)
    for name, df in tables.items():
        if {"magnitude", "lat", "lon"} <= set(detect_schema(df)):
            quake_name = name
            break
    return flood_name, quake_name


# ----------------------------------------------------------------------------
# Weather-only datasets (no flood label): derive a flood-PRECURSOR target
# ----------------------------------------------------------------------------
def engineer_weather(df):
    """Dataset has weather but no flood label -> build derived features + a proxy target:
    'heavy rainfall expected NEXT day' (>= 90th percentile of daily rainfall). Returns (df, meta) or None."""
    s = detect_schema(df)
    if "rainfall" not in s:
        return None
    d = df.copy()
    dcol = s.get("date")
    if dcol:
        d[dcol] = pd.to_datetime(d[dcol], errors="coerce")
        d = d.sort_values(dcol).reset_index(drop=True)
    rain = pd.to_numeric(d[s["rainfall"]], errors="coerce").fillna(0)
    d["rain_prev_3d"] = rain.shift(1).rolling(3, min_periods=1).sum().fillna(0)
    d["rain_prev_14d"] = rain.shift(1).rolling(14, min_periods=1).sum().fillna(0)
    norm = {_norm(c): c for c in d.columns}
    tmax = next((c for n, c in norm.items() if n in ("tempmax", "tmax", "maxtemp", "temperaturemax")), None)
    tmin = next((c for n, c in norm.items() if n in ("tempmin", "tmin", "mintemp", "temperaturemin")), None)
    derived = ["rain_prev_3d", "rain_prev_14d"]
    if tmax and tmin:
        d["temp_range"] = d[tmax] - d[tmin]
        derived.append("temp_range")
    if dcol:
        m = d[dcol].dt.month
        d["month_sin"] = np.sin(2 * np.pi * m / 12)
        d["month_cos"] = np.cos(2 * np.pi * m / 12)
        derived += ["month_sin", "month_cos"]
    cat_dummies = {}
    for c in list(d.columns):
        if c == dcol or d[c].dtype != object and not str(d[c].dtype).startswith("str"):
            continue
        if d[c].nunique() <= 12:
            dm = pd.get_dummies(d[c], prefix=c).astype(int)
            cat_dummies[c] = list(dm.columns)
            d = pd.concat([d, dm], axis=1)
    thr = float(max(rain.quantile(0.9), 0.1))
    target = "heavy_rain_next_day"
    d[target] = (rain.shift(-1) >= thr).astype(float)
    d.loc[d.index[-1], target] = np.nan
    meta = {"rain_col": s["rainfall"], "tmax": tmax, "tmin": tmin, "wind": s.get("wind"), "date": dcol,
            "threshold": thr, "target": target, "derived": derived, "cat_dummies": cat_dummies}
    return d, meta


def pick_weather_table(tables):
    for name, df in tables.items():
        s = detect_schema(df)
        if "rainfall" in s and "magnitude" not in s:
            return name
    return None


def feature_status(df_raw, meta, has_quake):
    """Table: which requested features are available / derived / missing."""
    s = detect_schema(df_raw)
    rows = []
    want = [("Rainfall / precipitation", "rainfall"), ("Temperature", "temperature"), ("Humidity", "humidity"),
            ("Wind speed", "wind"), ("Atmospheric pressure", "pressure"), ("River / water level", "water_level"),
            ("Soil moisture", "soil_moisture"), ("Latitude", "lat"), ("Longitude", "lon"), ("Elevation", "elevation")]
    for label, role in want:
        if role in s:
            rows.append((label, "Available", s[role]))
        elif role == "soil_moisture" and meta:
            rows.append((label, "Derived (proxy)", "rain_prev_14d = antecedent rainfall (wetness proxy)"))
        elif role == "water_level" and meta:
            rows.append((label, "Derived (proxy)", "rain_prev_3d / rain_prev_14d = accumulated rainfall (runoff proxy)"))
        else:
            rows.append((label, "Missing", "-"))
    rows.append(("Flood label (target)", "Derived (proxy)" if meta else "Missing",
                 f"heavy_rain_next_day (next-day rainfall >= {meta['threshold']:.1f}) - NOT observed floods" if meta else "-"))
    rows.append(("Earthquake / seismic data", "Available" if has_quake else "Missing",
                 "catalogue loaded" if has_quake else "Upload a catalogue (magnitude, latitude, longitude, depth)"))
    return pd.DataFrame(rows, columns=["Requested feature", "Status", "Source / note"])

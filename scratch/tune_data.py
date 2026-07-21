import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, f1_score
import warnings
warnings.filterwarnings('ignore')

# 1. Generate clean synthetic data
np.random.seed(42)
start_date = datetime(2025, 10, 1)
days = 180
dates = [start_date + timedelta(days=i) for i in range(days)]

services = [
    ("aws", "Virtual Machines", "compute", "eng-alpha", "production", "us-east"),
    ("azure", "Managed Database", "database", "data-eng", "production", "eu-west"),
    ("gcp", "Object Storage", "storage", "analytics", "production", "us-central")
]

records = []
labels = []
label_id = 1

for date in dates:
    is_weekend = date.weekday() >= 5
    for prov, srv, cat, team, env, reg in services:
        # Base cost + weekend dip + noise
        base = 100 if srv == "Virtual Machines" else (200 if srv == "Managed Database" else 50)
        cost = base * (0.6 if is_weekend else 1.0)
        cost += np.random.normal(0, base * 0.05) # 5% noise
        
        is_anomaly = False
        
        # Inject anomalies (~5% chance)
        if np.random.rand() < 0.05 and not is_weekend:
            multiplier = np.random.uniform(3.0, 5.0)
            cost *= multiplier
            is_anomaly = True
            
            # Create label entry
            labels.append({
                "id": f"ANO-{label_id:03d}",
                "provider": prov,
                "service": srv,
                "team": team,
                "start_date": date.strftime("%Y-%m-%d"),
                "end_date": date.strftime("%Y-%m-%d"),
                "type": "spike",
                "description": f"Injected synthetic spike {multiplier:.1f}x"
            })
            label_id += 1
            
        records.append({
            "date": date,
            "provider": prov,
            "service": srv,
            "category": cat,
            "team": team,
            "environment": env,
            "region": reg,
            "cost_usd": max(0, cost),
            "is_anomaly": 1 if is_anomaly else 0
        })

df = pd.DataFrame(records)
print(f"Generated {len(df)} rows, {df['is_anomaly'].sum()} anomalies")

# 2. Engineer features
df = df.sort_values(["provider", "service", "date"]).reset_index(drop=True)
grp = ["provider", "service"]
df["roll7_mean"] = df.groupby(grp)["cost_usd"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).mean())
df["roll7_std"] = df.groupby(grp)["cost_usd"].transform(lambda x: x.shift(1).rolling(7, min_periods=1).std().fillna(0))
df["cost_vs_roll7"] = df["cost_usd"] / df["roll7_mean"].fillna(1)
df["dayofweek"] = df["date"].dt.dayofweek

df = df.dropna().reset_index(drop=True)

# 3. Model Training
FEATURES = ["cost_vs_roll7", "dayofweek"] # Optimized features
X = df[FEATURES]
y = df["is_anomaly"].apply(lambda x: -1 if x == 1 else 1)

split = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

contamination = float(np.clip((y_train==-1).sum() / len(y_train), 0.01, 0.5))

iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
iso.fit(X_train)
y_pred = iso.predict(X_test)

f1 = f1_score(y_test, y_pred, pos_label=-1)
print(f"F1 Score: {f1:.4f}")
print(classification_report(y_test, y_pred, labels=[-1, 1], target_names=["Anomaly", "Normal"]))

import json
import os

notebook_dict = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# FinOps Anomaly Detection: ML Pipeline & Analysis\n",
    "\n",
    "This notebook demonstrates a complete Machine Learning pipeline for anomaly detection\n",
    "using **Isolation Forest** on the full **~60k-row** processed billing dataset.\n",
    "We load normalized multi-cloud billing data, engineer temporal features, train\n",
    "an Isolation Forest model, and evaluate performance with real ground-truth labels."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import json\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "from sklearn.ensemble import IsolationForest\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.metrics import (\n",
    "    classification_report, confusion_matrix,\n",
    "    precision_score, recall_score, f1_score\n",
    ")\n",
    "from sklearn.metrics import mean_absolute_error, mean_squared_error\n",
    "\n",
    "sns.set_theme(style=\"whitegrid\")\n",
    "plt.rcParams[\"figure.figsize\"] = (12, 6)\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load Processed Data & Ground-Truth Labels\n",
    "\n",
    "We load directly from `data/processed/daily_billing.csv` — the full ~60k-row\n",
    "normalised dataset produced by the Phase-1 normalizer — and from\n",
    "`data/raw/anomaly_labels.json` which contains the six injected anomaly windows."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Paths ─────────────────────────────────────────────────────────────────────\n",
    "CSV_PATH    = \"data/processed/daily_billing.csv\"\n",
    "LABELS_PATH = \"data/raw/anomaly_labels.json\"\n",
    "\n",
    "# ── Load billing data ─────────────────────────────────────────────────────────\n",
    "df = pd.read_csv(CSV_PATH, parse_dates=[\"date\"])\n",
    "\n",
    "print(f\"Loaded {len(df):,} billing rows\")\n",
    "print(f\"Date range : {df['date'].min().date()} -> {df['date'].max().date()}\")\n",
    "print(f\"Providers  : {sorted(df['provider'].unique())}\")\n",
    "print(f\"Services   : {df['service'].nunique()} unique\")\n",
    "\n",
    "display(df.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Create Ground-Truth Labels\n",
    "\n",
    "The JSON labels store the *raw* service names (e.g. `Amazon EC2`).\n",
    "The processed CSV already maps those to canonical names (e.g. `Virtual Machines`).\n",
    "We apply the same `SERVICE_CANONICAL` mapping used by `normalizer.py` so the join works correctly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Same mapping as normalizer.py\n",
    "SERVICE_CANONICAL = {\n",
    "    \"Amazon EC2\":               \"Virtual Machines\",\n",
    "    \"Virtual Machines\":         \"Virtual Machines\",\n",
    "    \"Compute Engine\":           \"Virtual Machines\",\n",
    "    \"Amazon EKS\":               \"Kubernetes\",\n",
    "    \"Azure Kubernetes Service\": \"Kubernetes\",\n",
    "    \"Kubernetes Engine\":        \"Kubernetes\",\n",
    "    \"AWS Lambda\":               \"Serverless Functions\",\n",
    "    \"Azure Functions\":          \"Serverless Functions\",\n",
    "    \"Cloud Run\":                \"Serverless Functions\",\n",
    "    \"Amazon S3\":                \"Object Storage\",\n",
    "    \"Azure Blob Storage\":       \"Object Storage\",\n",
    "    \"Cloud Storage\":            \"Object Storage\",\n",
    "    \"Amazon RDS\":               \"Managed Database\",\n",
    "    \"Azure SQL Database\":       \"Managed Database\",\n",
    "    \"Cloud SQL\":                \"Managed Database\",\n",
    "    \"Amazon DynamoDB\":          \"NoSQL Database\",\n",
    "    \"Amazon CloudFront\":        \"CDN\",\n",
    "    \"Azure CDN\":                \"CDN\",\n",
    "    \"AWS Data Transfer\":        \"Data Transfer\",\n",
    "    \"Azure Bandwidth\":          \"Data Transfer\",\n",
    "    \"Networking\":               \"Data Transfer\",\n",
    "    \"BigQuery\":                 \"Data Analytics\",\n",
    "}\n",
    "\n",
    "# Load labels\n",
    "with open(LABELS_PATH) as f:\n",
    "    raw_labels = json.load(f)\n",
    "\n",
    "labels = pd.DataFrame(raw_labels)\n",
    "# JSON uses 'start'/'end' keys\n",
    "labels[\"start_date\"] = pd.to_datetime(labels[\"start\"])\n",
    "labels[\"end_date\"]   = pd.to_datetime(labels[\"end\"])\n",
    "# Map raw service names -> canonical\n",
    "labels[\"service_canonical\"] = labels[\"service\"].map(SERVICE_CANONICAL).fillna(labels[\"service\"])\n",
    "\n",
    "print(f\"Loaded {len(labels)} anomaly label windows\")\n",
    "display(labels[[\"id\", \"provider\", \"service\", \"service_canonical\",\n",
    "                \"start_date\", \"end_date\", \"type\", \"description\"]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Label join ────────────────────────────────────────────────────────────────\n",
    "df[\"is_anomaly\"] = 0\n",
    "\n",
    "for _, row in labels.iterrows():\n",
    "    mask = (\n",
    "        (df[\"date\"] >= row[\"start_date\"]) &\n",
    "        (df[\"date\"] <= row[\"end_date\"]) &\n",
    "        (df[\"provider\"] == row[\"provider\"]) &\n",
    "        (df[\"service\"]  == row[\"service_canonical\"])\n",
    "    )\n",
    "    # team filter only if label specifies one\n",
    "    if pd.notna(row.get(\"team\")) and str(row.get(\"team\", \"\")).strip():\n",
    "        mask = mask & (df[\"team\"] == row[\"team\"])\n",
    "    df.loc[mask, \"is_anomaly\"] = 1\n",
    "\n",
    "# target: 1 = Normal, -1 = Anomaly  (sklearn convention)\n",
    "df[\"target\"] = df[\"is_anomaly\"].apply(lambda x: -1 if x == 1 else 1)\n",
    "\n",
    "n_anom = df[\"is_anomaly\"].sum()\n",
    "print(\"Ground-Truth Distribution:\")\n",
    "print(df[\"is_anomaly\"].value_counts())\n",
    "print(f\"\\nAnomaly rate: {100*n_anom/len(df):.2f}%  ({n_anom:,} / {len(df):,} rows)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Exploratory Data Analysis (EDA)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n",
    "\n",
    "# Cost distribution\n",
    "sns.histplot(df[\"cost_usd\"], bins=80, kde=True, ax=axes[0], color=\"steelblue\")\n",
    "axes[0].set_title(\"Distribution of Daily Costs (USD)\")\n",
    "axes[0].set_xlabel(\"Cost (USD)\")\n",
    "\n",
    "# Cost by provider, colour by anomaly\n",
    "sns.boxplot(data=df, x=\"provider\", y=\"cost_usd\", hue=\"is_anomaly\",\n",
    "            palette={0: \"#4C72B0\", 1: \"#DD8452\"}, ax=axes[1])\n",
    "axes[1].set_title(\"Cost by Provider (Normal vs Anomaly)\")\n",
    "axes[1].legend(title=\"is_anomaly\", labels=[\"Normal\", \"Anomaly\"])\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Time-series: total daily spend with anomaly highlights\n",
    "daily_total     = df.groupby(\"date\")[\"cost_usd\"].sum().reset_index()\n",
    "anomaly_dates   = df[df[\"is_anomaly\"] == 1][\"date\"].unique()\n",
    "\n",
    "plt.figure(figsize=(15, 6))\n",
    "sns.lineplot(data=daily_total, x=\"date\", y=\"cost_usd\", label=\"Total Daily Spend\", linewidth=1.2)\n",
    "for d in anomaly_dates:\n",
    "    plt.axvline(d, color=\"red\", alpha=0.08)\n",
    "plt.title(\"Total Daily Cloud Spend (red shading = ground-truth anomaly windows)\")\n",
    "plt.ylabel(\"Cost (USD)\")\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "# Spend by service\n",
    "top_services = df.groupby(\"service\")[\"cost_usd\"].sum().nlargest(10).index\n",
    "plt.figure(figsize=(12, 5))\n",
    "df_top = df[df[\"service\"].isin(top_services)]\n",
    "sns.barplot(data=df_top.groupby(\"service\", as_index=False)[\"cost_usd\"].sum(),\n",
    "            x=\"cost_usd\", y=\"service\", palette=\"Blues_r\")\n",
    "plt.title(\"Top-10 Services by Total Spend\")\n",
    "plt.xlabel(\"Total Cost (USD)\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Feature Engineering\n",
    "\n",
    "We build the same rich feature set used by `anomaly_detector.py`:\n",
    "temporal features + 7-day and 30-day rolling statistics per `(provider, service, team, environment)` group,\n",
    "plus the cost-ratio relative to the rolling mean."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.sort_values([\"provider\", \"service\", \"team\", \"environment\", \"date\"]).reset_index(drop=True)\n",
    "grp_cols = [\"provider\", \"service\", \"team\", \"environment\"]\n",
    "\n",
    "# Temporal\n",
    "df[\"dayofweek\"]  = df[\"date\"].dt.dayofweek\n",
    "df[\"dayofmonth\"] = df[\"date\"].dt.day\n",
    "df[\"month\"]      = df[\"date\"].dt.month\n",
    "df[\"weekofyear\"] = df[\"date\"].dt.isocalendar().week.astype(int)\n",
    "\n",
    "# Rolling features (shift-1 to avoid data leakage)\n",
    "g = df.groupby(grp_cols, observed=True)[\"cost_usd\"]\n",
    "\n",
    "df[\"roll7_mean\"]  = g.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).mean())\n",
    "df[\"roll7_std\"]   = g.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).std().fillna(0))\n",
    "df[\"roll30_mean\"] = g.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())\n",
    "\n",
    "df[\"cost_vs_roll7\"]  = df[\"cost_usd\"] / df[\"roll7_mean\"].replace(0, np.nan).fillna(1)\n",
    "df[\"cost_vs_roll30\"] = df[\"cost_usd\"] / df[\"roll30_mean\"].replace(0, np.nan).fillna(1)\n",
    "\n",
    "# Encode categoricals\n",
    "for col in grp_cols:\n",
    "    df[f\"{col}_code\"] = df[col].astype(\"category\").cat.codes\n",
    "\n",
    "df = df.dropna().reset_index(drop=True)\n",
    "\n",
    "FEATURES = [\n",
    "    \"cost_usd\", \"dayofweek\", \"dayofmonth\", \"month\", \"weekofyear\",\n",
    "    \"roll7_mean\", \"roll7_std\", \"roll30_mean\",\n",
    "    \"cost_vs_roll7\", \"cost_vs_roll30\",\n",
    "    \"provider_code\", \"service_code\", \"team_code\", \"environment_code\",\n",
    "]\n",
    "\n",
    "X = df[FEATURES]\n",
    "y = df[\"target\"]\n",
    "\n",
    "print(f\"Feature matrix : {X.shape}\")\n",
    "print(f\"Anomaly rows   : {(y == -1).sum():,} / {len(y):,}  ({100*(y==-1).mean():.2f}%)\")\n",
    "print(\"\\nSample features:\")\n",
    "display(X.head())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Chronological Train / Test Split & Model Training"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "split_idx = int(len(X) * 0.8)\n",
    "X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]\n",
    "y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]\n",
    "df_test = df.iloc[split_idx:].copy()\n",
    "\n",
    "# Split date ranges\n",
    "train_dates = df.iloc[:split_idx][\"date\"]\n",
    "test_dates  = df.iloc[split_idx:][\"date\"]\n",
    "print(f\"Train: {len(X_train):,} rows  |  {train_dates.min().date()} -> {train_dates.max().date()}  \"\n",
    "      f\"|  anomalies: {(y_train == -1).sum():,}\")\n",
    "print(f\"Test : {len(X_test):,} rows   |  {test_dates.min().date()}  -> {test_dates.max().date()}  \"\n",
    "      f\"|  anomalies: {(y_test == -1).sum():,}\")\n",
    "\n",
    "# Contamination = actual anomaly rate in training set\n",
    "contamination = float(np.clip((y_train == -1).sum() / len(y_train), 0.01, 0.5))\n",
    "print(f\"\\nContamination  : {contamination:.4f}\")\n",
    "\n",
    "# Build and fit pipeline\n",
    "pipeline = Pipeline([\n",
    "    (\"scaler\",  StandardScaler()),\n",
    "    (\"iforest\", IsolationForest(\n",
    "        n_estimators=200,\n",
    "        contamination=contamination,\n",
    "        random_state=42,\n",
    "        n_jobs=-1,\n",
    "    ))\n",
    "])\n",
    "\n",
    "print(\"\\nTraining Isolation Forest (n_estimators=200) ...\")\n",
    "pipeline.fit(X_train)\n",
    "\n",
    "y_pred = pipeline.predict(X_test)\n",
    "df_test[\"pred_target\"] = y_pred\n",
    "\n",
    "print(\"Done.\")\n",
    "pred_counts = pd.Series(y_pred).value_counts().rename({1: \"Normal (1)\", -1: \"Anomaly (-1)\"})\n",
    "print(\"\\nPredictions on test set:\")\n",
    "print(pred_counts)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Evaluation Metrics"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"=== Classification Report ===\")\n",
    "\n",
    "if (y_test == -1).sum() == 0:\n",
    "    print(\"No ground-truth anomalies in the test window — metrics not meaningful.\")\n",
    "else:\n",
    "    print(classification_report(\n",
    "        y_test, y_pred,\n",
    "        labels=[-1, 1],\n",
    "        target_names=[\"Anomaly (-1)\", \"Normal (1)\"],\n",
    "        zero_division=0\n",
    "    ))\n",
    "\n",
    "    # Confusion matrix\n",
    "    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])\n",
    "    plt.figure(figsize=(6, 4))\n",
    "    sns.heatmap(cm, annot=True, fmt=\"d\", cmap=\"Blues\",\n",
    "                xticklabels=[\"Anomaly\", \"Normal\"],\n",
    "                yticklabels=[\"Anomaly\", \"Normal\"])\n",
    "    plt.xlabel(\"Predicted\")\n",
    "    plt.ylabel(\"Actual\")\n",
    "    plt.title(\"Confusion Matrix (Test Set)\")\n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "\n",
    "    precision = precision_score(y_test, y_pred, pos_label=-1, zero_division=0)\n",
    "    recall    = recall_score(y_test, y_pred,    pos_label=-1, zero_division=0)\n",
    "    f1        = f1_score(y_test, y_pred,        pos_label=-1, zero_division=0)\n",
    "\n",
    "    print(f\"Precision (Anomaly): {precision:.4f}\")\n",
    "    print(f\"Recall    (Anomaly): {recall:.4f}\")\n",
    "    print(f\"F1-Score  (Anomaly): {f1:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Detected Anomalies — Sample View"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "detected = df_test[df_test[\"pred_target\"] == -1].copy()\n",
    "detected[\"true_anomaly\"] = (df_test[\"target\"] == -1).values\n",
    "\n",
    "print(f\"Total rows flagged by model : {len(detected):,}\")\n",
    "print(f\"  of which are true anomalies: {detected['true_anomaly'].sum():,}  \"\n",
    "      f\"(TP={detected['true_anomaly'].sum()}, FP={len(detected)-detected['true_anomaly'].sum()})\")\n",
    "\n",
    "display(\n",
    "    detected[[\"date\", \"provider\", \"service\", \"team\", \"environment\",\n",
    "              \"cost_usd\", \"roll7_mean\", \"true_anomaly\"]]\n",
    "    .sort_values(\"cost_usd\", ascending=False)\n",
    "    .head(20)\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Forecasting Baseline: MAE & RMSE (Normal rows only)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "normal_test = df_test[df_test[\"target\"] == 1]\n",
    "\n",
    "if len(normal_test) > 0:\n",
    "    mae  = mean_absolute_error(normal_test[\"cost_usd\"], normal_test[\"roll7_mean\"])\n",
    "    rmse = np.sqrt(mean_squared_error(normal_test[\"cost_usd\"], normal_test[\"roll7_mean\"]))\n",
    "    print(f\"Normal rows in test   : {len(normal_test):,}\")\n",
    "    print(f\"MAE  (actual vs 7-day rolling mean) : ${mae:.2f}\")\n",
    "    print(f\"RMSE (actual vs 7-day rolling mean) : ${rmse:.2f}\")\n",
    "else:\n",
    "    print(\"No normal rows in test set.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Overfitting / Underfitting — Validation Curve\n",
    "\n",
    "We vary `n_estimators` and track Train vs Test F1 to check for over/under-fitting."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if (y_test == -1).sum() == 0 or (y_train == -1).sum() == 0:\n",
    "    print(\"Skipping validation curve — one split has no anomalies.\")\n",
    "else:\n",
    "    estimators = [10, 50, 100, 200, 300, 500]\n",
    "    train_f1, test_f1 = [], []\n",
    "\n",
    "    print(\"Evaluating n_estimators...\")\n",
    "    for n in estimators:\n",
    "        pipe = Pipeline([\n",
    "            (\"scaler\",  StandardScaler()),\n",
    "            (\"iforest\", IsolationForest(\n",
    "                n_estimators=n,\n",
    "                contamination=contamination,\n",
    "                random_state=42,\n",
    "                n_jobs=-1,\n",
    "            ))\n",
    "        ])\n",
    "        pipe.fit(X_train)\n",
    "        train_f1.append(f1_score(y_train, pipe.predict(X_train), pos_label=-1, zero_division=0))\n",
    "        test_f1.append( f1_score(y_test,  pipe.predict(X_test),  pos_label=-1, zero_division=0))\n",
    "        print(f\"  n={n:4d}  train_F1={train_f1[-1]:.4f}  test_F1={test_f1[-1]:.4f}\")\n",
    "\n",
    "    plt.figure(figsize=(10, 6))\n",
    "    plt.plot(estimators, train_f1, marker=\"o\", label=\"Train F1\")\n",
    "    plt.plot(estimators, test_f1,  marker=\"o\", linestyle=\"--\", label=\"Test F1\")\n",
    "    plt.xlabel(\"n_estimators\")\n",
    "    plt.ylabel(\"F1-Score (Anomaly class)\")\n",
    "    plt.title(\"Validation Curve: Overfitting vs Underfitting\")\n",
    "    plt.legend()\n",
    "    plt.grid(True)\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "**Interpreting the curve:**\n",
    "- **Train F1 >> Test F1** → overfitting (reduce `n_estimators` or increase `contamination`).\n",
    "- **Both low** → underfitting (add more features or lower `contamination`).\n",
    "- **Converged & high** → well-generalised model."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("Isolation_Forest_Pipeline.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_dict, f, indent=1)

print("Notebook regenerated successfully!")

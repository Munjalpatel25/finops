"""
Phase 1 - Storage Layer
SQLite-based persistent store for unified billing data.
Provides fast query helpers used by Phase 2 (anomaly detection)
and Phase 3 (forecasting).
"""

import sqlite3
import sys

# Make stdout UTF-8 safe on Windows
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except AttributeError:
    pass
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from contextlib import contextmanager

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "database", "finops.db")


# ══════════════════════════════════════════════════════════════════════════════
#  Schema
# ══════════════════════════════════════════════════════════════════════════════

SCHEMA_SQL = """
-- Daily billing facts
CREATE TABLE IF NOT EXISTS daily_billing (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT    NOT NULL,
    provider    TEXT    NOT NULL,   -- aws | azure | gcp
    service     TEXT    NOT NULL,
    category    TEXT    NOT NULL,   -- compute | storage | networking | database | analytics | other
    team        TEXT    NOT NULL,
    environment TEXT    NOT NULL,
    region      TEXT    NOT NULL,
    cost_usd    REAL    NOT NULL
);

-- Anomaly labels (ground truth from generator)
CREATE TABLE IF NOT EXISTS anomaly_labels (
    id          TEXT    PRIMARY KEY,
    provider    TEXT    NOT NULL,
    type        TEXT    NOT NULL,   -- spike | gradual_drift | correlated
    service     TEXT,
    team        TEXT,
    start_date  TEXT    NOT NULL,
    end_date    TEXT    NOT NULL,
    multiplier  REAL,
    description TEXT,
    label       INTEGER NOT NULL DEFAULT 1
);

-- Detected anomalies (filled by Phase 2)
CREATE TABLE IF NOT EXISTS detected_anomalies (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at     TEXT    NOT NULL,
    date            TEXT    NOT NULL,
    provider        TEXT    NOT NULL,
    service         TEXT    NOT NULL,
    team            TEXT    NOT NULL,
    environment     TEXT    NOT NULL,
    cost_usd        REAL    NOT NULL,
    expected_cost   REAL    NOT NULL,
    deviation_pct   REAL    NOT NULL,
    severity        TEXT    NOT NULL,   -- LOW | MEDIUM | HIGH | CRITICAL
    anomaly_type    TEXT    NOT NULL,   -- spike | drift | seasonal
    detector        TEXT    NOT NULL,   -- zscore | isolation_forest | lstm
    shap_factors    TEXT,               -- JSON
    description     TEXT
);

-- Forecasts (filled by Phase 3)
CREATE TABLE IF NOT EXISTS forecasts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    horizon     INTEGER NOT NULL,   -- 7 | 30 | 90
    target_date TEXT    NOT NULL,
    provider    TEXT,
    service     TEXT,
    team        TEXT,
    environment TEXT,
    p10         REAL    NOT NULL,
    p50         REAL    NOT NULL,
    p90         REAL    NOT NULL,
    model       TEXT    NOT NULL    -- prophet | lgbm | ensemble
);

-- Budget config
CREATE TABLE IF NOT EXISTS budgets (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    team        TEXT    NOT NULL,
    provider    TEXT,
    service     TEXT,
    period      TEXT    NOT NULL,   -- monthly | quarterly
    amount_usd  REAL    NOT NULL,
    created_at  TEXT    NOT NULL
);

-- Upload history (tracks each CSV upload)
CREATE TABLE IF NOT EXISTS upload_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    filename        TEXT    NOT NULL,
    uploaded_at     TEXT    NOT NULL,
    total_rows      INTEGER NOT NULL DEFAULT 0,
    total_cost      REAL    NOT NULL DEFAULT 0,
    anomaly_count   INTEGER NOT NULL DEFAULT 0,
    savings         REAL    NOT NULL DEFAULT 0,
    detection_rate  REAL    NOT NULL DEFAULT 0,
    providers       TEXT,               -- JSON array
    severity_breakdown TEXT             -- JSON object
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_daily_date     ON daily_billing(date);
CREATE INDEX IF NOT EXISTS idx_daily_provider ON daily_billing(provider, date);
CREATE INDEX IF NOT EXISTS idx_daily_team     ON daily_billing(team, date);
CREATE INDEX IF NOT EXISTS idx_daily_service  ON daily_billing(service, date);
CREATE INDEX IF NOT EXISTS idx_anomaly_date   ON detected_anomalies(date);
CREATE INDEX IF NOT EXISTS idx_forecast_date  ON forecasts(target_date, horizon);
"""


# ══════════════════════════════════════════════════════════════════════════════
#  Connection Manager
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def get_conn(db_path: str = DB_PATH):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Initialization
# ══════════════════════════════════════════════════════════════════════════════

def init_db(db_path: str = DB_PATH):
    """Create all tables and indexes."""
    with get_conn(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
    print(f"  ✅ Database initialized: {db_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Loaders
# ══════════════════════════════════════════════════════════════════════════════

def load_billing_data(df: pd.DataFrame, db_path: str = DB_PATH):
    """
    Load normalized daily billing DataFrame into the DB.
    Clears existing data first (idempotent reload).
    """
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM daily_billing")
        df_copy = df.copy()
        df_copy["date"] = df_copy["date"].astype(str)
        df_copy.to_sql("daily_billing", conn, if_exists="append", index=False,
                       method="multi", chunksize=1000)
    print(f"  ✅ Loaded {len(df):,} billing rows into DB")


def load_anomaly_labels(labels: list, db_path: str = DB_PATH):
    """Load ground truth anomaly labels."""
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM anomaly_labels")
        for a in labels:
            conn.execute("""
                INSERT OR REPLACE INTO anomaly_labels
                    (id, provider, type, service, team, start_date, end_date, multiplier, description, label)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                a["id"], a["provider"], a["type"],
                a.get("service"), a.get("team"),
                a["start"], a["end"],
                a.get("multiplier", a.get("max_multiplier")),
                a.get("description"), a.get("label", 1)
            ))
    print(f"  ✅ Loaded {len(labels)} anomaly labels")


def insert_default_budgets(db_path: str = DB_PATH):
    """Insert sample team budgets for breach prediction."""
    from data_generator import TEAMS
    # ── Real-life differentiated monthly cloud budgets ──────────────────────
    # Total actual spend ≈ $780k/month across all teams.
    # Budgets are set intentionally varied and slightly tighter than actuals
    # so Teams show a realistic mix of OK / WARNING / BREACH states.
    #
    # ml-team    → Highest: GPU training jobs, BigQuery, LLM APIs (~$210k)
    # platform   → Wide infra ownership: K8s, networking, observability (~$185k)
    # data-eng   → Data pipelines, warehousing, S3 at scale (~$155k)
    # backend    → APIs, RDS, Lambda, caching layers (~$125k)
    # frontend   → CDN, minimal compute, static hosting (~$72k)
    budgets = [
        ("ml-team",   None, None, "monthly", 210_000),
        ("platform",  None, None, "monthly", 185_000),
        ("data-eng",  None, None, "monthly", 155_000),
        ("backend",   None, None, "monthly", 125_000),
        ("frontend",  None, None, "monthly",  72_000),
    ]
    with get_conn(db_path) as conn:
        conn.execute("DELETE FROM budgets")
        now = datetime.now().isoformat()
        for team, provider, service, period, amount in budgets:
            conn.execute("""
                INSERT INTO budgets (team, provider, service, period, amount_usd, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (team, provider, service, period, amount, now))
    print(f"  ✅ Inserted {len(budgets)} budget records")


# ══════════════════════════════════════════════════════════════════════════════
#  Query Helpers  (used by Phase 2 + Phase 3)
# ══════════════════════════════════════════════════════════════════════════════

def get_time_series(provider: str = None,
                    service: str = None,
                    team: str = None,
                    environment: str = None,
                    start_date: str = None,
                    end_date: str = None,
                    db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Fetch aggregated daily cost time series.
    All parameters are optional filters.
    """
    conditions = ["1=1"]
    params = []

    if provider:    conditions.append("provider = ?");    params.append(provider)
    if service:     conditions.append("service = ?");     params.append(service)
    if team:        conditions.append("team = ?");        params.append(team)
    if environment: conditions.append("environment = ?"); params.append(environment)
    if start_date:  conditions.append("date >= ?");       params.append(start_date)
    if end_date:    conditions.append("date <= ?");       params.append(end_date)

    sql = f"""
        SELECT date, provider, service, category, team, environment, region,
               SUM(cost_usd) AS cost_usd
        FROM daily_billing
        WHERE {' AND '.join(conditions)}
        GROUP BY date, provider, service, category, team, environment, region
        ORDER BY date
    """
    with get_conn(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)

    df["date"] = pd.to_datetime(df["date"])
    return df


def get_total_daily_spend(start_date: str = None,
                          end_date: str = None,
                          db_path: str = DB_PATH) -> pd.DataFrame:
    """Aggregated total spend per day across all providers."""
    conditions = ["1=1"]
    params = []
    if start_date: conditions.append("date >= ?"); params.append(start_date)
    if end_date:   conditions.append("date <= ?"); params.append(end_date)

    sql = f"""
        SELECT date, SUM(cost_usd) AS cost_usd
        FROM daily_billing
        WHERE {' AND '.join(conditions)}
        GROUP BY date ORDER BY date
    """
    with get_conn(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_spend_by_dimension(dimension: str = "provider",
                           db_path: str = DB_PATH) -> pd.DataFrame:
    """Get total spend grouped by any dimension: provider/service/team/category."""
    valid = {"provider", "service", "team", "category", "environment"}
    if dimension not in valid:
        raise ValueError(f"dimension must be one of {valid}")

    sql = f"""
        SELECT {dimension}, SUM(cost_usd) AS total_cost,
               MIN(date) AS first_date, MAX(date) AS last_date
        FROM daily_billing
        GROUP BY {dimension}
        ORDER BY total_cost DESC
    """
    with get_conn(db_path) as conn:
        return pd.read_sql_query(sql, conn)


def get_budgets(team: str = None, db_path: str = DB_PATH) -> pd.DataFrame:
    sql = "SELECT * FROM budgets"
    params = []
    if team:
        sql += " WHERE team = ?"
        params.append(team)
    with get_conn(db_path) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def save_detected_anomaly(anomaly: dict, db_path: str = DB_PATH):
    with get_conn(db_path) as conn:
        conn.execute("""
            INSERT INTO detected_anomalies
                (detected_at, date, provider, service, team, environment,
                 cost_usd, expected_cost, deviation_pct, severity,
                 anomaly_type, detector, shap_factors, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            anomaly["date"], anomaly["provider"], anomaly["service"],
            anomaly["team"], anomaly["environment"],
            anomaly["cost_usd"], anomaly["expected_cost"],
            anomaly["deviation_pct"], anomaly["severity"],
            anomaly["anomaly_type"], anomaly["detector"],
            json.dumps(anomaly.get("shap_factors", {})),
            anomaly.get("description", "")
        ))


def save_detected_anomalies_batch(records: list[dict],
                                   db_path: str = DB_PATH) -> list[int]:
    """
    Bulk-insert anomaly records in a SINGLE transaction.
    Much faster than calling save_detected_anomaly() N times.
    Returns list of inserted row IDs (same order as *records*).
    """
    ids: list[int] = []
    now = datetime.now().isoformat()
    with get_conn(db_path) as conn:
        for a in records:
            conn.execute("""
                INSERT INTO detected_anomalies
                    (detected_at, date, provider, service, team, environment,
                     cost_usd, expected_cost, deviation_pct, severity,
                     anomaly_type, detector, shap_factors, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now,
                a["date"], a["provider"], a["service"],
                a["team"], a["environment"],
                a["cost_usd"], a["expected_cost"],
                a["deviation_pct"], a["severity"],
                a["anomaly_type"], a["detector"],
                json.dumps(a.get("shap_factors", {})),
                a.get("description", ""),
            ))
            ids.append(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
    return ids


def update_anomaly_shap(row_id: int, shap_factors: dict,
                         db_path: str = DB_PATH) -> None:
    """Patch the shap_factors JSON for an already-inserted anomaly record."""
    with get_conn(db_path) as conn:
        conn.execute(
            "UPDATE detected_anomalies SET shap_factors = ? WHERE id = ?",
            (json.dumps(shap_factors), row_id),
        )



def save_forecast(forecast_rows: list, db_path: str = DB_PATH):
    with get_conn(db_path) as conn:
        now = datetime.now().isoformat()
        for row in forecast_rows:
            conn.execute("""
                INSERT INTO forecasts
                    (created_at, horizon, target_date, provider, service,
                     team, environment, p10, p50, p90, model)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now, row["horizon"], row["target_date"],
                row.get("provider"), row.get("service"),
                row.get("team"), row.get("environment"),
                row["p10"], row["p50"], row["p90"], row["model"]
            ))


# ══════════════════════════════════════════════════════════════════════════════
#  Main — run full Phase 1 pipeline
# ══════════════════════════════════════════════════════════════════════════════

def run_phase1_pipeline():
    """
    End-to-end Phase 1:
    1. Generate synthetic data
    2. Normalize to unified schema
    3. Load into SQLite
    """
    from data_generator import main as generate
    from normalizer import run_normalization_pipeline

    print("\n" + "=" * 52)
    print("  PHASE 1 - DATA LAYER SETUP")
    print("=" * 52)

    # Step 1: Generate raw data
    print("\n[1/3] Generating raw billing data...")
    aws_df, azure_df, gcp_df, anomalies = generate()

    # Step 2: Normalize
    print("\n[2/3] Normalizing to unified schema...")
    daily_df = run_normalization_pipeline()

    # Step 3: Store
    print("\n[3/3] Loading into SQLite database...")
    init_db()
    load_billing_data(daily_df)
    load_anomaly_labels(anomalies)
    insert_default_budgets()

    # Quick sanity check
    print("\n[OK] Sanity check queries:")
    total = get_total_daily_spend()
    print(f"    Total days in DB : {len(total)}")
    print(f"    Total spend      : ${total['cost_usd'].sum():,.2f}")

    by_provider = get_spend_by_dimension("provider")
    print("\n    Spend by provider:")
    for _, row in by_provider.iterrows():
        print(f"      {row['provider'].upper():<8} ${row['total_cost']:>12,.2f}")

    print("\n[OK] Phase 1 Complete! Database ready at finops.db")
    print("   -> Next: Phase 2 - Anomaly Detection Engine\n")


if __name__ == "__main__":
    run_phase1_pipeline()

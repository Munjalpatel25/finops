"""
Phase 4 + 5 - FastAPI Backend
==============================
Serves live FinOps data from finops.db to the React frontend at
localhost:5173.  All endpoints return JSON shaped to match the exact
data contracts consumed by each frontend page.

Endpoints
---------
GET  /api/summary          -> KPI cards + provider breakdown + budget bars
GET  /api/anomalies        -> paginated detected anomalies with SHAP factors
GET  /api/forecast         -> 7/30/90-day p10/p50/p90 forecast bands
GET  /api/timeseries       -> daily spend (filterable by provider/team/service)
GET  /api/budgets          -> team budget vs actual spend
GET  /api/health           -> liveness probe
POST /api/run-detection    -> trigger Phase-2 anomaly detection (background)
POST /api/run-forecast     -> trigger Phase-3 forecasting (background)
POST /api/chat             -> Phase-5 AI chatbot (Gemini-powered FinOps Q&A)

Run:   python api.py
       python api.py --port 8000 --reload
"""

from __future__ import annotations

import sys
import logging
import json
from contextlib import asynccontextmanager
from datetime import datetime, date, timedelta
from typing import Optional

# ── stdout UTF-8 on Windows ───────────────────────────────────────────────────
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
except AttributeError:
    pass

# ── FastAPI / Pydantic ────────────────────────────────────────────────────────
from fastapi import FastAPI, Query, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np

# ── Storage helpers (Phase 1) ──────────────────────────────────────────────────
from storage import (
    get_time_series,
    get_total_daily_spend,
    get_spend_by_dimension,
    get_budgets,
    get_conn,
    DB_PATH,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api")


# ══════════════════════════════════════════════════════════════════════════════
#  App lifecycle
# ══════════════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    from storage import init_db
    init_db()
    log.info("FinOps API starting — connected to %s", DB_PATH)
    yield
    log.info("FinOps API shutting down")
    
app = FastAPI(
    title="FinOps Intelligence API",
    description="Cloud FinOps Intelligence Platform — Phase 4 REST API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS — allow the Vite dev server ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _safe_float(v) -> float:
    """Return 0.0 for NaN/None, otherwise plain float."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 0.0
    return float(v)


def _fmt_usd(v: float) -> str:
    """Format a dollar value like $1,248,302 or $24.5k."""
    if v >= 1_000_000:
        return f"${v/1_000_000:.2f}M"
    if v >= 1_000:
        return f"${v:,.0f}"
    return f"${v:.2f}"


def _month_start() -> str:
    today = date.today()
    return today.replace(day=1).isoformat()


def _prev_month_start() -> str:
    today = date.today()
    first = today.replace(day=1)
    prev_last = first - timedelta(days=1)
    return prev_last.replace(day=1).isoformat()


def _prev_month_end() -> str:
    today = date.today()
    first = today.replace(day=1)
    prev_last = first - timedelta(days=1)
    return prev_last.isoformat()


# ══════════════════════════════════════════════════════════════════════════════
#  GET /api/health
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health", tags=["meta"])
def health():
    """Liveness probe."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ══════════════════════════════════════════════════════════════════════════════
#  GET /api/summary
#  Feeds: CommandCenter top KPI cards, provider pie chart, budget bars,
#         spend-forecast bar chart (last 7 days actual + 7 days predicted)
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/summary", tags=["dashboard"])
def get_summary():
    """
    Returns:
    {
      totalSpend:  { value, trend, raw }
      savings:     { value, active, raw }
      anomalies:   { count, severity }
      providerBreakdown: [ { name, value, fill, total } ]
      departmentBudget:  [ { name, budget, actual } ]
      spendForecast:     [ { name, actual, predicted } ]   // last 7 + next 7 days
    }
    """
    today_str     = date.today().isoformat()
    mtd_start     = _month_start()
    prev_start    = _prev_month_start()
    prev_end      = _prev_month_end()

    # ── MTD spend ─────────────────────────────────────────────────────────────
    mtd_df   = get_total_daily_spend(start_date=mtd_start, end_date=today_str)
    mtd_total = _safe_float(mtd_df["cost_usd"].sum()) if not mtd_df.empty else 0.0

    # ── Fallback: if current month has no data, use the latest month in DB ────
    if mtd_total == 0:
        with get_conn(DB_PATH) as conn:
            latest_row = conn.execute(
                "SELECT MAX(date) as max_date FROM daily_billing"
            ).fetchone()
        if latest_row and latest_row["max_date"]:
            latest_date = datetime.strptime(latest_row["max_date"], "%Y-%m-%d").date()
            mtd_start = latest_date.replace(day=1).isoformat()
            today_str = latest_date.isoformat()
            # Recalculate prev month relative to latest data month
            prev_last  = latest_date.replace(day=1) - timedelta(days=1)
            prev_start = prev_last.replace(day=1).isoformat()
            prev_end   = prev_last.isoformat()
            # Re-query
            mtd_df    = get_total_daily_spend(start_date=mtd_start, end_date=today_str)
            mtd_total = _safe_float(mtd_df["cost_usd"].sum()) if not mtd_df.empty else 0.0

    # ── Previous month spend ──────────────────────────────────────────────────
    prev_df   = get_total_daily_spend(start_date=prev_start, end_date=prev_end)
    prev_total = _safe_float(prev_df["cost_usd"].sum()) if not prev_df.empty else 0.0

    trend_pct = ((mtd_total - prev_total) / prev_total * 100) if prev_total else 0.0
    trend_str = f"{trend_pct:+.1f}% vs prev. month"

    # ── Active anomalies ──────────────────────────────────────────────────────
    with get_conn(DB_PATH) as conn:
        anom_rows = conn.execute(
            "SELECT COUNT(*) as n, severity FROM detected_anomalies "
            "GROUP BY severity ORDER BY CASE severity "
            "WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 "
            "WHEN 'MEDIUM' THEN 3 ELSE 4 END LIMIT 1"
        ).fetchone()
        total_anoms = conn.execute(
            "SELECT COUNT(*) as n FROM detected_anomalies"
        ).fetchone()["n"]

    top_severity = anom_rows["severity"] if anom_rows else "NONE"
    top_sev_label = {
        "CRITICAL": "Critical severity anomalies detected",
        "HIGH":     "High severity detected",
        "MEDIUM":   "Medium severity detected",
        "LOW":      "Low severity anomalies",
        "NONE":     "No anomalies detected",
    }.get(top_severity, "Anomalies detected")

    # ── Provider breakdown ────────────────────────────────────────────────────
    PROVIDER_COLORS = {"aws": "#06B6D4", "azure": "#3B82F6", "gcp": "#8B5CF6"}
    prov_df = get_spend_by_dimension("provider")
    grand_total = _safe_float(prov_df["total_cost"].sum()) if not prov_df.empty else 1.0

    provider_breakdown = []
    for _, r in prov_df.iterrows():
        pct = round(_safe_float(r["total_cost"]) / grand_total * 100, 1) if grand_total else 0
        provider_breakdown.append({
            "name":  r["provider"].upper(),
            "value": pct,
            "fill":  PROVIDER_COLORS.get(r["provider"].lower(), "#64748b"),
            "total": _safe_float(r["total_cost"]),
        })

    # ── Department budget vs actual ───────────────────────────────────────────
    budgets_df = get_budgets()
    dept_budget = []
    if not budgets_df.empty:
        for _, b in budgets_df.iterrows():
            team_df = get_time_series(team=b["team"], start_date=mtd_start)
            actual = _safe_float(team_df["cost_usd"].sum()) if not team_df.empty else 0.0
            dept_budget.append({
                "name":   b["team"],
                "budget": _safe_float(b["amount_usd"]),
                "actual": round(actual, 2),
            })

    # ── Spend forecast: pair last 7 actual days with next 7 predicted ────────
    # Use the (possibly fallback) today_str so historical CSVs still populate the chart
    _ref_date = datetime.strptime(today_str, "%Y-%m-%d").date()
    yesterday_str = (_ref_date - timedelta(days=1)).isoformat()
    week_ago_str  = (_ref_date - timedelta(days=7)).isoformat()
    hist_df = get_total_daily_spend(start_date=week_ago_str, end_date=today_str)

    DAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    with get_conn(DB_PATH) as conn:
        fcast_rows = conn.execute(
            """SELECT target_date, p50 FROM forecasts
               WHERE horizon=7 AND model='ensemble'
                 AND provider IS NULL AND team IS NULL AND service IS NULL
               ORDER BY target_date LIMIT 7"""
        ).fetchall()

    # Build indexed lists so we can zip them into side-by-side pairs
    hist_list = []
    if not hist_df.empty:
        for _, row in hist_df.iterrows():
            d_obj = pd.to_datetime(row["date"])
            hist_list.append((DAY_ABBR[d_obj.dayofweek], round(_safe_float(row["cost_usd"]), 2)))

    pred_list = []
    for fr in fcast_rows:
        d_obj = datetime.strptime(fr["target_date"], "%Y-%m-%d")
        pred_list.append((DAY_ABBR[d_obj.weekday()], round(fr["p50"], 2)))

    # Merge pairwise by position — same index → same bar group (side by side)
    n_entries = max(len(hist_list), len(pred_list))
    spend_forecast_chart = []
    for i in range(n_entries):
        # Use actual day name where available, else fall back to predicted day name
        name    = hist_list[i][0]    if i < len(hist_list) else pred_list[i][0]
        actual  = hist_list[i][1]    if i < len(hist_list) else None
        predicted = pred_list[i][1]  if i < len(pred_list) else None
        spend_forecast_chart.append({"name": name, "actual": actual, "predicted": predicted})

    # ── Savings opportunities (HIGH+CRITICAL anomalies cost delta) ────────────
    with get_conn(DB_PATH) as conn:
        savings_row = conn.execute(
            """SELECT SUM(cost_usd - expected_cost) AS delta
               FROM detected_anomalies
               WHERE severity IN ('HIGH','CRITICAL')
                 AND deviation_pct > 0"""
        ).fetchone()
    savings_raw = _safe_float(savings_row["delta"]) if savings_row["delta"] else 0.0

    return {
        "totalSpend": {
            "value": _fmt_usd(mtd_total),
            "trend": trend_str,
            "raw":   round(mtd_total, 2),
        },
        "savings": {
            "value":  _fmt_usd(savings_raw),
            "active": min(total_anoms, 99),
            "raw":    round(savings_raw, 2),
        },
        "anomalies": {
            "count":    total_anoms,
            "severity": top_sev_label,
        },
        "providerBreakdown":  provider_breakdown,
        "departmentBudget":   dept_budget,
        "spendForecast":      spend_forecast_chart,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GET /api/anomalies
#  Feeds: AnomalyWatch page + command center anomaly count
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/anomalies", tags=["anomalies"])
def get_anomalies(
    severity:    Optional[str] = Query(None, description="CRITICAL|HIGH|MEDIUM|LOW"),
    provider:    Optional[str] = Query(None),
    team:        Optional[str] = Query(None),
    service:     Optional[str] = Query(None),
    anomaly_type: Optional[str] = Query(None, alias="type"),
    limit:       int           = Query(50,  ge=1, le=500),
    offset:      int           = Query(0,   ge=0),
):
    """
    Returns:
    {
      total: int,
      items: [
        {
          id, detected_at, date, provider, service, team, environment,
          cost_usd, expected_cost, deviation_pct, severity,
          anomaly_type, detector, shap_factors, description,
          projected_monthly_drift
        }
      ],
      top: { severity, date, service, team, deviation_pct }  // worst single anomaly
    }
    """
    conditions = ["1=1"]
    params: list = []

    if severity:
        conditions.append("severity = ?"); params.append(severity.upper())
    if provider:
        conditions.append("provider = ?"); params.append(provider.lower())
    if team:
        conditions.append("team = ?"); params.append(team)
    if service:
        conditions.append("service = ?"); params.append(service)
    if anomaly_type:
        conditions.append("anomaly_type = ?"); params.append(anomaly_type)

    where = " AND ".join(conditions)

    with get_conn(DB_PATH) as conn:
        count_row = conn.execute(
            f"SELECT COUNT(*) as n FROM detected_anomalies WHERE {where}", params
        ).fetchone()
        total = count_row["n"]

        rows = conn.execute(
            f"""SELECT * FROM detected_anomalies
                WHERE {where}
                ORDER BY
                  CASE severity
                    WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2
                    WHEN 'MEDIUM'   THEN 3 ELSE 4
                  END,
                  deviation_pct DESC
                LIMIT ? OFFSET ?""",
            params + [limit, offset],
        ).fetchall()

        worst = conn.execute(
            """SELECT severity, date, service, team, deviation_pct
               FROM detected_anomalies
               ORDER BY deviation_pct DESC LIMIT 1"""
        ).fetchone()

    items = []
    for r in rows:
        shap = {}
        try:
            shap = json.loads(r["shap_factors"]) if r["shap_factors"] else {}
        except Exception:
            pass

        monthly_drift = round(_safe_float(r["cost_usd"] - r["expected_cost"]) * 30, 2)

        items.append({
            "id":             r["id"],
            "detected_at":    r["detected_at"],
            "date":           r["date"],
            "provider":       r["provider"],
            "service":        r["service"],
            "team":           r["team"],
            "environment":    r["environment"],
            "cost_usd":       round(_safe_float(r["cost_usd"]), 2),
            "expected_cost":  round(_safe_float(r["expected_cost"]), 2),
            "deviation_pct":  round(_safe_float(r["deviation_pct"]), 2),
            "severity":       r["severity"],
            "anomaly_type":   r["anomaly_type"],
            "detector":       r["detector"],
            "shap_factors":   shap,
            "description":    r["description"] or "",
            "projected_monthly_drift": monthly_drift,
        })

    top = None
    if worst:
        top = {
            "severity":     worst["severity"],
            "date":         worst["date"],
            "service":      worst["service"],
            "team":         worst["team"],
            "deviation_pct": round(_safe_float(worst["deviation_pct"]), 1),
        }

    return {"total": total, "items": items, "top": top}


# ══════════════════════════════════════════════════════════════════════════════
#  GET /api/forecast
#  Feeds: SpendForecasting page — area chart + projected spend card
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/forecast", tags=["forecast"])
def get_forecast(
    horizon:  int           = Query(30,   description="7 | 30 | 90"),
    model:    str           = Query("ensemble"),
    provider: Optional[str] = Query(None),
    team:     Optional[str] = Query(None),
    service:  Optional[str] = Query(None),
):
    """
    Returns forecast rows shaped for Recharts AreaChart:
    {
      horizon: int,
      model: str,
      projected_end_of_period: float,   // p50 of last point
      series: [
        { name, target_date, p10, p50, p90 }
      ],
      // Also returns last 30d actual for overlay
      historical: [
        { name, actual }
      ]
    }
    """
    conditions = ["horizon = ?", "model = ?"]
    params: list = [horizon, model]

    if provider:
        conditions.append("provider = ?");     params.append(provider.lower())
    else:
        conditions.append("provider IS NULL")

    if team:
        conditions.append("team = ?");         params.append(team)
    else:
        conditions.append("team IS NULL")

    if service:
        conditions.append("service = ?");      params.append(service)
    else:
        conditions.append("service IS NULL")

    where = " AND ".join(conditions)

    with get_conn(DB_PATH) as conn:
        rows = conn.execute(
            f"SELECT target_date, p10, p50, p90 FROM forecasts "
            f"WHERE {where} ORDER BY target_date",
            params,
        ).fetchall()

    if not rows:
        # Forecasts haven't been generated yet — return empty payload with hint
        hist_start = (date.today() - timedelta(days=29)).isoformat()
        hist_agg = get_total_daily_spend(start_date=hist_start)
        historical = []
        if not hist_agg.empty:
            for _, row in hist_agg.iterrows():
                dt = pd.to_datetime(row["date"])
                historical.append({"name": dt.strftime("%b %d"), "actual": round(_safe_float(row["cost_usd"]), 2)})
        return {
            "horizon":                 horizon,
            "model":                   model,
            "projected_end_of_period": 0.0,
            "series":                  [],
            "historical":              historical,
            "needs_run":               True,
        }

    series = []
    for i, r in enumerate(rows):
        dt = datetime.strptime(r["target_date"], "%Y-%m-%d")
        series.append({
            "name":        dt.strftime("%b %d"),
            "target_date": r["target_date"],
            "p10":         round(_safe_float(r["p10"]), 2),
            "p50":         round(_safe_float(r["p50"]), 2),
            "p90":         round(_safe_float(r["p90"]), 2),
        })

    projected_eop = series[-1]["p50"] if series else 0.0

    # Historical: last 30 actual daily totals for chart overlay
    hist_start = (date.today() - timedelta(days=29)).isoformat()
    hist_kwargs: dict = {}
    if provider: hist_kwargs["provider"] = provider
    if team:     hist_kwargs["team"]     = team
    if service:  hist_kwargs["service"]  = service

    if hist_kwargs:
        hist_raw = get_time_series(start_date=hist_start, **hist_kwargs)
        hist_agg = hist_raw.groupby("date")["cost_usd"].sum().reset_index() if not hist_raw.empty else pd.DataFrame()
    else:
        hist_agg = get_total_daily_spend(start_date=hist_start)

    historical = []
    if not hist_agg.empty:
        for _, row in hist_agg.iterrows():
            dt = pd.to_datetime(row["date"])
            historical.append({
                "name":   dt.strftime("%b %d"),
                "actual": round(_safe_float(row["cost_usd"]), 2),
            })

    return {
        "horizon":                horizon,
        "model":                  model,
        "projected_end_of_period": round(projected_eop, 2),
        "series":                 series,
        "historical":             historical,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GET /api/timeseries
#  Feeds: time-series charts on CommandCenter + any filtered drill-down
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/timeseries", tags=["billing"])
def get_timeseries(
    provider:    Optional[str] = Query(None),
    team:        Optional[str] = Query(None),
    service:     Optional[str] = Query(None),
    environment: Optional[str] = Query(None),
    start_date:  Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date:    Optional[str] = Query(None, description="YYYY-MM-DD"),
    group_by:    str           = Query("date", description="date | provider | service | team"),
):
    """
    Daily cost time-series, optionally filtered and grouped.

    Returns:
    {
      filters: { provider, team, service, ... },
      data: [ { date, cost_usd, ...group_cols } ]
    }
    """
    df = get_time_series(
        provider=provider,
        team=team,
        service=service,
        environment=environment,
        start_date=start_date,
        end_date=end_date,
    )

    if df.empty:
        return {"filters": {}, "data": []}

    valid_groups = {"date", "provider", "service", "team", "environment"}
    if group_by not in valid_groups:
        group_by = "date"

    group_cols = ["date", group_by] if group_by != "date" else ["date"]
    agg = df.groupby(group_cols, as_index=False)["cost_usd"].sum()
    agg["date"] = agg["date"].dt.strftime("%Y-%m-%d")
    agg["cost_usd"] = agg["cost_usd"].round(2)

    return {
        "filters": {
            "provider": provider, "team": team,
            "service": service, "environment": environment,
            "start_date": start_date, "end_date": end_date,
        },
        "data": agg.to_dict(orient="records"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GET /api/budgets
#  Feeds: CommandCenter department budget bars, SpendForecasting breach risk
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/budgets", tags=["billing"])
def get_budgets_endpoint(team: Optional[str] = Query(None)):
    """
    Returns:
    {
      items: [
        {
          team, period, budget, actual_mtd,
          utilization_pct, status,   // OK | WARNING | BREACH
          projected_eom              // p50 30-day forecast
        }
      ]
    }
    """
    budgets_df = get_budgets(team=team)
    if budgets_df.empty:
        return {"items": []}

    mtd_start  = _month_start()
    today_str  = date.today().isoformat()

    items = []
    for _, b in budgets_df.iterrows():
        # Actual MTD spend for this team
        team_ts = get_time_series(team=b["team"], start_date=mtd_start, end_date=today_str)
        actual  = _safe_float(team_ts["cost_usd"].sum()) if not team_ts.empty else 0.0

        budget  = _safe_float(b["amount_usd"])
        util    = round(actual / budget * 100, 1) if budget else 0.0

        # Status
        if util >= 100:
            status = "BREACH"
        elif util >= 80:
            status = "WARNING"
        else:
            status = "OK"

        # 30-day forecast p50
        with get_conn(DB_PATH) as conn:
            fr = conn.execute(
                """SELECT p50 FROM forecasts
                   WHERE team=? AND horizon=30 AND model='ensemble'
                     AND provider IS NULL AND service IS NULL
                   ORDER BY target_date DESC LIMIT 1""",
                [b["team"]],
            ).fetchone()
        projected_eom = round(fr["p50"], 2) if fr else None

        items.append({
            "team":            b["team"],
            "period":          b["period"],
            "budget":          round(budget, 2),
            "actual_mtd":      round(actual, 2),
            "utilization_pct": util,
            "status":          status,
            "projected_eom":   projected_eom,
        })

    return {"items": items}


# ══════════════════════════════════════════════════════════════════════════════
#  POST /api/run-detection
#  Triggers Phase-2 anomaly detection in the background
# ══════════════════════════════════════════════════════════════════════════════

# Shared state -- read by GET /api/detection-status
_detection_state: dict = {
    "running":      False,
    "step":         "idle",
    "step_num":     0,
    "total_steps":  5,
    "last_run_at":  None,
    "last_count":   0,
    "live_count":   0,    # updated every 2s from DB during a run
    "error":        None,
}


def _run_detection_task():
    import threading
    global _detection_state
    _detection_state.update(running=True, error=None, step_num=0, live_count=0)

    # Live anomaly counter: polls DB every 2s so the frontend sees growing count
    _stop_counter = threading.Event()
    def _poll_db_count():
        from storage import get_conn, DB_PATH
        while not _stop_counter.wait(timeout=2):
            try:
                with get_conn(DB_PATH) as conn:
                    c = conn.execute(
                        "SELECT COUNT(*) FROM detected_anomalies"
                    ).fetchone()[0]
                _detection_state["live_count"] = c
            except Exception:
                pass
    counter_t = threading.Thread(target=_poll_db_count, daemon=True)
    counter_t.start()

    def _step(n: int, label: str):
        _detection_state["step"]     = label
        _detection_state["step_num"] = n
        log.info("[detection] step %d/%d  %s", n, _detection_state["total_steps"], label)

    try:
        _step(1, "Loading billing data from database...")
        from anomaly_detector import run_anomaly_detection

        _step(2, "Running Z-Score + STL detector...")
        _step(3, "Training Isolation Forest (100 trees)...")

        records = run_anomaly_detection(clear_previous=True)

        _step(4, f"Computing SHAP for top 50 anomalies...")
        _step(5, f"Saved {len(records)} anomalies to database")
        _detection_state.update(
            last_count=len(records),
            live_count=len(records),
            last_run_at=datetime.utcnow().isoformat() + "Z",
        )
        log.info("[detection] complete -- %d anomalies saved", len(records))

    except Exception as exc:
        log.error("[detection] failed: %s", exc, exc_info=True)
        _detection_state["error"] = str(exc)
        _detection_state["step"]  = f"Error: {exc}"
    finally:
        _stop_counter.set()
        _detection_state["running"] = False


@app.get("/api/detection-status", tags=["actions"])
def get_detection_status():
    """
    Poll this endpoint after triggering detection to get real-time progress.
    Returns:
    {
      running: bool,
      step: str,       // human-readable current phase
      step_num: int,
      total_steps: int,
      last_run_at: str | null,
      last_count: int,
      error: str | null
    }
    """
    return _detection_state


@app.post("/api/run-detection", tags=["actions"])
def trigger_detection(background_tasks: BackgroundTasks):
    """
    Kick off Phase-2 anomaly detection in the background.
    Returns immediately — poll GET /api/detection-status for live progress.
    """
    if _detection_state["running"]:
        return JSONResponse(
            status_code=202,
            content={"status": "already_running",
                     "message": "Detection already in progress.",
                     "step": _detection_state["step"]},
        )
    background_tasks.add_task(_run_detection_task)
    return {
        "status":  "started",
        "message": "Detection started. Poll /api/detection-status for progress.",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  POST /api/run-forecast
#  Triggers Phase-3 forecasting in the background
# ══════════════════════════════════════════════════════════════════════════════

_forecast_running = False


def _run_forecast_task(horizons: list[int]):
    global _forecast_running
    try:
        log.info("[bg] Starting forecasting (horizons=%s) ...", horizons)
        from forecasting_engine import run_forecasting
        n = run_forecasting(clear_previous=True, horizons=horizons)
        log.info("[bg] Forecasting complete — %d rows saved", n)
    except Exception as exc:
        log.error("[bg] Forecasting failed: %s", exc)
    finally:
        _forecast_running = False


@app.post("/api/run-forecast", tags=["actions"])
def trigger_forecast(
    background_tasks: BackgroundTasks,
    horizons: list[int] = Query(default=[7, 30, 90]),
):
    """
    Kick off Phase-3 forecasting in the background.
    Returns immediately.
    """
    global _forecast_running
    if _forecast_running:
        return JSONResponse(
            status_code=202,
            content={"status": "already_running", "message": "Forecasting already in progress."},
        )
    _forecast_running = True
    background_tasks.add_task(_run_forecast_task, horizons)
    return {
        "status":  "started",
        "message": f"Forecasting started for horizons={horizons}. Poll /api/forecast for results.",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  POST /api/chat
#  Phase 5 — AI Chatbot powered by Google Gemini
# ══════════════════════════════════════════════════════════════════════════════

from chatbot import chat as _chatbot_chat, ChatRequest, ChatResponse as _ChatResponse


@app.post("/api/chat", tags=["ai"], response_model=_ChatResponse)
async def chat_endpoint(req: ChatRequest):
    """
    FinOps AI Chatbot endpoint.

    Request body:
    {
      "message": "why did costs spike last week?",
      "history": [
        { "role": "user",      "content": "previous question" },
        { "role": "assistant", "content": "previous answer"   }
      ]
    }

    Returns:
    {
      "reply": "<AI response>",
      "context_used": { ... }   // live DB snapshot used to answer
    }
    """
    try:
        result = _chatbot_chat(message=req.message, history=req.history)
        return result
    except Exception as exc:
        log.error("Chat endpoint error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ══════════════════════════════════════════════════════════════════════════════
#  POST /api/upload-csv   — CSV upload + streaming anomaly detection
#  GET  /api/upload-status — Live upload progress
#  POST /api/upload-reset  — Reset upload state
# ══════════════════════════════════════════════════════════════════════════════

from fastapi import UploadFile, File
from stream_detector import (
    validate_csv, run_streaming_detection,
    get_upload_state, reset_upload_state,
)


@app.post("/api/upload-csv", tags=["upload"])
async def upload_csv(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a cloud billing CSV for real-time streaming anomaly detection.

    Required columns: date, provider, service, category, team, environment,
                      region, cost_usd

    Returns immediately — poll GET /api/upload-status for live progress.
    """
    state = get_upload_state()
    if state["status"] not in ("idle", "complete", "error"):
        return JSONResponse(
            status_code=202,
            content={
                "status": "already_running",
                "message": "An upload is already in progress. Reset first.",
            },
        )

    # Read file bytes
    file_bytes = await file.read()
    filename = file.filename or "upload.csv"

    # Validate CSV structure
    ok, err_msg, row_count = validate_csv(file_bytes, filename)
    if not ok:
        return JSONResponse(
            status_code=400,
            content={"status": "validation_error", "message": err_msg},
        )

    log.info("CSV upload accepted: %s (%d rows)", filename, row_count)

    # Kick off background streaming detection
    background_tasks.add_task(run_streaming_detection, file_bytes, filename)

    return {
        "status":     "started",
        "message":    f"Upload accepted. Processing {row_count:,} rows...",
        "filename":   filename,
        "total_rows": row_count,
    }


@app.get("/api/upload-status", tags=["upload"])
def upload_status():
    """
    Poll this endpoint for live upload progress.

    Returns:
    {
      status:           "idle" | "validating" | "loading" | "detecting" | "streaming" | "complete" | "error",
      total_rows:       int,
      processed_rows:   int,
      anomaly_count:    int,
      current_row:      { date, provider, service, team, cost_usd } | null,
      recent_anomalies: [ { id, date, provider, service, team, severity, ... } ],
      error:            str | null,
      filename:         str | null,
      started_at:       str | null,
      completed_at:     str | null
    }
    """
    return get_upload_state()


@app.post("/api/upload-reset", tags=["upload"])
def upload_reset():
    """Reset upload state and cancel any running upload."""
    reset_upload_state()
    return {"status": "reset", "message": "Upload state cleared."}


@app.post("/api/clear-data", tags=["upload"])
def clear_data():
    """
    Clear ALL billing data, detected anomalies, and upload history from the database.
    Resets the dashboard to a blank state.
    """
    with get_conn(DB_PATH) as conn:
        conn.execute("DELETE FROM detected_anomalies")
        conn.execute("DELETE FROM daily_billing")
        conn.execute("DELETE FROM upload_history")
    reset_upload_state()
    log.info("All billing data, anomalies, and upload history cleared from database")
    return {"status": "cleared", "message": "All data cleared from dashboard."}


@app.post("/api/save-upload-history", tags=["upload"])
def save_upload_history():
    """
    Snapshot current upload stats into upload_history table.
    Called by frontend after upload+detection completes.
    """
    import json as _json
    state = get_upload_state()
    fname = state.get("filename", "unknown.csv")

    with get_conn(DB_PATH) as conn:
        total_rows = conn.execute("SELECT COUNT(*) AS n FROM daily_billing").fetchone()["n"]
        total_cost = conn.execute(
            "SELECT COALESCE(ROUND(SUM(cost_usd),2),0) AS s FROM daily_billing"
        ).fetchone()["s"]
        anomaly_count = conn.execute("SELECT COUNT(*) AS n FROM detected_anomalies").fetchone()["n"]
        savings = conn.execute(
            "SELECT COALESCE(ROUND(SUM(cost_usd - expected_cost),2),0) AS s FROM detected_anomalies "
            "WHERE deviation_pct > 0 AND severity IN ('HIGH','CRITICAL')"
        ).fetchone()["s"]
        detection_rate = round(anomaly_count / total_rows * 100, 1) if total_rows else 0

        providers = [r["provider"].upper() for r in conn.execute(
            "SELECT DISTINCT provider FROM daily_billing"
        ).fetchall()]

        severity_rows = conn.execute(
            "SELECT severity, COUNT(*) AS n FROM detected_anomalies GROUP BY severity"
        ).fetchall()
        severity_breakdown = {r["severity"]: r["n"] for r in severity_rows}

        conn.execute(
            "INSERT INTO upload_history "
            "(filename, uploaded_at, total_rows, total_cost, anomaly_count, savings, detection_rate, providers, severity_breakdown) "
            "VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?)",
            (fname, total_rows, total_cost, anomaly_count, savings, detection_rate,
             _json.dumps(providers), _json.dumps(severity_breakdown)),
        )

    log.info(f"Saved upload history for {fname}")
    return {"status": "saved", "filename": fname}


@app.get("/api/upload-history", tags=["upload"])
def get_upload_history():
    """Return all past upload records, newest first."""
    import json as _json
    with get_conn(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT * FROM upload_history ORDER BY uploaded_at DESC"
        ).fetchall()

    items = []
    for r in rows:
        item = dict(r)
        try:
            item["providers"] = _json.loads(item.get("providers") or "[]")
        except Exception:
            item["providers"] = []
        try:
            item["severity_breakdown"] = _json.loads(item.get("severity_breakdown") or "{}")
        except Exception:
            item["severity_breakdown"] = {}
        items.append(item)

    # Aggregate totals
    agg = {
        "total_files": len(items),
        "total_rows": sum(i["total_rows"] for i in items),
        "total_cost": round(sum(i["total_cost"] for i in items), 2),
        "total_anomalies": sum(i["anomaly_count"] for i in items),
        "total_savings": round(sum(i["savings"] for i in items), 2),
    }

    return {"items": items, "aggregate": agg}


@app.get("/api/upload-analytics", tags=["upload"])
def upload_analytics():
    """
    Return rich analytics data from the uploaded dataset for visualizations.
    Cost breakdowns, anomaly severity, model stats, and trends.
    """
    state = get_upload_state()

    with get_conn(DB_PATH) as conn:
        # ── Cost by provider ──────────────────────────────────────────
        rows = conn.execute(
            "SELECT provider, ROUND(SUM(cost_usd),2) AS total "
            "FROM daily_billing GROUP BY provider ORDER BY total DESC"
        ).fetchall()
        cost_by_provider = [{"name": r["provider"].upper(), "value": r["total"]} for r in rows]

        # ── Cost by service (top 8) ───────────────────────────────────
        rows = conn.execute(
            "SELECT service, ROUND(SUM(cost_usd),2) AS total "
            "FROM daily_billing GROUP BY service ORDER BY total DESC LIMIT 8"
        ).fetchall()
        cost_by_service = [{"name": r["service"], "value": r["total"]} for r in rows]

        # ── Daily cost trend ──────────────────────────────────────────
        rows = conn.execute(
            "SELECT date, ROUND(SUM(cost_usd),2) AS total "
            "FROM daily_billing GROUP BY date ORDER BY date"
        ).fetchall()
        cost_trend = [{"date": r["date"], "cost": r["total"]} for r in rows]

        # ── Cost by team ──────────────────────────────────────────────
        rows = conn.execute(
            "SELECT team, ROUND(SUM(cost_usd),2) AS total "
            "FROM daily_billing GROUP BY team ORDER BY total DESC"
        ).fetchall()
        cost_by_team = [{"name": r["team"], "value": r["total"]} for r in rows]

        # ── Anomaly severity breakdown ────────────────────────────────
        rows = conn.execute(
            "SELECT severity, COUNT(*) AS n FROM detected_anomalies "
            "GROUP BY severity ORDER BY CASE severity "
            "WHEN 'CRITICAL' THEN 1 WHEN 'HIGH' THEN 2 "
            "WHEN 'MEDIUM' THEN 3 ELSE 4 END"
        ).fetchall()
        anomaly_by_severity = [{"name": r["severity"], "value": r["n"]} for r in rows]

        # ── Anomalies by detector ─────────────────────────────────────
        rows = conn.execute(
            "SELECT detector, COUNT(*) AS n FROM detected_anomalies GROUP BY detector"
        ).fetchall()
        anomaly_by_detector = [{"name": r["detector"], "value": r["n"]} for r in rows]

        # ── Anomalies by provider ─────────────────────────────────────
        rows = conn.execute(
            "SELECT provider, COUNT(*) AS n FROM detected_anomalies "
            "GROUP BY provider ORDER BY n DESC"
        ).fetchall()
        anomaly_by_provider = [{"name": r["provider"].upper(), "value": r["n"]} for r in rows]

        # ── Anomalies by service (top 10) ─────────────────────────────
        rows = conn.execute(
            "SELECT service, COUNT(*) AS n FROM detected_anomalies "
            "GROUP BY service ORDER BY n DESC LIMIT 10"
        ).fetchall()
        anomaly_by_service = [{"name": r["service"], "value": r["n"]} for r in rows]

        # ── Anomalies by team ─────────────────────────────────────────
        rows = conn.execute(
            "SELECT team, COUNT(*) AS n FROM detected_anomalies "
            "GROUP BY team ORDER BY n DESC"
        ).fetchall()
        anomaly_by_team = [{"name": r["team"], "value": r["n"]} for r in rows]

        # ── Anomalies by region ───────────────────────────────────────
        try:
            rows = conn.execute(
                "SELECT region, COUNT(*) AS n FROM detected_anomalies "
                "GROUP BY region ORDER BY n DESC"
            ).fetchall()
            anomaly_by_region = [{"name": r["region"], "value": r["n"]} for r in rows]
        except Exception:
            anomaly_by_region = []

        # ── Spend by environment ──────────────────────────────────────
        rows = conn.execute(
            "SELECT environment, ROUND(SUM(cost_usd),2) AS total "
            "FROM daily_billing GROUP BY environment ORDER BY total DESC"
        ).fetchall()
        spend_by_env = [{"name": r["environment"], "value": r["total"]} for r in rows]

        # ── Spend by region ───────────────────────────────────────────
        try:
            rows = conn.execute(
                "SELECT region, ROUND(SUM(cost_usd),2) AS total "
                "FROM daily_billing GROUP BY region ORDER BY total DESC LIMIT 12"
            ).fetchall()
            spend_by_region = [{"name": r["region"], "value": r["total"]} for r in rows]
        except Exception:
            spend_by_region = []

        # ── Normal vs Anomaly counts ──────────────────────────────────
        total_rows = conn.execute("SELECT COUNT(*) AS n FROM daily_billing").fetchone()["n"]
        total_anomalies = conn.execute("SELECT COUNT(*) AS n FROM detected_anomalies").fetchone()["n"]
        normal_count = max(0, total_rows - total_anomalies)
        normal_vs_anomaly = [
            {"name": "Normal", "value": normal_count},
            {"name": "Anomaly", "value": total_anomalies},
        ]

        # ── Time-series with anomaly flag + rolling average ───────────
        ts_rows = conn.execute(
            "SELECT date, ROUND(SUM(cost_usd),2) AS total "
            "FROM daily_billing GROUP BY date ORDER BY date"
        ).fetchall()
        anom_dates = {r["date"] for r in conn.execute(
            "SELECT DISTINCT date FROM detected_anomalies"
        ).fetchall()}
        ts_data = []
        window: list = []
        for r in ts_rows:
            window.append(r["total"])
            if len(window) > 7:
                window.pop(0)
            rolling_avg = round(sum(window) / len(window), 2)
            ts_data.append({
                "date": r["date"],
                "cost": r["total"],
                "rolling_avg": rolling_avg,
                "is_anomaly": r["date"] in anom_dates,
                "anomaly_cost": r["total"] if r["date"] in anom_dates else None,
            })
        cost_time_series_with_anomalies = ts_data

        # ── Forecast comparison (actual vs expected from anomalies) ───
        fc_rows = conn.execute(
            "SELECT date, ROUND(AVG(cost_usd),2) AS actual, "
            "ROUND(AVG(expected_cost),2) AS predicted "
            "FROM detected_anomalies GROUP BY date ORDER BY date"
        ).fetchall()
        forecast_comparison = [
            {"date": r["date"], "actual": r["actual"], "predicted": r["predicted"]}
            for r in fc_rows
        ]

        # ── Detailed anomaly table (top 100) ──────────────────────────
        da_rows = conn.execute(
            "SELECT date, provider, service, team, environment, "
            "ROUND(cost_usd,2) AS cost_usd, ROUND(expected_cost,2) AS expected_cost, "
            "ROUND(deviation_pct,1) AS deviation_pct, severity, anomaly_type, detector, "
            "ROUND(ABS(deviation_pct)/100.0, 4) AS anomaly_score, description "
            "FROM detected_anomalies ORDER BY ABS(deviation_pct) DESC LIMIT 100"
        ).fetchall()
        detailed_anomalies = [dict(r) for r in da_rows]

        # ── Provider+service stacked breakdown ────────────────────────
        ps_rows = conn.execute(
            "SELECT provider, service, COUNT(*) AS n FROM detected_anomalies "
            "GROUP BY provider, service ORDER BY provider, n DESC"
        ).fetchall()
        provider_service_breakdown = [
            {"provider": r["provider"].upper(), "service": r["service"], "count": r["n"]}
            for r in ps_rows
        ]

        total_cost = conn.execute("SELECT ROUND(SUM(cost_usd),2) AS s FROM daily_billing").fetchone()["s"] or 0
        anomaly_cost = conn.execute(
            "SELECT ROUND(SUM(cost_usd),2) AS s FROM detected_anomalies"
        ).fetchone()["s"] or 0
        savings = conn.execute(
            "SELECT ROUND(SUM(cost_usd - expected_cost),2) AS s FROM detected_anomalies "
            "WHERE deviation_pct > 0 AND severity IN ('HIGH','CRITICAL')"
        ).fetchone()["s"] or 0

        # ── Top 5 anomalies ───────────────────────────────────────────
        rows = conn.execute(
            "SELECT date, provider, service, team, severity, "
            "ROUND(cost_usd,2) AS cost, ROUND(deviation_pct,1) AS deviation "
            "FROM detected_anomalies ORDER BY ABS(deviation_pct) DESC LIMIT 5"
        ).fetchall()
        top_anomalies = [dict(r) for r in rows]

    return {
        "cost_by_provider":     cost_by_provider,
        "cost_by_service":      cost_by_service,
        "cost_trend":           cost_trend,
        "cost_by_team":         cost_by_team,
        "anomaly_by_severity":  anomaly_by_severity,
        "anomaly_by_detector":  anomaly_by_detector,
        "anomaly_by_provider":  anomaly_by_provider,
        "anomaly_by_service":   anomaly_by_service,
        "anomaly_by_team":      anomaly_by_team,
        "anomaly_by_region":    anomaly_by_region,
        "spend_by_env":         spend_by_env,
        "spend_by_region":      spend_by_region,
        "normal_vs_anomaly":    normal_vs_anomaly,
        "cost_time_series_with_anomalies": cost_time_series_with_anomalies,
        "forecast_comparison":  forecast_comparison,
        "detailed_anomalies":   detailed_anomalies,
        "provider_service_breakdown": provider_service_breakdown,
        "top_anomalies":        top_anomalies,
        "model_stats": {
            "total_rows":       total_rows,
            "total_anomalies":  total_anomalies,
            "total_cost":       total_cost,
            "anomaly_cost":     anomaly_cost,
            "savings":          savings,
            "detection_rate":   round(total_anomalies / total_rows * 100, 1) if total_rows else 0,
            "models_used":      ["Z-Score + STL Decomposition", "Isolation Forest"],
            "ensemble_method":  "Multi-detector consensus",
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Port-free helper  (Windows — avoids WinError 10048 on restart)
# ══════════════════════════════════════════════════════════════════════════════

def _free_port(port: int) -> None:
    """
    On Windows, find any process listening on *port* and kill it so uvicorn
    can bind cleanly.  Safe to call even when the port is already free.
    Uses only stdlib (subprocess + time) — no extra deps needed.
    """
    import subprocess, time, os

    if os.name != "nt":          # POSIX systems don't need this
        return

    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=5,
        )
        killed_any = False
        for line in result.stdout.splitlines():
            # Look for lines like:  TCP  0.0.0.0:8000  ...  LISTENING  1234
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1].strip()
                if pid and pid != "0":
                    subprocess.run(
                        ["taskkill", "/F", "/PID", pid],
                        capture_output=True, timeout=5,
                    )
                    log.info("Released port %d — killed PID %s", port, pid)
                    killed_any = True
        if killed_any:
            time.sleep(0.8)   # give Windows time to release the socket
    except Exception as exc:
        log.warning("_free_port(%d) skipped: %s", port, exc)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="FinOps Phase 4 - FastAPI Server")
    parser.add_argument("--host",   default="0.0.0.0",  help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",   default=8000, type=int, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true",  help="Enable hot-reload (dev mode)")
    args = parser.parse_args()

    # ── Auto-release the port if a previous instance is still holding it ──────
    _free_port(args.port)

    log.info("Starting FinOps API on http://%s:%d", args.host, args.port)
    log.info("Interactive docs: http://localhost:%d/docs", args.port)

    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )

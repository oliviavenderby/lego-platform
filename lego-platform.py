# find_next_buy.py
# ReUseBricks — "Find Next Buy" with Quick Set Lookup (v1.2)
#
# Type a LEGO set number, we fetch live data from:
# - Rebrickable (name/theme/year/image)
# - Brickset (MSRP + launch/exit dates) — optional
# - BrickLink (price guide: stock/sold, new/used)
# Then we drop it into your candidate pool and run the scoring/basket/export flow.
#
# Requirements:
#   pip install streamlit pandas numpy python-dateutil requests requests-oauthlib
#
# Notes:
# - BrickLink uses OAuth1 (Consumer Key/Secret + Token/Secret).
# - Rebrickable uses an API key in the HTTP Authorization header: "Authorization: key <KEY>".
# - Brickset uses an API key and a simple POST/GET to /api/v3.asmx/getSets.
#
# ⚠️ You control what goes into the score; many signals are optional and gracefully default.

from __future__ import annotations

import io
import json
import math
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dateutil.relativedelta import relativedelta
from requests_oauthlib import OAuth1
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ---------- Page setup ----------
st.set_page_config(page_title="ReUseBricks – Find Next Buy", page_icon="🧱", layout="wide")
st.title("🧱 Find Next Buy")
st.caption("Type a set number → fetch live data → score → build a budget‑aware basket. Export to BrickLink Wanted List or CSV.")

# ---------- Constants & helpers ----------
ID_COL = "set_num"
NAME_COL = "name"
THEME_COL = "theme"
DATE_COL = "release_date"

NUM_COLS = [
    "retail_price",
    "current_new_price",
    "current_used_price",
    "part_out_value",
    "units_sold_30d",
    "active_sellers",
    "exclusive_minifigs",
    "box_volume_l",
    "avg_discount",
    "rerelease_risk",
    "theme_avg_growth",
    "trend_90d",
]

DEFAULT_THEME_GROWTH = 0.07
DEFAULT_TREND_90D = 0.02

LICENSED_THEMES = {
    "Star Wars", "Harry Potter", "Marvel Super Heroes", "DC Super Heroes",
    "Disney", "Jurassic World", "Sonic the Hedgehog", "Minecraft",
    "Avatar", "Lord of the Rings", "Super Mario", "Wicked"
}

def is_licensed_theme(theme_name: str) -> int:
    return int(any(t.lower() in theme_name.lower() for t in LICENSED_THEMES))

def normalize_set_num(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    return s if "-" in s else f"{s}-1"

def _safe_bool01(x) -> int:
    if isinstance(x, bool):
        return int(x)
    try:
        return 1 if str(x).strip().lower() in {"1", "true", "t", "yes", "y"} else 0
    except Exception:
        return 0

def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def _minmax_0_100(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.full(len(s), 50.0), index=s.index)
    return (s - lo) / (hi - lo) * 100.0

def _money(x: float) -> str:
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return "-"

def _today() -> datetime:
    return datetime.utcnow()

# ---------- Session state ----------
if "candidates" not in st.session_state:
    st.session_state["candidates"] = pd.DataFrame(columns=[ID_COL, NAME_COL, THEME_COL, DATE_COL] + NUM_COLS)

# ---------- Sidebar: keys, constraints, weights ----------
with st.sidebar:
    st.subheader("API keys")
    colk1, colk2 = st.columns(2)
    with colk1:
        bl_consumer_key = st.text_input("BrickLink consumer key", help="From BrickLink API settings")
        bl_token_value = st.text_input("BrickLink token value")
    with colk2:
        bl_consumer_secret = st.text_input("BrickLink consumer secret", type="password")
        bl_token_secret = st.text_input("BrickLink token secret", type="password")
    rb_key = st.text_input("Rebrickable API key", help="Header: Authorization: key <KEY>")
    bs_key = st.text_input("Brickset API key (optional)", help="Used to fetch MSRP + launch/exit dates")

    st.markdown("---")
    st.subheader("Your constraints")
    colA, colB = st.columns(2)
    with colA:
        budget_total = st.number_input("Total budget (USD)", min_value=0.0, value=500.0, step=50.0)
    with colB:
        per_set_cap = st.checkbox("Only show sets ≤ target buy cap", value=True)
    horizon_months = st.number_input("Holding horizon (months)", min_value=1, value=12, step=1)
    desired_discount_pct = st.slider("Desired discount off retail", 0, 50, 20, 1)
    pref = st.radio("Path preference", ["Neutral", "Sealed", "Part‑out"], horizontal=True, index=0)
    pref_weight_sealed = {"Neutral": 0.5, "Sealed": 1.0, "Part‑out": 0.0}[pref]
    liquidity_floor = st.slider("Min 30‑day units sold", 0, 400, 50, 10)
    exclude_licensed = st.checkbox("Exclude licensed themes", value=False)

    st.markdown("---")
    st.subheader("Weights")
    w_growth = st.slider("Growth", 0.0, 1.0, 0.30, 0.01)
    w_liq = st.slider("Liquidity", 0.0, 1.0, 0.25, 0.01)
    w_margin = st.slider("Margin", 0.0, 1.0, 0.20, 0.01)
    w_risk = st.slider("Risk penalty", 0.0, 1.0, 0.15, 0.01)
    w_ops = st.slider("Operational penalty", 0.0, 1.0, 0.10, 0.01)
    weights = dict(growth=w_growth, liquidity=w_liq, margin=w_margin, risk=w_risk, operational=w_ops)

    st.markdown("---")
    st.subheader("Load from CSV (optional)")
    csv_file = st.file_uploader("Upload candidate sets CSV", type=["csv"])
    st.caption("If omitted, you can build a pool by typing set numbers below.")

# ---------- API helpers (cached) ----------
@st.cache_data(show_spinner=False)
def rb_get(path: str, key: str, params: Optional[dict] = None):
    if not key:
        raise ValueError("Rebrickable key required")
    headers = {"Authorization": f"key {key}"}
    url = f"https://rebrickable.com/api/v3{path}"
    r = requests.get(url, headers=headers, params=params or {}, timeout=20)
    if r.status_code == 429:
        # polite backoff if throttled
        time.sleep(2.0)
        r = requests.get(url, headers=headers, params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()

def bl_auth(ck, cs, tv, ts) -> OAuth1:
    if not all([ck, cs, tv, ts]):
        raise ValueError("BrickLink OAuth1 credentials required")
    return OAuth1(ck, cs, tv, ts, signature_type="auth_header")

@st.cache_data(show_spinner=False)
def bl_get(path: str, params: Optional[dict], ck: str, cs: str, tv: str, ts: str):
    url = f"https://api.bricklink.com/api/store/v1{path}"
    auth = bl_auth(ck, cs, tv, ts)
    r = requests.get(url, params=params or {}, auth=auth, timeout=25)
    if r.status_code == 429:
        time.sleep(1.5)
        r = requests.get(url, params=params or {}, auth=auth, timeout=25)
    r.raise_for_status()
    data = r.json()
    # BrickLink wraps payload in {"meta":{...}, "data":{...}}
    return data.get("data", {})

@st.cache_data(show_spinner=False)
def bs_get_sets(set_number: str, api_key: str):
    """Brickset getSets: returns [] or list of set dicts (we want retail & dates)."""
    if not api_key:
        return []
    url = "https://brickset.com/api/v3.asmx/getSets"
    # Brickset supports GET or POST; using GET is OK here for small params
    params = {
        "apiKey": api_key,
        "params": json.dumps({"setNumber": set_number, "pageSize": 1, "pageNumber": 1, "extendedData": 0}),
    }
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    payload = r.json()
    if payload.get("status") != "success":
        return []
    return payload.get("sets", []) or []

# ---------- Data enrichment & scoring ----------
def enrich_release_months(dt: Optional[datetime], year_fallback: Optional[int]) -> int:
    today = _today()
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except Exception:
            dt = None
    if isinstance(dt, datetime):
        months = (today.year - dt.year) * 12 + (today.month - dt.month)
        return max(0, months)
    if year_fallback:
        approx = datetime(year_fallback, 1, 1)
        return max(0, (today.year - approx.year) * 12 + (today.month - approx.month))
    return 0

def compute_subscores(df: pd.DataFrame, preference_sealed_weight: float) -> pd.DataFrame:
    out = df.copy()
    growth = (0.5 * _zscore(out["theme_avg_growth"]) +
              0.3 * _zscore(out["trend_90d"]) +
              0.2 * _zscore(out["release_months"]))
    liquidity = 0.6 * _zscore(out["units_sold_30d"]) + 0.4 * _zscore(out["active_sellers"])
    margin = (1 - preference_sealed_weight) * _zscore(out["part_out_ratio"]) + \
             preference_sealed_weight * _zscore(out["sealed_margin_pct"])
    risk = (0.5 * _zscore(out["rerelease_risk"]) +
            0.3 * _zscore(out["licensed"]) +
            0.2 * _zscore(1.0 - out["avg_discount"]))
    operational = 0.7 * _zscore(out["box_volume_l"])  # handling complexity placeholder
    out["growth_sub"] = growth
    out["liquidity_sub"] = liquidity
    out["margin_sub"] = margin
    out["risk_sub"] = risk
    out["operational_sub"] = operational
    return out

def compute_scores(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    out["raw_score"] = (
        weights["growth"] * out["growth_sub"] +
        weights["liquidity"] * out["liquidity_sub"] +
        weights["margin"] * out["margin_sub"] -
        weights["risk"] * out["risk_sub"] -
        weights["operational"] * out["operational_sub"]
    )
    out["score"] = _minmax_0_100(out["raw_score"])
    return out

def suggest_basket(df: pd.DataFrame, budget_total: float) -> Tuple[pd.DataFrame, float]:
    items, spend = [], 0.0
    for _, row in df.sort_values("score", ascending=False).iterrows():
        price = float(row["target_buy_price"])
        if price <= 0 or math.isnan(price):
            continue
        if spend + price <= budget_total:
            items.append(row)
            spend += price
    if not items:
        return pd.DataFrame(columns=df.columns), 0.0
    return pd.DataFrame(items), spend

def build_wanted_list_xml(rows: pd.DataFrame, condition: str = "N", qty: int = 1, list_name: Optional[str] = None) -> bytes:
    inv = ET.Element("INVENTORY")
    if list_name:
        wl = ET.SubElement(inv, "WANTEDLIST")
        ET.SubElement(wl, "NAME").text = str(list_name)
    for _, r in rows.iterrows():
        item = ET.SubElement(inv, "ITEM")
        ET.SubElement(item, "ITEMTYPE").text = "S"
        ET.SubElement(item, "ITEMID").text = str(r[ID_COL])
        ET.SubElement(item, "CONDITION").text = condition
        ET.SubElement(item, "MINQTY").text = str(qty)
        ET.SubElement(item, "MAXPRICE").text = f"{float(r['target_buy_price']):.2f}"
    pretty = minidom.parseString(ET.tostring(inv, encoding="utf-8")).toprettyxml(indent="  ", encoding="utf-8")
    return pretty

def rows_to_buylist_csv(rows: pd.DataFrame) -> bytes:
    export = rows.copy()
    if "sealed_roi_pct" not in export.columns:
        export["sealed_roi_pct"] = (export["current_new_price"] - export["target_buy_price"]) / export["target_buy_price"]
    if "part_out_roi_pct" not in export.columns:
        export["part_out_roi_pct"] = (0.7 * export["part_out_value"] - export["target_buy_price"]) / export["target_buy_price"]
    cols = [
        "set_num", "name", "theme",
        "target_buy_price", "retail_price", "current_new_price",
        "part_out_value", "sealed_roi_pct", "part_out_roi_pct", "score"
    ]
    return export[cols].to_csv(index=False).encode("utf-8")

def explain_row(r: pd.Series, med: Dict[str, float]) -> List[str]:
    msgs = []
    if r["theme_avg_growth"] >= med["theme_avg_growth"]:
        msgs.append("Theme momentum above peer median")
    if r["trend_90d"] >= med["trend_90d"]:
        msgs.append("Positive 90‑day trend")
    if r["release_months"] >= med["release_months"]:
        msgs.append("Older release (closer to/after EOL)")
    if r["units_sold_30d"] >= med["units_sold_30d"]:
        msgs.append("Moves quickly (30‑day velocity)")
    if r["active_sellers"] >= med["active_sellers"]:
        msgs.append("Depth of sellers")
    if r["sealed_margin_pct"] >= med["sealed_margin_pct"]:
        msgs.append("Healthy sealed margin vs retail")
    if r["part_out_ratio"] >= med["part_out_ratio"]:
        msgs.append("Strong part‑out/retail ratio")
    if r["rerelease_risk"] <= med["rerelease_risk"]:
        msgs.append("Lower re‑release risk")
    if r["box_volume_l"] <= med["box_volume_l"]:
        msgs.append("Compact to store")
    if _safe_bool01(r.get("licensed", 0)) == 1:
        msgs.append("Licensed IP (can be more volatile)")
    return msgs

# ---------- Quick add by set number ----------
st.subheader("Quick add by set number")
with st.expander("Lookup & add a set", expanded=True):
    c1, c2 = st.columns([2, 1])
    with c1:
        user_set_input = st.text_input("Enter set number (e.g., 10265-1 or 10265)")
    with c2:
        avg_discount_override = st.slider("Assumed avg retail discount", 0, 40, 10, 1, help="Used in risk calc if we don't have discount history")

    col_go1, col_go2, col_go3 = st.columns([1, 1, 2])
    fetch_clicked = col_go1.button("Fetch info")
    add_clicked = col_go2.button("Add to candidate pool")

    st.caption("Tip: you can add multiple sets one after another, then scroll down to score them.")

# This dict will hold the most recent fetch so "Add" can use it
if "last_fetch_row" not in st.session_state:
    st.session_state["last_fetch_row"] = None

def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)

def fetch_set_row(set_num_raw: str) -> Optional[pd.Series]:
    set_num = normalize_set_num(set_num_raw)
    if not set_num:
        st.warning("Enter a set number.")
        return None

    # --- Rebrickable metadata ---
    rb_data = {}
    theme_name = ""
    year = None
    image_url = ""
    try:
        if rb_key:
            rb_set = rb_get(f"/lego/sets/{set_num}/", key=rb_key)
            rb_data = rb_set or {}
            year = rb_data.get("year")
            image_url = rb_data.get("set_img_url") or ""
            theme_id = rb_data.get("theme_id")
            if theme_id:
                rb_theme = rb_get(f"/lego/themes/{theme_id}/", key=rb_key)
                theme_name = rb_theme.get("name", "") if isinstance(rb_theme, dict) else ""
    except Exception as e:
        st.info(f"Rebrickable lookup skipped/failed: {e}")

    # --- Brickset MSRP + dates (optional) ---
    msrp = None
    launch_dt = None
    exit_dt = None
    try:
        if bs_key:
            bs_sets = bs_get_sets(set_num, bs_key)
            if bs_sets:
                bs0 = bs_sets[0]
                lego_us = (bs0.get("LEGOCom") or {}).get("US") or {}
                msrp = lego_us.get("retailPrice")
                launch_dt = lego_us.get("dateFirstAvailable") or bs0.get("launchDate")
                exit_dt = lego_us.get("dateLastAvailable") or bs0.get("exitDate")
    except Exception as e:
        st.info(f"Brickset lookup skipped/failed: {e}")

    # --- BrickLink price guides ---
    stock_new = sold_new = stock_used = sold_used = {}
    active_listings = 0
    sold_total_6mo = 0
    try:
        if all([bl_consumer_key, bl_consumer_secret, bl_token_value, bl_token_secret]):
            # Catalog sanity (optional)
            bl_item = bl_get(f"/items/set/{set_num}", None, bl_consumer_key, bl_consumer_secret, bl_token_value, bl_token_secret)

            stock_new = bl_get(
                f"/items/set/{set_num}/price",
                {"guide_type": "stock", "new_or_used": "N", "currency_code": "USD"},
                bl_consumer_key, bl_consumer_secret, bl_token_value, bl_token_secret
            )
            sold_new = bl_get(
                f"/items/set/{set_num}/price",
                {"guide_type": "sold", "new_or_used": "N", "currency_code": "USD"},
                bl_consumer_key, bl_consumer_secret, bl_token_value, bl_token_secret
            )
            stock_used = bl_get(
                f"/items/set/{set_num}/price",
                {"guide_type": "stock", "new_or_used": "U", "currency_code": "USD"},
                bl_consumer_key, bl_consumer_secret, bl_token_value, bl_token_secret
            )
            sold_used = bl_get(
                f"/items/set/{set_num}/price",
                {"guide_type": "sold", "new_or_used": "U", "currency_code": "USD"},
                bl_consumer_key, bl_consumer_secret, bl_token_value, bl_token_secret
            )

            # Approximations
            active_listings = len((stock_new or {}).get("price_detail") or [])
            sold_total_6mo = safe_float((sold_new or {}).get("total_quantity"), 0.0)
    except Exception as e:
        st.info(f"BrickLink lookup skipped/failed: {e}")

    # Build one candidate row
    retail_price = safe_float(msrp, 0.0)
    current_new_price = safe_float((sold_new or {}).get("avg_price") or (stock_new or {}).get("avg_price"), 0.0)
    current_used_price = safe_float((sold_used or {}).get("avg_price") or (stock_used or {}).get("avg_price"), 0.0)
    units_sold_30d = sold_total_6mo / 6.0  # rough monthly proxy from 6‑month total
    release_months = enrich_release_months(launch_dt, year)
    sealed_margin_pct = (current_new_price - retail_price) / retail_price if retail_price else 0.0

    row = {
        "set_num": set_num,
        "name": rb_data.get("name") or "",
        "theme": theme_name or "",
        "retail_price": retail_price,
        "current_new_price": current_new_price,
        "current_used_price": current_used_price,
        "part_out_value": np.nan,  # optional later (slow)
        "units_sold_30d": units_sold_30d,
        "active_sellers": active_listings,
        "release_date": launch_dt or (f"{year}-01-01" if year else None),
        "licensed": is_licensed_theme(theme_name),
        "exclusive_minifigs": 0,  # can be filled later
        "box_volume_l": np.nan,   # filled if Brickset dimensions available later
        "avg_discount": avg_discount_override / 100.0,
        "rerelease_risk": 0.15 if is_licensed_theme(theme_name) else 0.05,
        "theme_avg_growth": DEFAULT_THEME_GROWTH,
        "trend_90d": DEFAULT_TREND_90D,
        # derived for scoring:
        "release_months": release_months,
        "sealed_margin_pct": sealed_margin_pct,
        "part_out_ratio": np.nan,
        "_image_url": image_url,
    }
    return pd.Series(row)

# Fetch flow
if fetch_clicked:
    sr = fetch_set_row(user_set_input)
    if sr is not None:
        st.session_state["last_fetch_row"] = sr

# Show fetched info card
if st.session_state.get("last_fetch_row") is not None:
    sr = st.session_state["last_fetch_row"]
    st.success(f"Fetched: {sr['set_num']} — {sr['name'] or '(name pending)'}")
    img = sr.get("_image_url", "")
    cA, cB, cC = st.columns([1, 2, 2])
    with cA:
        if img:
            st.image(img, use_container_width=True)
        st.metric("MSRP", _money(sr["retail_price"]))
        st.metric("New price (avg)", _money(sr["current_new_price"]))
        st.metric("Used price (avg)", _money(sr["current_used_price"]))
    with cB:
        st.write(f"**Theme**: {sr['theme'] or '—'}")
        st.write(f"**Licensed**: {'Yes' if sr['licensed'] else 'No'}")
        st.write(f"**Release months**: {int(sr['release_months'])}")
        st.write(f"**Velocity (est. 30d)**: {int(sr['units_sold_30d'] or 0)} units")
        st.write(f"**Active listings**: {int(sr['active_sellers'] or 0)}")
    with cC:
        tgt = sr["retail_price"] * (1 - desired_discount_pct / 100.0) if sr["retail_price"] else 0.0
        sealed_roi = (sr["current_new_price"] - tgt) / tgt if tgt else 0.0
        st.metric("Target buy", _money(tgt))
        st.metric("Sealed ROI vs target", f"{sealed_roi*100:.1f}%")
        st.caption("Part‑out value is optional (you can add later).")

# Add to pool
if add_clicked and st.session_state.get("last_fetch_row") is not None:
    sr = st.session_state["last_fetch_row"]
    df = st.session_state["candidates"].copy()
    # ensure required columns exist
    for col in [ID_COL, NAME_COL, THEME_COL, DATE_COL] + NUM_COLS + ["release_months", "sealed_margin_pct", "part_out_ratio"]:
        if col not in df.columns:
            df[col] = np.nan
    # append/replace by set_num
    df = df[df[ID_COL] != sr[ID_COL]]
    df = pd.concat([df, pd.DataFrame([sr])], ignore_index=True)
    st.session_state["candidates"] = df
    st.success(f"Added {sr['set_num']} to candidate pool.")

# ---------- Optional CSV import to seed pool ----------
if csv_file:
    try:
        imported = pd.read_csv(csv_file)
        st.session_state["candidates"] = imported
        st.success(f"Loaded {len(imported)} rows from CSV.")
    except Exception as e:
        st.error(f"CSV load failed: {e}")

# ---------- Prepare candidate pool ----------
df = st.session_state["candidates"].copy()

# If pool is empty, show hint and stop
if df.empty:
    st.info("Your candidate pool is empty. Add sets with the lookup above or upload a CSV.")
    st.stop()

# Coerce types and compute derived
if DATE_COL in df.columns:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
else:
    df[DATE_COL] = pd.NaT

# Fill missing numerics
for col in NUM_COLS:
    if col not in df.columns:
        df[col] = np.nan
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Derived fields
df["release_months"] = df.apply(
    lambda r: enrich_release_months(r.get(DATE_COL), None) if pd.isna(r.get("release_months", np.nan)) else r["release_months"],
    axis=1
)
df["sealed_margin_pct"] = df.apply(
    lambda r: ((r["current_new_price"] - r["retail_price"]) / r["retail_price"]) if (r.get("sealed_margin_pct") is np.nan or pd.isna(r.get("sealed_margin_pct"))) and r["retail_price"] else r.get("sealed_margin_pct", 0.0),
    axis=1
)
df["part_out_ratio"] = df.apply(
    lambda r: (r["part_out_value"] / r["retail_price"]) if (r.get("part_out_ratio") is np.nan or pd.isna(r.get("part_out_ratio"))) and r["retail_price"] and not pd.isna(r["part_out_value"]) else r.get("part_out_ratio", np.nan),
    axis=1
)
df["licensed"] = df.get("licensed", 0).apply(_safe_bool01)

# Target buy price from desired discount
df["target_buy_price"] = df["retail_price"] * (1 - desired_discount_pct / 100.0)

# Filters
mask = pd.Series(True, index=df.index)
if exclude_licensed:
    mask &= df["licensed"] == 0
mask &= (df["units_sold_30d"].fillna(0) >= liquidity_floor)
if per_set_cap:
    mask &= df["target_buy_price"].fillna(0) <= budget_total

filtered = df.loc[mask].copy()
if filtered.empty:
    st.warning("No sets match your current filters. Adjust liquidity/budget or add more sets.")
    st.stop()

# Subscores + score
scored = compute_subscores(filtered, preference_sealed_weight=pref_weight_sealed)
scored = compute_scores(scored, weights=weights)
scored["sealed_roi_pct"] = (scored["current_new_price"] - scored["target_buy_price"]) / scored["target_buy_price"]
scored["part_out_roi_pct"] = (0.7 * scored["part_out_value"] - scored["target_buy_price"]) / scored["target_buy_price"]

# ---------- Display: candidates + picks ----------
st.subheader(f"Candidates ready ({len(scored)}/{len(df)})")
mini = scored[[ID_COL, NAME_COL, THEME_COL, "retail_price", "current_new_price", "units_sold_30d"]].copy()
st.dataframe(mini.reset_index(drop=True), use_container_width=True, height=220)

st.subheader("Top picks")
display_cols = [
    "score", "set_num", "name", "theme",
    "target_buy_price", "retail_price", "current_new_price", "sealed_roi_pct",
    "part_out_value", "part_out_roi_pct", "units_sold_30d", "active_sellers", "release_date"
]
pretty = scored[display_cols].sort_values("score", ascending=False).copy()
pretty["score"] = pretty["score"].round(1)
for mcol in ["target_buy_price", "retail_price", "current_new_price", "part_out_value"]:
    pretty[mcol] = pretty[mcol].map(_money)
for pcol in ["sealed_roi_pct", "part_out_roi_pct"]:
    pretty[pcol] = (pretty[pcol] * 100.0).map(lambda x: f"{x:.1f}%")
st.dataframe(pretty.reset_index(drop=True), use_container_width=True, height=420)

st.subheader("Why these scored well")
meds = {
    "theme_avg_growth": scored["theme_avg_growth"].median(),
    "trend_90d": scored["trend_90d"].median(),
    "release_months": scored["release_months"].median(),
    "units_sold_30d": scored["units_sold_30d"].median(),
    "active_sellers": scored["active_sellers"].median(),
    "sealed_margin_pct": scored["sealed_margin_pct"].median(),
    "part_out_ratio": scored["part_out_ratio"].median(),
    "rerelease_risk": scored["rerelease_risk"].median(),
    "box_volume_l": scored["box_volume_l"].median() if "box_volume_l" in scored.columns else 0,
}
for _, row in scored.sort_values("score", ascending=False).head(3).iterrows():
    with st.expander(f"#{row['set_num']} — {row['name']}  (Score {row['score']:.1f})"):
        bullets = explain_row(row, meds)
        st.write("\n".join([f"• {b}" for b in bullets]) or "• Balanced profile across growth, liquidity, and margin.")

# ---------- Basket & exports ----------
st.subheader("Suggested basket under your budget")
basket, spend = suggest_basket(scored, budget_total)
if basket.empty:
    st.info("No combination of sets fits under your total budget. Increase the budget or desired discount.")
else:
    use_sealed = pref_weight_sealed >= 0.5
    basket = basket.copy()
    basket["est_profit"] = (basket["current_new_price"] - basket["target_buy_price"]) if use_sealed \
        else (0.7 * basket["part_out_value"] - basket["target_buy_price"])
    total_profit = float(basket["est_profit"].sum())
    est_roi_pct = total_profit / spend if spend > 0 else 0.0

    view = basket[["score", "set_num", "name", "theme", "target_buy_price", "est_profit"]].sort_values("score", ascending=False)
    view["target_buy_price"] = view["target_buy_price"].map(_money)
    view["est_profit"] = view["est_profit"].map(_money)
    st.dataframe(view.reset_index(drop=True), use_container_width=True, height=260)

    c1, c2, c3 = st.columns(3)
    c1.metric("Planned spend", _money(spend))
    c2.metric("Estimated profit", _money(total_profit))
    c3.metric("Estimated ROI (basket)", f"{est_roi_pct * 100:.1f}%")

    st.markdown("#### Export")
    d1, d2, d3 = st.columns([2, 2, 3])
    with d1:
        wanted_name = st.text_input("Wanted list name", value="ReUseBricks Buylist")
    with d2:
        cond = st.radio("Condition", ["New (N)", "Used (U)"], horizontal=True)
        cond_code = "N" if cond.startswith("New") else "U"
    with d3:
        qty = st.number_input("Qty per set", min_value=1, value=1, step=1)

    xml_bytes = build_wanted_list_xml(basket, condition=cond_code, qty=qty, list_name=wanted_name)
    csv_bytes = rows_to_buylist_csv(basket)

    e1, e2 = st.columns(2)
    with e1:
        st.download_button("⬇️ BrickLink Wanted List (XML)", data=xml_bytes, file_name="wanted_list.xml", mime="application/xml")
    with e2:
        st.download_button("⬇️ Buylist CSV", data=csv_bytes, file_name="buylist.csv", mime="text/csv")

st.markdown("---")
with st.expander("CSV schema (columns)"):
    st.code(
        """set_num,name,theme,retail_price,current_new_price,current_used_price,part_out_value,
units_sold_30d,active_sellers,release_date(YYYY-MM-DD),licensed(0/1),
exclusive_minifigs,box_volume_l,avg_discount,rerelease_risk,theme_avg_growth,trend_90d
""",
        language="text",
    )

st.caption(
    "Scoring math: score = minmax( w_g*growth + w_l*liquidity + w_m*margin − w_r*risk − w_o*operational ). "
    "This is NOT financial advice."
)

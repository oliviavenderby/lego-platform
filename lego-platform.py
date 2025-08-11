# find_next_buy.py
# ReUseBricks — "Find Next Buy" Streamlit page (v1)
#
# What it does:
# - Loads a dataset of candidate LEGO sets
# - Lets the user tune investment constraints and weights
# - Scores each set (0–100) with explainability
# - Suggests a best basket of sets under your budget
# - Exports a BrickLink Wanted List (XML) + a Buylist CSV
#
# How to run:
#   pip install streamlit pandas numpy python-dateutil
#   streamlit run find_next_buy.py

from __future__ import annotations

import io
import math
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import streamlit as st
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ---------- Page setup ----------
st.set_page_config(
    page_title="ReUseBricks – Find Next Buy",
    page_icon="🧱",
    layout="wide",
)

st.title("Find Next Buy")
st.caption(
    "Rank sets by investment fit (budget, horizon, liquidity) with transparent scoring. "
    "Export to BrickLink Wanted List or a simple buylist CSV."
)

# ---------- Helper utils ----------

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

DATE_COL = "release_date"
ID_COL = "set_num"
NAME_COL = "name"
THEME_COL = "theme"


def _safe_bool01(series: pd.Series) -> pd.Series:
    """Convert a column to 0/1 robustly, accepting True/False, 1/0, 'true'/'false'."""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        # assume already 0/1-ish
        return (series > 0).astype(int)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1, "t": 1, "yes": 1, "y": 1, "1": 1, "false": 0, "f": 0, "no": 0, "n": 0, "0": 0})
        .fillna(0)
        .astype(int)
    )


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def _minmax_0_100(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series(np.full(len(s), 50.0), index=s.index)
    return (s - lo) / (hi - lo) * 100.0


def _pretty_money(x: float) -> str:
    try:
        return f"${x:,.2f}"
    except Exception:
        return "-"


def _today() -> datetime:
    # If you want deterministic tests, hard-code a date here.
    return datetime.utcnow()


@st.cache_data(show_spinner=False)
def load_candidates(upload: io.BytesIO | None) -> pd.DataFrame:
    """Load candidate sets from uploaded CSV or use a tiny built-in sample."""
    if upload:
        df = pd.read_csv(upload)
    else:
        # Minimal inline sample so the page still runs without a CSV.
        df = pd.DataFrame(
            [
                {
                    "set_num": "10281-1",
                    "name": "Bonsai Tree",
                    "theme": "Icons",
                    "retail_price": 50,
                    "current_new_price": 60,
                    "current_used_price": 45,
                    "part_out_value": 80,
                    "units_sold_30d": 300,
                    "active_sellers": 350,
                    "release_date": "2021-01-01",
                    "licensed": 0,
                    "exclusive_minifigs": 0,
                    "box_volume_l": 5,
                    "avg_discount": 0.10,
                    "rerelease_risk": 0.05,
                    "theme_avg_growth": 0.11,
                    "trend_90d": 0.02,
                },
                {
                    "set_num": "21319-1",
                    "name": "Central Perk",
                    "theme": "Ideas",
                    "retail_price": 60,
                    "current_new_price": 85,
                    "current_used_price": 70,
                    "part_out_value": 110,
                    "units_sold_30d": 250,
                    "active_sellers": 300,
                    "release_date": "2019-09-01",
                    "licensed": 1,
                    "exclusive_minifigs": 7,
                    "box_volume_l": 8,
                    "avg_discount": 0.15,
                    "rerelease_risk": 0.20,
                    "theme_avg_growth": 0.08,
                    "trend_90d": 0.03,
                },
                {
                    "set_num": "71741-1",
                    "name": "NINJAGO City Gardens",
                    "theme": "NINJAGO",
                    "retail_price": 300,
                    "current_new_price": 380,
                    "current_used_price": 320,
                    "part_out_value": 520,
                    "units_sold_30d": 80,
                    "active_sellers": 120,
                    "release_date": "2021-02-01",
                    "licensed": 0,
                    "exclusive_minifigs": 20,
                    "box_volume_l": 45,
                    "avg_discount": 0.05,
                    "rerelease_risk": 0.15,
                    "theme_avg_growth": 0.10,
                    "trend_90d": 0.05,
                },
                {
                    "set_num": "10265-1",
                    "name": "Ford Mustang",
                    "theme": "Creator Expert",
                    "retail_price": 150,
                    "current_new_price": 220,
                    "current_used_price": 180,
                    "part_out_value": 280,
                    "units_sold_30d": 120,
                    "active_sellers": 190,
                    "release_date": "2019-03-01",
                    "licensed": 1,
                    "exclusive_minifigs": 0,
                    "box_volume_l": 23,
                    "avg_discount": 0.09,
                    "rerelease_risk": 0.25,
                    "theme_avg_growth": 0.09,
                    "trend_90d": 0.04,
                },
                {
                    "set_num": "21058-1",
                    "name": "The Great Pyramid of Giza",
                    "theme": "Architecture",
                    "retail_price": 130,
                    "current_new_price": 150,
                    "current_used_price": 125,
                    "part_out_value": 200,
                    "units_sold_30d": 140,
                    "active_sellers": 180,
                    "release_date": "2022-06-01",
                    "licensed": 0,
                    "exclusive_minifigs": 0,
                    "box_volume_l": 16,
                    "avg_discount": 0.09,
                    "rerelease_risk": 0.10,
                    "theme_avg_growth": 0.06,
                    "trend_90d": 0.02,
                },
                {
                    "set_num": "42115-1",
                    "name": "Lamborghini Sián FKP 37",
                    "theme": "Technic",
                    "retail_price": 380,
                    "current_new_price": 450,
                    "current_used_price": 400,
                    "part_out_value": 600,
                    "units_sold_30d": 70,
                    "active_sellers": 110,
                    "release_date": "2020-05-28",
                    "licensed": 1,
                    "exclusive_minifigs": 0,
                    "box_volume_l": 55,
                    "avg_discount": 0.08,
                    "rerelease_risk": 0.20,
                    "theme_avg_growth": 0.05,
                    "trend_90d": 0.02,
                },
            ]
        )

    # Tidy up types
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    if "licensed" in df.columns:
        df["licensed"] = _safe_bool01(df["licensed"])
    else:
        df["licensed"] = 0

    # Ensure numeric columns are numeric
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill numeric NAs with column medians
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    # Guarantee required columns exist
    for col in [ID_COL, NAME_COL, THEME_COL, DATE_COL]:
        if col not in df.columns:
            df[col] = ""

    return df


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Months since release
    today = _today()
    if DATE_COL in df.columns:
        df["release_months"] = df[DATE_COL].map(
            lambda d: max(0, (today.year - d.year) * 12 + (today.month - d.month)) if pd.notna(d) else np.nan
        )
    else:
        df["release_months"] = np.nan

    # Margins / ratios
    df["sealed_margin_pct"] = (df["current_new_price"] - df["retail_price"]) / df["retail_price"]
    df["part_out_ratio"] = df["part_out_value"] / df["retail_price"]

    # Target buy price from desired discount (filled later after we read the slider)
    return df


def compute_subscores(df: pd.DataFrame, preference_sealed_weight: float) -> pd.DataFrame:
    """Compute Growth, Liquidity, Margin, Risk, Operational (each ~z-scaled)."""
    df = df.copy()

    # Growth: theme momentum, recent trend, age (proxy for EOL proximity)
    growth = (
        0.5 * _zscore(df["theme_avg_growth"])
        + 0.3 * _zscore(df["trend_90d"])
        + 0.2 * _zscore(df["release_months"])
    )

    # Liquidity: velocity and breadth of sellers
    liquidity = 0.6 * _zscore(df["units_sold_30d"]) + 0.4 * _zscore(df["active_sellers"])

    # Margin: blend based on path preference
    # preference_sealed_weight in [0,1]; 0 = Part-out, 1 = Sealed
    margin = (1 - preference_sealed_weight) * _zscore(df["part_out_ratio"]) + preference_sealed_weight * _zscore(
        df["sealed_margin_pct"]
    )

    # Risk: re-release risk, license volatility, overpay risk (low discount availability)
    risk = (
        0.5 * _zscore(df["rerelease_risk"])
        + 0.3 * _zscore(df["licensed"])
        + 0.2 * _zscore(1.0 - df["avg_discount"])
    )

    # Operational: storage burden (bigger box → higher burden)
    operational = 0.7 * _zscore(df["box_volume_l"]) + 0.3 * 0  # handling complexity placeholder

    out = df.copy()
    out["growth_sub"] = growth
    out["liquidity_sub"] = liquidity
    out["margin_sub"] = margin
    out["risk_sub"] = risk
    out["operational_sub"] = operational
    return out


def compute_scores(
    df: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.DataFrame:
    """
    Combine subscores with weights:
      raw = + w_g*growth + w_l*liquidity + w_m*margin - w_r*risk - w_o*operational
      score = min-max 0..100 within the current filtered pool
    """
    df = df.copy()
    df["raw_score"] = (
        weights["growth"] * df["growth_sub"]
        + weights["liquidity"] * df["liquidity_sub"]
        + weights["margin"] * df["margin_sub"]
        - weights["risk"] * df["risk_sub"]
        - weights["operational"] * df["operational_sub"]
    )
    df["score"] = _minmax_0_100(df["raw_score"])
    return df


def suggest_basket(df: pd.DataFrame, budget_total: float) -> Tuple[pd.DataFrame, float]:
    """
    Greedy pick by score until budget exhausted.
    Returns (basket_df, spend_total).
    """
    items = []
    spend = 0.0
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


def build_wanted_list_xml(rows: pd.DataFrame, condition: str = "N", qty: int = 1, list_name: str | None = None) -> bytes:
    """
    BrickLink Wanted List XML
      ITEMTYPE: S (set)
      ITEMID: use the BrickLink catalog number (e.g., 10265-1). Ensure your dataset's set_num matches BL.
      CONDITION: 'N' or 'U'
      MAXPRICE: target buy price
    """
    inv = ET.Element("INVENTORY")
    if list_name:
        name_el = ET.SubElement(inv, "WANTEDLIST")
        ET.SubElement(name_el, "NAME").text = str(list_name)

    for _, r in rows.iterrows():
        item = ET.SubElement(inv, "ITEM")
        ET.SubElement(item, "ITEMTYPE").text = "S"
        ET.SubElement(item, "ITEMID").text = str(r[ID_COL])
        ET.SubElement(item, "CONDITION").text = condition
        ET.SubElement(item, "MINQTY").text = str(qty)
        ET.SubElement(item, "MAXPRICE").text = f"{float(r['target_buy_price']):.2f}"

    xml_bytes = ET.tostring(inv, encoding="utf-8")
    # Pretty print
    pretty = minidom.parseString(xml_bytes).toprettyxml(indent="  ", encoding="utf-8")
    return pretty


def rows_to_buylist_csv(rows: pd.DataFrame) -> bytes:
    cols = [
        "set_num",
        "name",
        "theme",
        "target_buy_price",
        "retail_price",
        "current_new_price",
        "part_out_value",
        "sealed_roi_pct",
        "part_out_roi_pct",
        "score",
    ]
    export = rows.copy()
    if "sealed_roi_pct" not in export.columns:
        export["sealed_roi_pct"] = (export["current_new_price"] - export["target_buy_price"]) / export["target_buy_price"]
    if "part_out_roi_pct" not in export.columns:
        export["part_out_roi_pct"] = (0.7 * export["part_out_value"] - export["target_buy_price"]) / export["target_buy_price"]

    export = export[cols]
    return export.to_csv(index=False).encode("utf-8")


def explain_row(r: pd.Series, medians: Dict[str, float]) -> List[str]:
    """Generate simple natural-language reasons for why a set scored as it did."""
    msgs = []

    # Growth signals
    if r["theme_avg_growth"] >= medians["theme_avg_growth"]:
        msgs.append("Theme momentum above peer median")
    if r["trend_90d"] >= medians["trend_90d"]:
        msgs.append("Positive 90‑day price trend")
    if r["release_months"] >= medians["release_months"]:
        msgs.append("Older release (closer to/after EOL)")

    # Liquidity
    if r["units_sold_30d"] >= medians["units_sold_30d"]:
        msgs.append("Moves quickly (high 30‑day sold velocity)")
    if r["active_sellers"] >= medians["active_sellers"]:
        msgs.append("Widely available (depth of sellers)")

    # Margin
    if r["sealed_margin_pct"] >= medians["sealed_margin_pct"]:
        msgs.append("Healthy sealed margin vs. retail")
    if r["part_out_ratio"] >= medians["part_out_ratio"]:
        msgs.append("Strong part‑out value/retail ratio")

    # Risk
    if r["rerelease_risk"] <= medians["rerelease_risk"]:
        msgs.append("Lower re‑release risk")
    if r["licensed"] > 0:
        msgs.append("Licensed IP (can be volatile)")

    # Operational
    if r["box_volume_l"] <= medians["box_volume_l"]:
        msgs.append("Compact to store/ship")

    return msgs


# ---------- Sidebar controls ----------

with st.sidebar:
    st.subheader("Your constraints")

    colA, colB = st.columns(2)
    with colA:
        budget_total = st.number_input("Total budget (USD)", min_value=0.0, value=500.0, step=50.0, help="Basket budget cap")
    with colB:
        per_set_cap = st.checkbox("Only show sets ≤ target buy price cap", value=True)

    horizon_months = st.number_input("Holding horizon (months)", min_value=1, value=12, step=1)

    desired_discount_pct = st.slider(
        "Desired discount off retail for buys",
        min_value=0,
        max_value=50,
        value=20,
        step=1,
        help="Target buy = retail × (1 − this %)",
    )

    pref = st.radio("Path preference", options=["Neutral", "Sealed", "Part‑out"], index=0, horizontal=True)
    pref_weight_sealed = {"Neutral": 0.5, "Sealed": 1.0, "Part‑out": 0.0}[pref]

    liquidity_floor = st.slider(
        "Min 30‑day units sold",
        min_value=0,
        max_value=400,
        value=50,
        step=10,
    )

    exclude_licensed = st.checkbox("Exclude licensed sets (Star Wars, HP, etc.)", value=False)

    st.markdown("---")
    st.subheader("Weights")
    w_growth = st.slider("Growth weight", 0.0, 1.0, 0.30, 0.01)
    w_liq = st.slider("Liquidity weight", 0.0, 1.0, 0.25, 0.01)
    w_margin = st.slider("Margin weight", 0.0, 1.0, 0.20, 0.01)
    w_risk = st.slider("Risk penalty weight", 0.0, 1.0, 0.15, 0.01)
    w_ops = st.slider("Operational penalty weight", 0.0, 1.0, 0.10, 0.01)

    weights = dict(growth=w_growth, liquidity=w_liq, margin=w_margin, risk=w_risk, operational=w_ops)

    st.markdown("---")
    st.subheader("Load data")
    file = st.file_uploader("Upload candidate sets CSV", type=["csv"])
    st.caption(
        "CSV schema shown at the bottom of this page. If you don't upload one, a tiny built‑in sample is used."
    )

# ---------- Data pipeline ----------

df = load_candidates(file)
df = enrich_features(df)

# Target buy price from desired discount
df["target_buy_price"] = df["retail_price"] * (1 - desired_discount_pct / 100.0)

# Filters
mask = pd.Series(True, index=df.index)
if exclude_licensed and "licensed" in df.columns:
    mask &= df["licensed"] == 0
mask &= df["units_sold_30d"] >= liquidity_floor
if per_set_cap:
    mask &= df["target_buy_price"] <= budget_total

filtered = df.loc[mask].copy()

if filtered.empty:
    st.warning("No sets match your current filters. Try lowering the liquidity floor or increasing your budget.")
    st.stop()

# Subscores and composite
scored = compute_subscores(filtered, preference_sealed_weight=pref_weight_sealed)
scored = compute_scores(scored, weights=weights)

# ROI proxies for display
scored["sealed_roi_pct"] = (scored["current_new_price"] - scored["target_buy_price"]) / scored["target_buy_price"]
scored["part_out_roi_pct"] = (0.7 * scored["part_out_value"] - scored["target_buy_price"]) / scored["target_buy_price"]

# Display: Top picks table
st.subheader("Top picks")
display_cols = [
    "score",
    "set_num",
    "name",
    "theme",
    "target_buy_price",
    "retail_price",
    "current_new_price",
    "sealed_roi_pct",
    "part_out_value",
    "part_out_roi_pct",
    "units_sold_30d",
    "active_sellers",
    "release_date",
]
pretty = scored[display_cols].sort_values("score", ascending=False).copy()
pretty["score"] = pretty["score"].round(1)
pretty["target_buy_price"] = pretty["target_buy_price"].map(_pretty_money)
pretty["retail_price"] = pretty["retail_price"].map(_pretty_money)
pretty["current_new_price"] = pretty["current_new_price"].map(_pretty_money)
pretty["part_out_value"] = pretty["part_out_value"].map(_pretty_money)
pretty["sealed_roi_pct"] = (pretty["sealed_roi_pct"] * 100.0).map(lambda x: f"{x:.1f}%")
pretty["part_out_roi_pct"] = (pretty["part_out_roi_pct"] * 100.0).map(lambda x: f"{x:.1f}%")

st.dataframe(pretty.reset_index(drop=True), use_container_width=True, height=420)

# Explainability: reasons for top 3
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
    "box_volume_l": scored["box_volume_l"].median(),
}

for _, row in scored.sort_values("score", ascending=False).head(3).iterrows():
    with st.expander(f"#{row['set_num']} — {row['name']}  (Score {row['score']:.1f})"):
        bullets = explain_row(row, meds)
        if bullets:
            st.write("\n".join([f"• {b}" for b in bullets]))
        else:
            st.write("• Balanced profile across growth, liquidity, and margin.")

# Basket suggestion under budget
st.subheader("Suggested basket under your budget")

basket, spend = suggest_basket(scored, budget_total=budget_total)
if basket.empty:
    st.info("No combination of sets fits under your total budget. Increase the budget or desired discount.")
else:
    # Decide ROI path based on preference
    use_sealed = pref_weight_sealed >= 0.5
    if use_sealed:
        basket["est_profit"] = basket["current_new_price"] - basket["target_buy_price"]
    else:
        basket["est_profit"] = 0.7 * basket["part_out_value"] - basket["target_buy_price"]

    total_profit = float(basket["est_profit"].sum())
    est_roi_pct = total_profit / spend if spend > 0 else 0.0

    basket_view = basket[
        ["score", "set_num", "name", "theme", "target_buy_price", "est_profit"]
    ].sort_values("score", ascending=False)
    basket_view["target_buy_price"] = basket_view["target_buy_price"].map(_pretty_money)
    basket_view["est_profit"] = basket_view["est_profit"].map(_pretty_money)

    st.dataframe(basket_view.reset_index(drop=True), use_container_width=True, height=260)

    col1, col2, col3 = st.columns(3)
    col1.metric("Planned spend", _pretty_money(spend))
    col2.metric("Estimated profit", _pretty_money(total_profit))
    col3.metric("Estimated ROI (basket)", f"{est_roi_pct * 100:.1f}%")

    # Downloads
    st.markdown("#### Export")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        wanted_name = st.text_input("Wanted list name (optional)", value="ReUseBricks Buylist")
    with c2:
        cond = st.radio("Condition", options=["New (N)", "Used (U)"], index=0, horizontal=True)
        cond_code = "N" if cond.startswith("New") else "U"
    with c3:
        qty = st.number_input("Qty per set", min_value=1, value=1, step=1)

    xml_bytes = build_wanted_list_xml(basket, condition=cond_code, qty=qty, list_name=wanted_name)
    csv_bytes = rows_to_buylist_csv(basket)

    colA, colB = st.columns(2)
    with colA:
        st.download_button(
            "BrickLink Wanted List (XML)",
            data=xml_bytes,
            file_name="wanted_list.xml",
            mime="application/xml",
        )
    with colB:
        st.download_button(
            "Buylist CSV",
            data=csv_bytes,
            file_name="buylist.csv",
            mime="text/csv",
        )

st.markdown("---")
with st.expander("CSV schema (columns)"):
    st.code(
        """set_num,name,theme,retail_price,current_new_price,current_used_price,part_out_value,
units_sold_30d,active_sellers,release_date(YYYY-MM-DD),licensed(0/1),
exclusive_minifigs,box_volume_l,avg_discount,rerelease_risk(0..1),theme_avg_growth,trend_90d
""",
        language="text",
    )

st.caption(
    "Scoring math: score = minmax( w_g*growth + w_l*liquidity + w_m*margin − w_r*risk − w_o*operational ). "
    "This is NOT financial advice; use your judgment and your own data sources."
)


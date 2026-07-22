from __future__ import annotations

from pathlib import Path
from typing import Iterable

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# App configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Samlet portefølje",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILE = Path(__file__).with_name("AI_portfolio.xlsx")
SHEET_NAME = "Total"
VALID_ASSET_TYPES = {"Stock", "ETF"}


# ---------------------------------------------------------
# Styling and formatting helpers
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        [data-testid="stMetric"] {
            background: rgba(127, 127, 127, 0.08);
            border: 1px solid rgba(127, 127, 127, 0.18);
            padding: 0.8rem 1rem;
            border-radius: 0.8rem;
        }
        .small-note {color: #777; font-size: 0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def fmt_dkk(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "–"
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".") + " kr."


def fmt_pct(value: float, decimals: int = 1) -> str:
    if pd.isna(value):
        return "–"
    return f"{value * 100:.{decimals}f}%".replace(".", ",")


def clean_text(series: pd.Series, fallback: str = "Ikke angivet") -> pd.Series:
    result = series.astype("string").str.strip()
    return result.mask(result.isna() | result.eq(""), fallback)


def first_existing(columns: Iterable[str], available: Iterable[str]) -> str | None:
    available_set = set(available)
    return next((column for column in columns if column in available_set), None)


# ---------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_portfolio(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Filen blev ikke fundet: {path.name}")

    raw = pd.read_excel(path, sheet_name=SHEET_NAME)
    raw.columns = [str(column).strip() for column in raw.columns]
    raw = raw.dropna(how="all").copy()

    required = ["Asset_type", "Name", "Nuværende eksponering"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError("Manglende obligatoriske kolonner: " + ", ".join(missing))

    for column in [
        "Quantity",
        "Purchase_price",
        "Current_price",
        "Nuværende eksponering",
        "Gevinst",
    ]:
        if column in raw.columns:
            raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw["Name"] = clean_text(raw["Name"])
    raw["Asset_type"] = clean_text(raw["Asset_type"], fallback="Cash")

    cash_mask = raw["Name"].str.casefold().eq("kontant")
    raw.loc[cash_mask, "Asset_type"] = "Cash"

    for column in ["Ticker", "Sector", "Account"]:
        if column not in raw.columns:
            raw[column] = "Ikke angivet"
        raw[column] = clean_text(raw[column])

    raw["Market_Value_DKK"] = raw["Nuværende eksponering"].fillna(0.0)
    raw["Return_DKK"] = raw.get("Gevinst", pd.Series(index=raw.index, dtype=float)).fillna(0.0)
    raw["Cost_Value_DKK"] = raw["Market_Value_DKK"] - raw["Return_DKK"]
    raw["Return_Pct"] = np.where(
        raw["Cost_Value_DKK"].abs() > 1e-9,
        raw["Return_DKK"] / raw["Cost_Value_DKK"],
        np.nan,
    )

    total_value = raw["Market_Value_DKK"].sum()
    raw["Portfolio_Weight"] = np.where(
        total_value > 0,
        raw["Market_Value_DKK"] / total_value,
        np.nan,
    )

    securities = raw[raw["Asset_type"].isin(VALID_ASSET_TYPES)].copy()
    securities = securities.sort_values("Market_Value_DKK", ascending=False)

    quality_issues: list[str] = []
    if securities.empty:
        quality_issues.append("Der blev ikke fundet aktier eller ETF'er.")
    if securities["Ticker"].eq("Ikke angivet").any():
        quality_issues.append("En eller flere positioner mangler ticker.")
    if securities["Sector"].eq("Ikke angivet").any():
        quality_issues.append("En eller flere positioner mangler sektor.")
    if securities["Account"].eq("Ikke angivet").any():
        quality_issues.append("En eller flere positioner mangler depot/konto.")
    if securities["Market_Value_DKK"].le(0).any():
        quality_issues.append("En eller flere værdipapirpositioner har nul eller negativ markedsværdi.")
    if securities["Cost_Value_DKK"].le(0).any():
        quality_issues.append("En eller flere positioner har nul eller negativ beregnet kostpris.")
    if securities["Ticker"].duplicated().any():
        duplicates = ", ".join(sorted(securities.loc[securities["Ticker"].duplicated(False), "Ticker"].unique()))
        quality_issues.append(f"Dublerede tickere: {duplicates}.")

    return raw, securities, quality_issues


try:
    portfolio_all, securities_all, quality_issues = load_portfolio(DATA_FILE)
except Exception as exc:
    st.error(f"Dashboardet kunne ikke indlæse data: {exc}")
    st.info("Placér AI_portfolio.xlsx i samme mappe som app.py og kontrollér, at fanen hedder 'Total'.")
    st.stop()


# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.header("Filtre")

asset_options = ["Alle", "Stock", "ETF"]
asset_filter = st.sidebar.radio("Aktivtype", asset_options, horizontal=False)

account_options = sorted(securities_all["Account"].dropna().unique().tolist())
selected_accounts = st.sidebar.multiselect(
    "Depot/konto",
    account_options,
    default=account_options,
)

sector_options = sorted(securities_all["Sector"].dropna().unique().tolist())
selected_sectors = st.sidebar.multiselect(
    "Sektor",
    sector_options,
    default=sector_options,
)

filtered = securities_all.copy()
if asset_filter != "Alle":
    filtered = filtered[filtered["Asset_type"].eq(asset_filter)]
filtered = filtered[
    filtered["Account"].isin(selected_accounts)
    & filtered["Sector"].isin(selected_sectors)
]

st.sidebar.divider()
st.sidebar.caption(f"Datakilde: {DATA_FILE.name} · fane: {SHEET_NAME}")
if st.sidebar.button("Genindlæs data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------
# Header and portfolio KPIs
# ---------------------------------------------------------
st.title("📊 Samlet porteføljedashboard")
st.caption(
    "Første version baseret på AI_portfolio.xlsx. "
    "Markedsværdi og gevinst læses fra Excel; kostpris, afkast og porteføljevægt beregnes i appen."
)

cash_value = portfolio_all.loc[portfolio_all["Asset_type"].eq("Cash"), "Market_Value_DKK"].sum()
security_value = securities_all["Market_Value_DKK"].sum()
total_value = portfolio_all["Market_Value_DKK"].sum()
total_gain = securities_all["Return_DKK"].sum()
total_cost = securities_all["Cost_Value_DKK"].sum()
total_return_pct = total_gain / total_cost if total_cost else np.nan

kpi_cols = st.columns(6)
kpi_cols[0].metric("Samlet portefølje", fmt_dkk(total_value))
kpi_cols[1].metric("Værdipapirer", fmt_dkk(security_value))
kpi_cols[2].metric("Kontant", fmt_dkk(cash_value), fmt_pct(cash_value / total_value) if total_value else "–")
kpi_cols[3].metric("Samlet kostpris", fmt_dkk(total_cost))
kpi_cols[4].metric("Gevinst/tab", fmt_dkk(total_gain), fmt_pct(total_return_pct))
kpi_cols[5].metric("Positioner", f"{len(securities_all)}")

stock_value = securities_all.loc[securities_all["Asset_type"].eq("Stock"), "Market_Value_DKK"].sum()
etf_value = securities_all.loc[securities_all["Asset_type"].eq("ETF"), "Market_Value_DKK"].sum()

st.markdown(
    f"<div class='small-note'>Aktier: <b>{fmt_dkk(stock_value)}</b> · "
    f"ETF'er: <b>{fmt_dkk(etf_value)}</b> · "
    f"Viste positioner efter filter: <b>{len(filtered)}</b></div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
overview_tab, positions_tab, allocation_tab, quality_tab = st.tabs(
    ["Overblik", "Positioner", "Fordeling", "Datakvalitet"]
)

with overview_tab:
    if filtered.empty:
        st.warning("Ingen positioner matcher de valgte filtre.")
    else:
        left, right = st.columns([1.15, 1])

        with left:
            st.subheader("Største positioner")
            top_positions = filtered.nlargest(12, "Market_Value_DKK").copy()
            chart = (
                alt.Chart(top_positions)
                .mark_bar()
                .encode(
                    x=alt.X("Market_Value_DKK:Q", title="Markedsværdi (DKK)"),
                    y=alt.Y("Name:N", sort="-x", title=None),
                    tooltip=[
                        alt.Tooltip("Name:N", title="Navn"),
                        alt.Tooltip("Ticker:N", title="Ticker"),
                        alt.Tooltip("Asset_type:N", title="Type"),
                        alt.Tooltip("Market_Value_DKK:Q", title="Markedsværdi", format=",.0f"),
                        alt.Tooltip("Portfolio_Weight:Q", title="Porteføljevægt", format=".1%"),
                    ],
                )
                .properties(height=420)
            )
            st.altair_chart(chart, use_container_width=True)

        with right:
            st.subheader("Aktivfordeling")
            asset_breakdown = (
                portfolio_all.groupby("Asset_type", as_index=False)["Market_Value_DKK"]
                .sum()
                .query("Market_Value_DKK > 0")
            )
            donut = (
                alt.Chart(asset_breakdown)
                .mark_arc(innerRadius=70)
                .encode(
                    theta=alt.Theta("Market_Value_DKK:Q"),
                    color=alt.Color("Asset_type:N", title="Aktivtype"),
                    tooltip=[
                        alt.Tooltip("Asset_type:N", title="Type"),
                        alt.Tooltip("Market_Value_DKK:Q", title="Værdi", format=",.0f"),
                    ],
                )
                .properties(height=300)
            )
            st.altair_chart(donut, use_container_width=True)

            st.subheader("Bedste og svageste afkast")
            performance = filtered.sort_values("Return_Pct", ascending=False)
            best = performance.head(3)[["Name", "Return_Pct"]]
            worst = performance.tail(3).sort_values("Return_Pct")[["Name", "Return_Pct"]]
            bcol, wcol = st.columns(2)
            with bcol:
                st.markdown("**Bedste**")
                for row in best.itertuples():
                    st.write(f"{row.Name}: {fmt_pct(row.Return_Pct)}")
            with wcol:
                st.markdown("**Svageste**")
                for row in worst.itertuples():
                    st.write(f"{row.Name}: {fmt_pct(row.Return_Pct)}")

        st.subheader("Koncentrationskontrol")
        concentration = filtered.nlargest(10, "Portfolio_Weight")[[
            "Name", "Asset_type", "Sector", "Market_Value_DKK", "Portfolio_Weight"
        ]].copy()
        concentration["Status"] = np.select(
            [concentration["Portfolio_Weight"] > 0.20, concentration["Portfolio_Weight"] > 0.12],
            ["🔴 Over 20%", "🟠 Over 12%"],
            default="🟢 Under 12%",
        )
        st.dataframe(
            concentration,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Name": "Navn",
                "Asset_type": "Type",
                "Sector": "Sektor",
                "Market_Value_DKK": st.column_config.NumberColumn("Markedsværdi", format="%.0f kr."),
                "Portfolio_Weight": st.column_config.ProgressColumn(
                    "Porteføljevægt", min_value=0.0, max_value=max(0.20, float(concentration["Portfolio_Weight"].max())), format="%.1f%%"
                ),
                "Status": "Status",
            },
        )

with positions_tab:
    st.subheader("Samlet positionstabel")

    display_columns = [
        "Asset_type", "Name", "Ticker", "Quantity", "Purchase_price", "Current_price",
        "Market_Value_DKK", "Cost_Value_DKK", "Return_DKK", "Return_Pct",
        "Portfolio_Weight", "Sector", "Account",
    ]
    display = filtered[[column for column in display_columns if column in filtered.columns]].copy()

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Asset_type": "Type",
            "Name": "Navn",
            "Ticker": "Ticker",
            "Quantity": st.column_config.NumberColumn("Antal", format="%.2f"),
            "Purchase_price": st.column_config.NumberColumn("Købskurs", format="%.2f"),
            "Current_price": st.column_config.NumberColumn("Aktuel kurs", format="%.2f"),
            "Market_Value_DKK": st.column_config.NumberColumn("Markedsværdi", format="%.0f kr."),
            "Cost_Value_DKK": st.column_config.NumberColumn("Kostpris", format="%.0f kr."),
            "Return_DKK": st.column_config.NumberColumn("Gevinst/tab", format="%.0f kr."),
            "Return_Pct": st.column_config.NumberColumn("Afkast", format="%.1f%%"),
            "Portfolio_Weight": st.column_config.NumberColumn("Vægt", format="%.1f%%"),
            "Sector": "Sektor",
            "Account": "Depot",
        },
    )

    csv = display.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    st.download_button(
        "Download viste positioner som CSV",
        data=csv,
        file_name="portfolio_positioner.csv",
        mime="text/csv",
    )

with allocation_tab:
    st.subheader("Fordeling af hele porteføljen")

    sector = securities_all.groupby("Sector", as_index=False)["Market_Value_DKK"].sum()
    sector["Weight"] = sector["Market_Value_DKK"] / total_value if total_value else np.nan

    account = portfolio_all.groupby("Account", as_index=False)["Market_Value_DKK"].sum()
    account = account[account["Market_Value_DKK"] > 0]
    account["Weight"] = account["Market_Value_DKK"] / total_value if total_value else np.nan

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sektorfordeling**")
        sector_chart = (
            alt.Chart(sector.sort_values("Market_Value_DKK", ascending=False))
            .mark_bar()
            .encode(
                x=alt.X("Market_Value_DKK:Q", title="Markedsværdi (DKK)"),
                y=alt.Y("Sector:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("Sector:N", title="Sektor"),
                    alt.Tooltip("Market_Value_DKK:Q", title="Værdi", format=",.0f"),
                    alt.Tooltip("Weight:Q", title="Vægt", format=".1%"),
                ],
            )
            .properties(height=430)
        )
        st.altair_chart(sector_chart, use_container_width=True)

    with c2:
        st.markdown("**Depotfordeling**")
        account_chart = (
            alt.Chart(account)
            .mark_arc(innerRadius=65)
            .encode(
                theta="Market_Value_DKK:Q",
                color=alt.Color("Account:N", title="Depot"),
                tooltip=[
                    alt.Tooltip("Account:N", title="Depot"),
                    alt.Tooltip("Market_Value_DKK:Q", title="Værdi", format=",.0f"),
                    alt.Tooltip("Weight:Q", title="Vægt", format=".1%"),
                ],
            )
            .properties(height=350)
        )
        st.altair_chart(account_chart, use_container_width=True)

        st.markdown("**Aktier kontra ETF'er**")
        asset_summary = securities_all.groupby("Asset_type", as_index=False).agg(
            Market_Value_DKK=("Market_Value_DKK", "sum"),
            Return_DKK=("Return_DKK", "sum"),
            Positions=("Ticker", "count"),
        )
        asset_summary["Weight"] = asset_summary["Market_Value_DKK"] / total_value if total_value else np.nan
        st.dataframe(
            asset_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Asset_type": "Type",
                "Market_Value_DKK": st.column_config.NumberColumn("Markedsværdi", format="%.0f kr."),
                "Return_DKK": st.column_config.NumberColumn("Gevinst/tab", format="%.0f kr."),
                "Positions": "Positioner",
                "Weight": st.column_config.NumberColumn("Vægt", format="%.1f%%"),
            },
        )

with quality_tab:
    st.subheader("Datakvalitet og beregningsgrundlag")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Rækker i Excel", len(portfolio_all))
    q2.metric("Værdipapirer", len(securities_all))
    q3.metric("Manglende tickere", int(securities_all["Ticker"].eq("Ikke angivet").sum()))
    q4.metric("Dublerede tickere", int(securities_all["Ticker"].duplicated().sum()))

    if quality_issues:
        for issue in quality_issues:
            st.warning(issue)
    else:
        st.success("Ingen kritiske datakvalitetsfejl fundet i de anvendte felter.")

    st.markdown("### Beregninger i denne version")
    st.markdown(
        """
        - **Markedsværdi:** læses fra `Nuværende eksponering`.
        - **Gevinst/tab:** læses fra `Gevinst`.
        - **Kostpris:** markedsværdi minus gevinst/tab.
        - **Afkast:** gevinst/tab divideret med beregnet kostpris.
        - **Porteføljevægt:** markedsværdi divideret med samlet porteføljeværdi inklusive kontant.
        """
    )

    st.info(
        "Kolonnerne Market_value_DKK, Cost_Value_DKK, Return_DKK, Return_Pct og Portfolio_Weight "
        "er tomme i den uploadede Excel-fil. Derfor beregner appen dem uden at ændre Excel-filen."
    )

    with st.expander("Vis rå data fra Excel"):
        st.dataframe(portfolio_all, use_container_width=True, hide_index=True)

st.divider()
st.caption("Version 1.0 · Samlet porteføljeoverblik · Datakilde: AI_portfolio.xlsx")

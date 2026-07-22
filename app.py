from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


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

# Fallback only used if Yahoo Finance cannot deliver an FX quote.
FALLBACK_FX_DKK = {
    "DKK": 1.0,
    "EUR": 7.46,
    "USD": 6.40,
    "SEK": 0.67,
    "GBP": 8.60,
}


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


def fmt_number(value: float | int) -> str:
    """Danish thousands separator, no decimals."""
    if pd.isna(value):
        return "–"
    return f"{float(value):,.0f}".replace(",", ".")


def fmt_dkk(value: float) -> str:
    if pd.isna(value):
        return "–"
    return f"{fmt_number(value)} kr."


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "–"
    return f"{value * 100:.0f}%"


def clean_text(series: pd.Series, fallback: str = "Ikke angivet") -> pd.Series:
    result = series.astype("string").str.strip()
    return result.mask(result.isna() | result.eq(""), fallback)


def infer_currency(ticker: str) -> str:
    """Infer trading currency from the exchange suffix in the supplied ticker."""
    ticker = str(ticker).strip().upper()

    if "XSTO" in ticker:
        return "SEK"
    if "XCSE" in ticker or ticker.endswith(".CO"):
        return "DKK"
    if "XAMS" in ticker or "XETR" in ticker or ticker.endswith(".DE"):
        return "EUR"
    if "XNYS" in ticker or ticker in {"FRO"}:
        return "USD"
    if ticker.endswith(".L"):
        # COPX.L in this portfolio is the USD listing.
        return "USD"
    return "DKK"


@st.cache_data(ttl=3600, show_spinner=False)
def get_fx_rates() -> tuple[dict[str, float], list[str]]:
    """Fetch DKK conversion rates. Returns fallbacks when data cannot be fetched."""
    pairs = {
        "EUR": "EURDKK=X",
        "USD": "USDDKK=X",
        "SEK": "SEKDKK=X",
        "GBP": "GBPDKK=X",
    }
    rates = FALLBACK_FX_DKK.copy()
    warnings: list[str] = []

    for currency, pair in pairs.items():
        try:
            history = yf.Ticker(pair).history(period="5d", auto_adjust=False)
            close = history["Close"].dropna()
            if close.empty:
                raise ValueError("ingen kursdata")
            rates[currency] = float(close.iloc[-1])
        except Exception:
            warnings.append(
                f"Valutakurs for {currency}/DKK kunne ikke hentes. "
                f"Fallback {rates[currency]:.4f} anvendes."
            )

    return rates, warnings


# ---------------------------------------------------------
# Data loading and preparation
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_portfolio(path: Path, fx_rates: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Filen blev ikke fundet: {path.name}")

    # Read the preferred sheet. If it has been renamed and the workbook only
    # contains one sheet, use that sheet automatically.
    excel_file = pd.ExcelFile(path)
    sheet_to_use = SHEET_NAME
    if SHEET_NAME not in excel_file.sheet_names:
        if len(excel_file.sheet_names) == 1:
            sheet_to_use = excel_file.sheet_names[0]
        else:
            raise ValueError(
                f"Fanen '{SHEET_NAME}' blev ikke fundet. "
                f"Tilgængelige faner: {', '.join(excel_file.sheet_names)}"
            )

    raw = pd.read_excel(excel_file, sheet_name=sheet_to_use)
    raw.columns = [str(column).strip() for column in raw.columns]
    raw = raw.dropna(how="all").copy()

    # Accept common variations in capitalization and spelling.
    normalized_headers = {
        str(column).strip().casefold().replace(" ", "_"): column
        for column in raw.columns
    }
    aliases = {
        "asset_type": ["asset_type", "assettype", "type", "aktivtype"],
        "name": ["name", "navn"],
        "ticker": ["ticker", "symbol"],
        "quantity": ["quantity", "antal"],
        "purchase_price": ["purchase_price", "purchaseprice", "købskurs", "koebskurs"],
        "current_price": ["current_price", "currentprice", "aktuel_kurs", "nuværende_kurs"],
        "sector": ["sector", "sektor"],
        "account": ["account", "depot", "konto"],
    }

    rename_map = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            original = normalized_headers.get(candidate.casefold().replace(" ", "_"))
            if original is not None:
                rename_map[original] = {
                    "asset_type": "Asset_type",
                    "name": "Name",
                    "ticker": "Ticker",
                    "quantity": "Quantity",
                    "purchase_price": "Purchase_price",
                    "current_price": "Current_price",
                    "sector": "Sector",
                    "account": "Account",
                }[canonical]
                break

    raw = raw.rename(columns=rename_map)

    # Only the six source fields below are mandatory. Values such as
    # Nuværende eksponering, Gevinst, Market_value_DKK and Cost_Value_DKK
    # are never required because the app calculates them itself.
    required = ["Asset_type", "Name", "Ticker", "Quantity", "Purchase_price", "Current_price"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(
            "Manglende obligatoriske grundkolonner: "
            + ", ".join(missing)
            + ". Beregningskolonner som 'Nuværende eksponering' og 'Gevinst' er ikke nødvendige."
        )

    for column in ["Quantity", "Purchase_price", "Current_price"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw["Name"] = clean_text(raw["Name"])
    raw["Asset_type"] = clean_text(raw["Asset_type"], fallback="Cash")

    # The current workbook stores the cash amount in the Ticker cell on the Kontant row.
    cash_mask = raw["Name"].str.casefold().eq("kontant")
    raw.loc[cash_mask, "Asset_type"] = "Cash"

    for column in ["Sector", "Account"]:
        if column not in raw.columns:
            raw[column] = "Ikke angivet"
        raw[column] = clean_text(raw[column])

    raw["Ticker"] = raw["Ticker"].astype("string").str.strip()
    raw.loc[~cash_mask, "Ticker"] = raw.loc[~cash_mask, "Ticker"].fillna("Ikke angivet")

    raw["Currency"] = "DKK"
    raw.loc[~cash_mask, "Currency"] = raw.loc[~cash_mask, "Ticker"].map(infer_currency)
    raw["FX_to_DKK"] = raw["Currency"].map(fx_rates).fillna(1.0)

    # All portfolio values are calculated from quantity, price and FX rate.
    raw["Market_Value_DKK"] = (
        raw["Quantity"].fillna(0)
        * raw["Current_price"].fillna(0)
        * raw["FX_to_DKK"]
    )
    raw["Cost_Value_DKK"] = (
        raw["Quantity"].fillna(0)
        * raw["Purchase_price"].fillna(0)
        * raw["FX_to_DKK"]
    )

    # Cash amount: first try Ticker, then Quantity, then an existing market-value field.
    if cash_mask.any():
        cash_candidates = pd.to_numeric(raw.loc[cash_mask, "Ticker"], errors="coerce")
        if cash_candidates.isna().all():
            cash_candidates = pd.to_numeric(raw.loc[cash_mask, "Quantity"], errors="coerce")
        if cash_candidates.isna().all() and "Market_value_DKK" in raw.columns:
            cash_candidates = pd.to_numeric(raw.loc[cash_mask, "Market_value_DKK"], errors="coerce")
        raw.loc[cash_mask, "Market_Value_DKK"] = cash_candidates.fillna(0).values
        raw.loc[cash_mask, "Cost_Value_DKK"] = cash_candidates.fillna(0).values
        raw.loc[cash_mask, "Currency"] = "DKK"
        raw.loc[cash_mask, "FX_to_DKK"] = 1.0
        raw.loc[cash_mask, "Ticker"] = "Kontant"
        raw.loc[cash_mask, "Sector"] = "Kontant"
        raw.loc[cash_mask, "Account"] = "Kontant"

    raw["Return_DKK"] = raw["Market_Value_DKK"] - raw["Cost_Value_DKK"]
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
    if securities[["Quantity", "Purchase_price", "Current_price"]].isna().any(axis=None):
        quality_issues.append("En eller flere positioner mangler antal, købskurs eller aktuel kurs.")
    if securities["Market_Value_DKK"].le(0).any():
        quality_issues.append("En eller flere værdipapirpositioner har nul eller negativ markedsværdi.")
    if securities["Cost_Value_DKK"].le(0).any():
        quality_issues.append("En eller flere positioner har nul eller negativ kostpris.")
    if securities["Ticker"].duplicated().any():
        duplicates = ", ".join(
            sorted(securities.loc[securities["Ticker"].duplicated(False), "Ticker"].unique())
        )
        quality_issues.append(f"Dublerede tickere: {duplicates}.")

    return raw, securities, quality_issues


fx_rates, fx_warnings = get_fx_rates()

try:
    portfolio_all, securities_all, quality_issues = load_portfolio(DATA_FILE, fx_rates)
except Exception as exc:
    st.error(f"Dashboardet kunne ikke indlæse data: {exc}")
    st.info(
        "Placér AI_portfolio.xlsx i samme mappe som app.py. "
        "Filen skal som minimum indeholde kolonnerne Asset_type, Name, Ticker, "
        "Quantity, Purchase_price og Current_price. "
        "Kolonner som Nuværende eksponering og Gevinst er ikke nødvendige."
    )
    st.stop()


# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------
st.sidebar.header("Filtre")

st.sidebar.subheader("Filtre")

# ---------- Aktivtype ----------
asset_options = ["Stock", "ETF"]
asset_counts = (
    securities_all["Asset_type"]
    .astype(str)
    .value_counts()
    .to_dict()
)

for option in asset_options:
    key = f"filter_asset_{option}"
    if key not in st.session_state:
        st.session_state[key] = True

st.sidebar.markdown("**Aktivtype**")
asset_col1, asset_col2 = st.sidebar.columns(2)

if asset_col1.button("Vælg alle", key="asset_select_all", use_container_width=True):
    for option in asset_options:
        st.session_state[f"filter_asset_{option}"] = True
    st.rerun()

if asset_col2.button("Ryd", key="asset_clear_all", use_container_width=True):
    for option in asset_options:
        st.session_state[f"filter_asset_{option}"] = False
    st.rerun()

selected_asset_types = []
for option in asset_options:
    count = int(asset_counts.get(option, 0))
    if st.sidebar.checkbox(
        f"{option} ({count})",
        key=f"filter_asset_{option}",
    ):
        selected_asset_types.append(option)

# ---------- Depot/konto ----------
account_options = sorted(
    securities_all["Account"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)
account_counts = (
    securities_all["Account"]
    .dropna()
    .astype(str)
    .value_counts()
    .to_dict()
)

for option in account_options:
    key = f"filter_account_{option}"
    if key not in st.session_state:
        st.session_state[key] = True

st.sidebar.markdown("**Depot/konto**")
account_col1, account_col2 = st.sidebar.columns(2)

if account_col1.button("Vælg alle", key="account_select_all", use_container_width=True):
    for option in account_options:
        st.session_state[f"filter_account_{option}"] = True
    st.rerun()

if account_col2.button("Ryd", key="account_clear_all", use_container_width=True):
    for option in account_options:
        st.session_state[f"filter_account_{option}"] = False
    st.rerun()

selected_accounts = []
for option in account_options:
    count = int(account_counts.get(option, 0))
    if st.sidebar.checkbox(
        f"{option} ({count})",
        key=f"filter_account_{option}",
    ):
        selected_accounts.append(option)

# ---------- Sektor ----------
sector_options = sorted(
    securities_all["Sector"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)
selected_sectors = st.sidebar.multiselect(
    "Sektor",
    sector_options,
    default=sector_options,
    key="sector_filter",
)

# ---------- Anvend filtre ----------
filtered = securities_all.copy()

filtered = filtered[
    filtered["Asset_type"].astype(str).isin(selected_asset_types)
]

filtered = filtered[
    filtered["Account"].astype(str).isin(selected_accounts)
]

filtered = filtered[
    filtered["Sector"].astype(str).isin(selected_sectors)
]

if not selected_asset_types:
    st.sidebar.warning("Ingen aktivtyper er valgt.")

if not selected_accounts:
    st.sidebar.warning("Ingen depoter/konti er valgt.")

if filtered.empty:
    st.warning(
        "Ingen positioner matcher de valgte filtre. "
        "Vælg mindst én aktivtype og mindst ét depot."
    )

st.sidebar.divider()
st.sidebar.caption(f"Datakilde: {DATA_FILE.name} · fane: {SHEET_NAME}")
st.sidebar.caption(
    "FX: " + " · ".join(f"{currency} {rate:.4f}" for currency, rate in fx_rates.items() if currency != "DKK")
)
if st.sidebar.button("Genindlæs data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()


# ---------------------------------------------------------
# Header and portfolio KPIs
# ---------------------------------------------------------
st.title("📊 Samlet porteføljedashboard")
st.caption(
    "Markedsværdi og kostpris beregnes som antal × kurs × valutakurs til DKK. "
    "Afkast og vægt beregnes efter DKK-omregningen."
)

for warning in fx_warnings:
    st.warning(warning)

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
kpi_cols[5].metric("Positioner", fmt_number(len(securities_all)))

stock_value = securities_all.loc[securities_all["Asset_type"].eq("Stock"), "Market_Value_DKK"].sum()
etf_value = securities_all.loc[securities_all["Asset_type"].eq("ETF"), "Market_Value_DKK"].sum()

st.markdown(
    f"<div class='small-note'>Aktier: <b>{fmt_dkk(stock_value)}</b> · "
    f"ETF'er: <b>{fmt_dkk(etf_value)}</b> · "
    f"Viste positioner efter filter: <b>{fmt_number(len(filtered))}</b></div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Formatting tables
# ---------------------------------------------------------
def format_position_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame.copy()
    rename = {
        "Asset_type": "Type",
        "Name": "Navn",
        "Ticker": "Ticker",
        "Quantity": "Antal",
        "Purchase_price": "Købskurs",
        "Current_price": "Aktuel kurs",
        "Currency": "Valuta",
        "FX_to_DKK": "FX til DKK",
        "Market_Value_DKK": "Markedsværdi",
        "Cost_Value_DKK": "Kostpris",
        "Return_DKK": "Gevinst/tab",
        "Return_Pct": "Afkast",
        "Portfolio_Weight": "Vægt",
        "Sector": "Sektor",
        "Account": "Depot",
    }
    table = table.rename(columns=rename)

    for column in ["Antal", "Købskurs", "Aktuel kurs", "FX til DKK", "Markedsværdi", "Kostpris", "Gevinst/tab"]:
        if column in table.columns:
            table[column] = table[column].map(fmt_number)
    for column in ["Afkast", "Vægt"]:
        if column in table.columns:
            table[column] = table[column].map(fmt_pct)
    return table


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
                        alt.Tooltip("Portfolio_Weight:Q", title="Porteføljevægt", format=".0%"),
                    ],
                )
                .properties(height=400)
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
                .properties(height=400)
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
        concentration = concentration.rename(
            columns={
                "Name": "Navn",
                "Asset_type": "Type",
                "Sector": "Sektor",
                "Market_Value_DKK": "Markedsværdi",
                "Portfolio_Weight": "Porteføljevægt",
            }
        )
        concentration["Markedsværdi"] = concentration["Markedsværdi"].map(fmt_number)
        concentration["Porteføljevægt"] = concentration["Porteføljevægt"].map(fmt_pct)
        st.dataframe(concentration, use_container_width=True, hide_index=True)

with positions_tab:
    st.subheader("Samlet positionstabel")

    display_columns = [
        "Asset_type", "Name", "Quantity", "Purchase_price", "Current_price",
        "Currency", "Market_Value_DKK", "Cost_Value_DKK", "Return_DKK", "Return_Pct",
        "Portfolio_Weight", "Sector", "Account",
    ]
    display_numeric = filtered[[column for column in display_columns if column in filtered.columns]].copy()
    display = format_position_table(display_numeric)
    st.dataframe(display, use_container_width=True, hide_index=True)

    csv = display_numeric.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
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
                    alt.Tooltip("Weight:Q", title="Vægt", format=".0%"),
                ],
            )
            .properties(height=800)
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
                    alt.Tooltip("Weight:Q", title="Vægt", format=".0%"),
                ],
            )
            .properties(height=800)
        )
        st.altair_chart(account_chart, use_container_width=True)

        st.markdown("**Aktier kontra ETF'er**")
        asset_summary = securities_all.groupby("Asset_type", as_index=False).agg(
            Market_Value_DKK=("Market_Value_DKK", "sum"),
            Return_DKK=("Return_DKK", "sum"),
            Positions=("Ticker", "count"),
        )
        asset_summary["Weight"] = asset_summary["Market_Value_DKK"] / total_value if total_value else np.nan
        asset_summary = asset_summary.rename(
            columns={
                "Asset_type": "Type",
                "Market_Value_DKK": "Markedsværdi",
                "Return_DKK": "Gevinst/tab",
                "Positions": "Positioner",
                "Weight": "Vægt",
            }
        )
        asset_summary["Markedsværdi"] = asset_summary["Markedsværdi"].map(fmt_number)
        asset_summary["Gevinst/tab"] = asset_summary["Gevinst/tab"].map(fmt_number)
        asset_summary["Positioner"] = asset_summary["Positioner"].map(fmt_number)
        asset_summary["Vægt"] = asset_summary["Vægt"].map(fmt_pct)
        st.dataframe(asset_summary, use_container_width=True, hide_index=True)

with quality_tab:
    st.subheader("Datakvalitet og beregningsgrundlag")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Rækker i Excel", fmt_number(len(portfolio_all)))
    q2.metric("Værdipapirer", fmt_number(len(securities_all)))
    q3.metric("Manglende tickere", fmt_number(int(securities_all["Ticker"].eq("Ikke angivet").sum())))
    q4.metric("Dublerede tickere", fmt_number(int(securities_all["Ticker"].duplicated().sum())))

    if quality_issues:
        for issue in quality_issues:
            st.warning(issue)
    else:
        st.success("Ingen kritiske datakvalitetsfejl fundet i de anvendte felter.")

    st.markdown("### Beregninger i denne version")
    st.markdown(
        """
        - **Valuta:** udledes af tickerens børs/suffiks.
        - **Markedsværdi:** antal × aktuel kurs × valutakurs til DKK.
        - **Kostpris:** antal × købskurs × valutakurs til DKK.
        - **Gevinst/tab:** markedsværdi minus kostpris.
        - **Afkast:** gevinst/tab divideret med kostpris.
        - **Porteføljevægt:** markedsværdi divideret med samlet porteføljeværdi inklusive kontant.
        """
    )

    st.info(
        "Aktuelle priser og købskurser læses fra Excel. Kun valutakurser hentes eksternt. "
        "Hvis en valutakurs ikke kan hentes, anvendes en tydeligt markeret fallbackkurs."
    )

    fx_table = pd.DataFrame(
        [{"Valuta": currency, "DKK pr. enhed": fmt_number(rate)} for currency, rate in fx_rates.items()]
    )
    st.markdown("### Anvendte valutakurser")
    st.dataframe(fx_table, use_container_width=True, hide_index=True)

    with st.expander("Vis beregnede rådata"):
        raw_columns = [
            "Asset_type", "Name", "Ticker", "Quantity", "Purchase_price", "Current_price",
            "Currency", "FX_to_DKK", "Market_Value_DKK", "Cost_Value_DKK", "Return_DKK",
            "Return_Pct", "Portfolio_Weight", "Sector", "Account",
        ]
        raw_display = format_position_table(
            portfolio_all[[column for column in raw_columns if column in portfolio_all.columns]]
        )
        st.dataframe(raw_display, use_container_width=True, hide_index=True)

st.divider()
st.caption("Version 2.0 · Fleksible inkluder/ekskluder-filtre · Datakilde: AI_portfolio.xlsx")

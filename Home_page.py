from pyzotero import zotero
import pandas as pd
import streamlit as st
import numpy as np
import altair as alt
from datetime import date
import datetime
from streamlit_extras.switch_page_button import switch_page
import plotly.express as px
import plotly.graph_objs as go
import re
import matplotlib.pyplot as plt
import nltk
@st.cache_resource
def download_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("omw-1.4", quiet=True)
download_nltk()  # only runs once across all sessions
import pydeck as pdk
from countryinfo import CountryInfo
from streamlit_theme import st_theme
from st_keyup import st_keyup
import json
import uuid
from authors_dict import get_df_authors, name_replacements
from copyright import display_custom_license
from sidebar_content import sidebar_content, set_page_config
from format_entry import format_entry
from events import evens_conferences
from format_entry import _resolve_author

from shared_utils import (
    parse_date_column,
    sort_by_date,
    load_reviews_map,
    render_wordcloud,
    split_and_expand,
    remove_numbers,
    convert_df_to_csv,
    render_metrics,
    render_report_charts,
    display_bibliographies,
    sort_radio,
    render_paginated_list,
    author_to_slug,      # ← add
    slug_to_author,      # ← add
)

BASE_URL = "https://intelligence.streamlit.app"

def journal_to_guid(name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, name))

def guid_to_journal(guid: str, journal_list: list) -> str:
    return next((j for j in journal_list if journal_to_guid(j) == guid), "")
    
# ── Zotero connection ───────────────────────────────────────────────────────
library_id = "2514686"
library_type = "group"
api_key = ""

set_page_config()
pd.set_option("display.max_colwidth", None)
zot = zotero.Zotero(library_id, library_type)

# ── Theme-aware logo ────────────────────────────────────────────────────────
theme = st_theme()
image_path = (
    "images/01_logo/IntelArchive_Digital_Logo_Colour-Negative.svg"
    if theme and theme.get("base") == "dark"
    else "images/01_logo/IntelArchive_Digital_Logo_Colour-Positive.svg"
)
with open(image_path) as f:
    st.image(f.read(), width=200)

st.subheader("Intelligence Studies Database", anchor=False)

cite_today = datetime.date.today().strftime("%d %B %Y")
intro = f"""
Welcome to **IntelArchive**.
The IntelArchive is one of the most comprehensive databases listing sources on intelligence studies and history.

Join our Google Groups to get updates and learn new features about the website and the database.
You can also ask questions or make suggestions. (https://groups.google.com/g/intelarchive)

Resources about the website:

Ozkan, Yusuf A. "'Intelligence Studies Network': A Human-Curated Database for Indexing Resources with Open-Source Tools." arXiv, August 7, 2024. https://doi.org/10.48550/arXiv.2408.03868.

Ozkan, Yusuf A. 'Intelligence Studies Network Dataset'. Zenodo, 15 August 2024. https://doi.org/10.5281/zenodo.13325699.

**Cite this page:** IntelArchive. '*Intelligence Studies Network*', Created 1 June 2020, Accessed {cite_today}. https://intelligence.streamlit.app/.
"""

# ── Collection hierarchy ─────────────────────────────────────────────────────
COLLECTION_HIERARCHY = {
    # Top-level containers (no direct items, show subcollections as radio)
    "01": {
        "label": "Intelligence history",
        "key": None,  # no key — doesn't exist as collection in data
        "children_prefix": "01.",
        "exclude": ["01 Intelligence history"],
    },
    "07": {
        "label": "Intelligence collection",
        "key": None,
        "children_prefix": "07.",
        "exclude": [],
    },
    # All collection keys mapped to their prefix for child detection
}

# Full key → name mapping
COLLECTION_KEY_MAP = {
    "CN9F5URY": "00 Intelligence bibliographies",
    "DS3WDJUS": "01.1 Pre-Napoleonic Wars",
    "8XA7D88D": "01.2 Napoleonic Wars",
    "9DTPTK46": "01.3 1800-1914",
    "BNPYHVD4": "01.4 WW1 (First Wold War)",
    "MP7FJ9UA": "01.5 Inter-war period",
    "SCCGXHMZ": "01.6 WW2 (Second World War)",
    "CZT6L9T7": "01.7 Cold War",
    "DHLN8GE4": "01.7.1 Arab-Israeli Conflict",
    "9I86L884": "01.7.2 Falklands War",
    "6XBG92FJ": "01.7.3 Suez Crisis",
    "V7KUA58M": "01.7.4 The Troubles",
    "BHVIFBRH": "01.7.5 Vietnam War",
    "WHBCJ8GW": "01.8 Post-Cold War",
    "TLFN4NAL": "01.9 Terrorism, insurgency, crime",
    "KGU8VLSW": "01.99 Intelligence archives and methodology",
    "HCN8YFI8": "02 Intelligence studies",
    "D67KFVND": "02.1 Intelligence and strategy",
    "NWAKWPT7": "02.2 Intelligence and culture",
    "H28QZ8XV": "02.3 Intelligence research and education",
    "B4CCZ7Y8": "02.4 Policy and intelligence",
    "2Y7S43YJ": "02.5 Intelligence and media",
    "TDUVX2TF": "02.98 Methodology",
    "7R9UG9WU": "02.99 Miscellaneous",
    "CZJ36V8L": "03 Intelligence analysis",
    "CK5MNYPQ": "04 Intelligence organisations",
    "D7XFV7JL": "05 Intelligence failures",
    "9YPHGMBS": "05.1 Intelligence, warning, and surprise",
    "CGAXYI88": "05.2 Politicization of intelligence",
    "DVEM4H4W": "06 Accountability, oversight, and ethics",
    "ZMVDB8A2": "07.1 HUMINT",
    "T92JK7A5": "07.2 SIGINT",
    "PBHFUE8W": "07.3 IMINT - GEOINT",
    "LXMU5UXP": "07.4 OSINT - SOCMINT",
    "N8VR3BYE": "07.5 Medical Intelligence",
    "TEMXY72R": "07.6 Intelligence Collection (other)",
    "RHJFPRAI": "08 Counterintelligence",
    "B6RJNLTK": "09 Covert action",
    "8XXD789V": "10 Intelligence and cybersphere",
    "AZ3BZ9BR": "14 Global intelligence",
    "EJW4BLAR": "16 Non-State Actors",
    "E5UVWK8S": "98.0 AI and intelligence studies",
    "UVSM9U3L": "98.1 Intelligence and Law",
    "Y959U28A": "98.2 War in Ukraine",
    "AWQSU6V5": "98.3 War in Gaza",
    "AKVWM8BZ": "98.4 Middle East conflict",
    "9YH9YSYQ": "98.5 Intelligence in literature and popular culture",
    "MQMHZUFD": "98.6 Disinformation",
    "28B8SB3Y": "98.7 Surveillance",
    "VHKQZA5S": "98.8 Current affairs",
    "R2V36RN8": "98.9 Private-sector intelligence",
    "9H865NIL": "99 Archival sources and reports",
    "FIXZQSS9": "Academic programs on intelligence",
    "Y4YJ2AWB": "Websites",
}

COLLECTION_KEY_MAP["01_CONTAINER"] = "01 Intelligence history"
COLLECTION_KEY_MAP["07_CONTAINER"] = "07 Intelligence collection"
COLLECTION_KEY_MAP["98_CONTAINER"] = "98 Special collections"

# Reverse map: name → key
COLLECTION_NAME_KEY_MAP = {v: k for k, v in COLLECTION_KEY_MAP.items()}

def get_collection_prefix(collection_name):
    """Extract numeric prefix from collection name e.g. '01.7' from '01.7 Cold War'"""
    import re
    match = re.match(r'^(\d+(?:\.\d+)*)', collection_name)
    return match.group(1) if match else None

def get_children(collection_name, df_duplicated):
    prefix = get_collection_prefix(collection_name)
    if not prefix:
        return []
    all_collections = df_duplicated[["Collection_Name", "Collection_Key"]].drop_duplicates()
    children = []
    for _, row in all_collections.iterrows():
        name = row["Collection_Name"]
        child_prefix = get_collection_prefix(name)
        if not child_prefix:
            continue
        if name == collection_name:
            continue
        if child_prefix.startswith(prefix + "."):
            remainder = child_prefix[len(prefix + "."):]
            # Direct child: no further dots in the remainder
            if "." not in remainder:
                children.append({
                    "name": name,
                    "key": row["Collection_Key"],
                    "clean_name": re.sub(r'^\d+[\.\d]*\s*', '', name).strip(),
                })
    return sorted(children, key=lambda x: x["name"])

def render_collection_profile(collection_key, df_dedup, df_duplicated):
    import numpy as np
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    reviews_map = load_reviews_map()

    collection_name = COLLECTION_KEY_MAP.get(collection_key, "")
    if not collection_name:
        st.warning("Collection not found.")
        return

    clean_name = re.sub(r'^\d+[\.\d]*\s*', '', collection_name).strip()
   
    # ── Header ───────────────────────────────────────────────────────────────
    st.title(clean_name, anchor=False)
    st.divider()

    # ── Special case for Global Intelligence ─────────────────────────────────
    if collection_key == "AZ3BZ9BR":
        st.info("For the full Global Intelligence experience including country-level filtering and maps, visit the [Global Intelligence page](/Global_intelligence).")
        st.page_link("pages/11_Global intelligence.py", label="Go to Global Intelligence →")
        st.divider()

    # ── Container collections ─────────────────────────────────────────────────
    if collection_key.endswith("_CONTAINER"):
        prefix_map = {
            "01_CONTAINER": "01.",
            "07_CONTAINER": "07.",
            "98_CONTAINER": "98.",
        }
        prefix = prefix_map.get(collection_key, "")
        all_cols = df_duplicated[["Collection_Name", "Collection_Key"]].drop_duplicates()
        subcols = all_cols[
            all_cols["Collection_Name"].str.startswith(prefix) &
            ~all_cols["Collection_Name"].str.contains(r'\d+\.\d+\.\d+')
        ].sort_values("Collection_Name")

        st.markdown("### Subcollections")
        for _, row in subcols.iterrows():
            child_clean = re.sub(r'^\d+[\.\d]*\s*', '', row["Collection_Name"]).strip()
            child_link  = f"{BASE_URL}/?collection={row['Collection_Key']}"
            count = len(df_duplicated[df_duplicated["Collection_Key"] == row["Collection_Key"]])
            st.markdown(f"- [{child_clean}]({child_link}) · {count} items")
        return

    # ── Check for children (e.g. 01.7 Cold War) ──────────────────────────────
    children = get_children(collection_name, df_duplicated)

    selected_child_key = collection_key
    display_name = clean_name  # ← default, overridden below if children exist
    selected_child_name = None  # ← default
    if children:
        child_options = {
            re.sub(r'^\d+[\.\d]*\s*', '', c["name"]).strip(): c["key"]
            for c in children
        }
        child_names = ["All (including subcollections)"] + list(child_options.keys())

        url_child = st.query_params.get("subcollection", "")
        default_idx = 0
        if url_child and url_child in child_options.values():
            default_idx = next(
                (i for i, c in enumerate(children) if c["key"] == url_child), 0
            )

        if "col_profile_child" not in st.session_state:
            st.session_state["col_profile_child"] = child_names[default_idx + 1 if default_idx > 0 else 0]

        selected_child_name = st.radio(
            "Select a subcollection",
            child_names,
            horizontal=True,
            key="col_profile_child",
        )
        if selected_child_name == "All (including subcollections)":
            selected_child_key = collection_key
            display_name = clean_name
            if st.query_params.get("subcollection"):
                st.query_params.from_dict({"collection": collection_key})
        else:
            selected_child_key = child_options[selected_child_name]
            display_name = selected_child_name
            if selected_child_key != url_child:
                st.query_params.from_dict({
                    "collection": collection_key,
                    "subcollection": selected_child_key,
                })



    # ── Filter data ───────────────────────────────────────────────────────────
# ── Filter data ───────────────────────────────────────────────────────────
    if children and selected_child_name == "All (including subcollections)":
        all_keys = [collection_key] + [c["key"] for c in children]
        df_col = df_duplicated[df_duplicated["Collection_Key"].isin(all_keys)].copy()
        df_col = df_col.drop_duplicates(subset=["Zotero link"])
    else:
        df_col = df_duplicated[df_duplicated["Collection_Key"] == selected_child_key].copy()
    df_col["Collection_Name"] = df_col["Collection_Name"].apply(remove_numbers)
    df_col["Date published"] = (
        df_col["Date published"]
        .str.strip()
        .apply(lambda x: pd.to_datetime(x, utc=True, errors="coerce"))
    )
    df_col["Date published"] = df_col["Date published"].dt.strftime("%Y-%m-%d")
    df_col["Date published"] = df_col["Date published"].fillna("")
    df_col["No date flag"] = df_col["Date published"].isnull().astype(np.uint8)
    df_col = df_col.sort_values(["No date flag", "Date published"], ascending=[True, True])
    df_col = df_col.sort_values("Date published", ascending=False)
    df_col = df_col.reset_index(drop=True)

    collection_link = df_col["Collection_Link"].iloc[0] if not df_col.empty else ""

    st.markdown(f"#### Collection theme: {display_name}")

    # ── Keyword search ────────────────────────────────────────────────────────
    name = st_keyup(
        "Enter keywords to search in title",
        key="col_profile_search",
        placeholder="Search keyword(s)",
        debounce=500,
    )
    if name:
        df_col = df_col[df_col["Title"].str.lower().str.contains(name.lower(), na=False)]

    # ── Metrics row ───────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 4])
    with col1:
        container_metric = st.container()
    with col2:
        with st.popover("More metrics"):
            container_citation        = st.container()
            container_citation_avg    = st.container()
            container_oa              = st.container()
            container_type            = st.container()
            container_author_no       = st.container()
            container_author_pub      = st.container()
            container_collab          = st.container()
    with col3:
        with st.popover("Filters and more"):
            st.write(f"View the collection in [Zotero]({collection_link})")
            col_a, col_b = st.columns(2)
            with col_a:
                display2 = st.checkbox("Display abstracts", key="col_profile_abstracts")
            with col_b:
                only_citation = st.checkbox("Show cited items only", key="col_profile_cited")
                if only_citation:
                    df_col = df_col[
                        (df_col["Citation"].notna()) & (df_col["Citation"] != 0)
                    ]
            view = st.radio(
                "View as:", ("Basic list", "Table", "Bibliography"),
                horizontal=True, key="col_profile_view",
            )
            types = st.multiselect(
                "Publication type",
                df_col["Publication type"].unique(),
                df_col["Publication type"].unique(),
                key="col_profile_types",
            )
            df_col = df_col[df_col["Publication type"].isin(types)].reset_index(drop=True)

            csv = convert_df_to_csv(
                df_col[["Publication type", "Title", "FirstName2", "Abstract",
                         "Date published", "Publisher", "Journal",
                         "Link to publication", "Zotero link"]]
                .assign(Abstract=lambda d: d["Abstract"].str.replace("\n", " "))
                .reset_index(drop=True)
            )
            st.download_button(
                "⬇ Download collection", csv,
                f"{display_name}_{datetime.date.today().isoformat()}.csv",
                mime="text/csv", key="dl-col-profile", icon=":material/download:",
            )

    # ── Compute metrics ───────────────────────────────────────────────────────
    num_items = len(df_col)
    publications_by_type = df_col["Publication type"].value_counts()
    breakdown_string = ", ".join([f"{k}: {v}" for k, v in publications_by_type.items()])
    item_type_no = df_col["Publication type"].nunique()
    citation_count = df_col["Citation"].sum()

    if num_items == 0:
        author_no, author_pub_ratio, collaboration_ratio = 0, 0.0, 0
    else:
        expanded_authors = df_col["FirstName2"].apply(
            lambda x: pd.Series([a.strip() for a in x.split(",")]) if isinstance(x, str) else pd.Series([x])
        ).stack().reset_index(level=1, drop=True)
        author_no = len(expanded_authors)
        author_pub_ratio = round(author_no / num_items, 2)
        df_col["multiple_authors"] = df_col["FirstName2"].astype(str).apply(lambda x: "," in x)
        collaboration_ratio = round(df_col["multiple_authors"].sum() / num_items * 100, 1)

    ja = df_col[df_col["Publication type"] == "Journal article"]
    oa_ratio = (ja["OA status"].sum() / len(ja) * 100) if len(ja) else 0.0

    outlier_detector = (df_col["Citation"] > 1000).any()
    if outlier_detector:
        outlier_count = int((df_col["Citation"] > 1000).sum())
        citation_avg  = round(df_col[df_col["Citation"] < 1000]["Citation"].mean(), 2)
        citation_avg_with = round(df_col["Citation"].mean(), 2)
        container_citation_avg.metric(
            "Average citation", citation_avg,
            help=f"**{outlier_count}** item(s) >1000 citations. With outliers: **{citation_avg_with}**."
        )
    else:
        container_citation_avg.metric("Average citation", round(df_col["Citation"].mean(), 2))

    container_metric.metric("Items found", num_items, help=breakdown_string)
    container_citation.metric("Number of citations", int(citation_count))
    container_oa.metric("Open access coverage", f"{int(oa_ratio)}%", help="Journal articles only")
    container_type.metric("Number of publication types", int(item_type_no))
    container_author_no.metric("Number of authors", int(author_no))
    container_author_pub.metric("Author/publication ratio", author_pub_ratio)
    container_collab.metric("Collaboration ratio", f"{collaboration_ratio}%")

    # ── Report toggle + shareable link ────────────────────────────────────────
    if "col_profile_report" not in st.session_state:
        st.session_state["col_profile_report"] = st.query_params.get("report", "0") == "1"

    on_report = st.toggle(
        ":material/monitoring: Generate report",
        key="col_profile_report",
    )
    current_url_report = st.query_params.get("report", "0") == "1"
    if on_report != current_url_report:
        params = {"collection": collection_key}
        if children and selected_child_key != collection_key:
            params["subcollection"] = selected_child_key
        if on_report:
            params["report"] = "1"
        st.query_params.from_dict(params)

    link = (
        f"{BASE_URL}/?collection={collection_key}"
        f"{'&subcollection=' + selected_child_key if children and selected_child_key != collection_key else ''}"
        f"{'&report=1' if on_report else ''}"
    )
    st.caption(f"🔗 Shareable link: [{link}]({link})")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📑 Publications", "📊 Dashboard"])

    with tab1:
        if on_report:
            st.info(f"Report for {display_name}")
            render_report_charts(df_col, display_name, name_replacements)
        else:
            sort_by = st.radio(
                "Sort by:",
                ("Publication date :arrow_down:", "Publication type", "Citation", "Date added :arrow_down:"),
                horizontal=True, key="col_profile_sort",
            )
            if sort_by == "Publication date :arrow_down:":
                df_col = df_col.sort_values("Date published", ascending=False).reset_index(drop=True)
            elif sort_by == "Publication type":
                df_col = df_col.sort_values("Publication type", ascending=True).reset_index(drop=True)
            elif sort_by == "Citation":
                df_col = df_col.sort_values("Citation", ascending=False).reset_index(drop=True)
            else:
                df_col = df_col.sort_values("Date added", ascending=False).reset_index(drop=True)

            if view == "Basic list":
                with st.expander("**Basic list view**", expanded=True):
                    if sort_by == "Publication type":
                        current_type = None
                        count_by_type = {}
                        for _, row in df_col.iterrows():
                            if row["Publication type"] != current_type:
                                current_type = row["Publication type"]
                                st.subheader(current_type)
                                count_by_type[current_type] = 1
                            st.write(f"{count_by_type[current_type]}) {format_entry(row, include_citation=True, reviews_map=reviews_map, base_url=BASE_URL)}")
                            count_by_type[current_type] += 1
                            if display2:
                                st.caption(row["Abstract"])
                    else:
                        for count, (_, row) in enumerate(df_col.iterrows(), 1):
                            st.write(f"{count}) {format_entry(row, include_citation=True, reviews_map=reviews_map, base_url=BASE_URL)}")
                            if display2:
                                st.caption(row["Abstract"])

            elif view == "Table":
                with st.expander("**Table view**", expanded=True):
                    st.dataframe(
                        df_col[["Publication type", "Title", "Date published", "FirstName2",
                                "Abstract", "Publisher", "Journal", "Citation",
                                "Collection_Name", "Link to publication", "Zotero link"]]
                        .rename(columns={
                            "FirstName2": "Author(s)",
                            "Collection_Name": "Collection",
                            "Link to publication": "Publication link",
                        })
                    )
            else:
                with st.expander("**Bibliographic listing**", expanded=True):
                    df_col["zotero_item_key"] = df_col["Zotero link"].str.replace(
                        "https://www.zotero.org/groups/intelarchive_intelligence_studies_database/items/", ""
                    )
                    df_zot = pd.read_csv("zotero_citation_format.csv")
                    df_zot.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
                    df_col = pd.merge(df_col, df_zot, on="zotero_item_key", how="left")
                    display_bibliographies(df_col)

    with tab2:
        st.header("Dashboard")
        on_dash = st.toggle("Display dashboard", key="col_profile_dash")
        if on_dash and num_items > 0:
            col1, col2 = st.columns(2)
            with col1:
                df_plot = df_col["Publication type"].value_counts().reset_index()
                df_plot.columns = ["Publication type", "Count"]
                fig = px.pie(df_plot, values="Count", names="Publication type",
                             title=f"Publications: {display_name}")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.bar(df_plot, x="Publication type", y="Count",
                             color="Publication type",
                             title=f"Publications: {display_name}")
                st.plotly_chart(fig, use_container_width=True)

            df_year = df_col.copy()
            df_year["Date year"] = pd.to_datetime(
                df_year["Date published"], utc=True, errors="coerce"
            ).dt.strftime("%Y").fillna("No date")
            df_year_count = df_year["Date year"].value_counts().reset_index()
            df_year_count.columns = ["Publication year", "Count"]
            df_year_count = df_year_count[df_year_count["Publication year"] != "No date"]
            df_year_count = df_year_count.sort_values("Publication year")

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(df_year_count, x="Publication year", y="Count",
                             title=f"Publications by year: {display_name}")
                fig.update_xaxes(tickangle=-70)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                df_auth = df_col.copy()
                df_auth["Author_name"] = df_auth["FirstName2"].apply(
                    lambda x: x.split(", ") if isinstance(x, str) and x else []
                )
                df_auth = df_auth.explode("Author_name")
                df_auth["Author_name"] = df_auth["Author_name"].map(
                    name_replacements
                ).fillna(df_auth["Author_name"])
                max_authors = max(len(df_auth["Author_name"].unique()), 1)
                num_authors = st.slider(
                    "Select number of authors to display:",
                    1, min(50, max_authors), 20,
                    key="col_profile_authors_slider",
                )
                pub_by_author = df_auth["Author_name"].value_counts().head(num_authors)
                fig = px.bar(
                    pub_by_author, x=pub_by_author.index, y=pub_by_author.values,
                    title=f"Top {num_authors} authors ({display_name})",
                    labels={"x": "Author", "y": "Publications"},
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                number = st.select_slider(
                    "Select a number of publishers",
                    options=[5, 10, 15, 20, 25, 30], value=10,
                    key="col_profile_pub_slider",
                )
                df_pub = df_col["Publisher"].value_counts().head(number).reset_index()
                df_pub.columns = ["Publisher", "Count"]
                fig = px.bar(df_pub, x="Publisher", y="Count", color="Publisher",
                             title=f"Top {number} publishers")
                fig.update_xaxes(tickangle=-70)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                number2 = st.select_slider(
                    "Select a number of journals",
                    options=[5, 10, 15, 20, 25, 30], value=10,
                    key="col_profile_jour_slider",
                )
                df_jour = df_col[df_col["Publication type"] == "Journal article"][
                    "Journal"
                ].value_counts().head(number2).reset_index()
                df_jour.columns = ["Journal", "Count"]
                fig = px.bar(df_jour, x="Journal", y="Count", color="Journal",
                             title=f"Top {number2} journals")
                fig.update_xaxes(tickangle=-70)
                st.plotly_chart(fig, use_container_width=True)

            # ── Wordcloud ─────────────────────────────────────────────────────
            st.write("---")
            render_wordcloud(df_col, title=f"Top words in titles ({display_name})")

        elif on_dash and num_items == 0:
            st.warning("No data to visualise.")
        else:
            st.info("Toggle to see the dashboard!")

@st.cache_data(ttl=3600)
def compute_author_similarity(df_authors):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    # Build one text blob per author
    author_texts = (
        df_authors.groupby("Author_name")
        .apply(lambda g: " ".join(
            (g["Title"].fillna("") + " " + g["Abstract"].fillna("")).str.strip()
        ))
        .reset_index()
    )
    author_texts.columns = ["Author_name", "text"]
    author_texts = author_texts[author_texts["text"].str.len() > 10].reset_index(drop=True)

    # TF-IDF vectorisation
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(author_texts["text"])

    # Cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return author_texts, similarity_matrix


def get_similar_authors(author_name, df_authors, top_n=5):
    import numpy as np

    author_texts, similarity_matrix = compute_author_similarity(df_authors)

    if author_name not in author_texts["Author_name"].values:
        return []

    idx = author_texts[author_texts["Author_name"] == author_name].index[0]
    scores = similarity_matrix[idx]

    # Exclude self
    scores[idx] = 0

    top_indices = np.argsort(scores)[::-1][:top_n]
    results = []
    for i in top_indices:
        if scores[i] > 0:
            results.append({
                "author": author_texts.iloc[i]["Author_name"],
                "score": round(float(scores[i]), 3),
            })
    return results

def render_author_profile(author_name, df_dedup, df_duplicated, df_authors):
    reviews_map = load_reviews_map()

    # ── Header ──────────────────────────────────────────────────────────────
    author_slug = author_to_slug(author_name)
    preview_link = f"{BASE_URL}/?author_preview={author_slug}"
    st.markdown(f"## 👤 {author_name}")
    st.caption("Full publication profile · IntelArchive")
    st.divider()

    # ── Build author dataframe ───────────────────────────────────────────────
    adf = df_authors[df_authors["Author_name"] == author_name].copy()
    adf["Date published"] = parse_date_column(adf["Date published"])
    adf["Date published"] = adf["Date published"].fillna("")
    adf = sort_by_date(adf).sort_values(
        ["No date flag", "Date published"], ascending=[True, True]
    )

    # ── Themes dataframe (built once, used in popover + report) ─────────────
    fdc = pd.merge(df_duplicated, adf[["Zotero link"]], on="Zotero link")
    fdc = fdc[["Zotero link", "Collection_Key", "Collection_Name", "Collection_Link"]]
    fdc2 = fdc["Collection_Name"].value_counts().reset_index().head(10)
    fdc2.columns = ["Collection_Name", "Number_of_Items"]
    fdc2 = fdc2[fdc2["Collection_Name"] != "01 Intelligence history"]
    fdc = pd.merge(fdc2, fdc, on="Collection_Name", how="left") \
            .drop_duplicates("Collection_Name").reset_index(drop=True)
    fdc["Collection_Name"] = fdc["Collection_Name"].apply(remove_numbers)

    # ── Three-column metrics row ─────────────────────────────────────────────
    ca1, ca2, ca3, ca4 = st.columns(4)

    with ca1:
        c_m = st.container()

    with ca2:
        with st.popover("More metrics"):
            c_cit     = st.container()
            c_cit_avg = st.container()
            c_oa      = st.container()
            c_type    = st.container()
            c_collab  = st.container()

    with ca3:
        with st.popover("Relevant themes"):
            st.markdown("##### Top 5 relevant themes")
            for i, row in fdc.iterrows():
                col_key = str(row.get("Collection_Key", "")).strip()
                app_link = f"{BASE_URL}/?collection={col_key}" if col_key else row['Collection_Link']
                st.caption(
                    f"{i+1}) [{row['Collection_Name']}]({app_link}) "
                    f"· {row['Number_of_Items']} items"
                )

    with ca4:
        with st.popover("Similar authors"):
            st.markdown("##### Top 5 similar authors")
            with st.spinner("Finding similar authors..."):
                similar = get_similar_authors(author_name, df_authors)
            if similar:
                for s in similar:
                    s_slug = author_to_slug(s["author"])
                    profile_url = f"{BASE_URL}/?author_profile={s_slug}"
                    st.caption(f"[{s['author']}]({profile_url}) · {round(s['score'] * 100)}% match")
            else:
                st.caption("No similar authors found.")

    st.write("*This database **may not show** all research outputs of the author.*")

    # ── Metrics ──────────────────────────────────────────────────────────────
    render_metrics(
        adf,
        container_metric=c_m,
        container_citation=c_cit,
        container_citation_average=c_cit_avg,
        container_oa=c_oa,
        container_type=c_type,
        container_publication_ratio=c_collab,
    )

    # ── Report toggle + shareable link ───────────────────────────────────────
    slug = author_to_slug(author_name)
    default_report = st.query_params.get("report", "0") == "1"

    if "ap_report_state" not in st.session_state:
        st.session_state["ap_report_state"] = default_report

    if "ap_report" not in st.session_state:
        st.session_state["ap_report"] = st.query_params.get("report", "0") == "1"

    st.toggle(
        ":material/monitoring: Generate report",
        key="ap_report",
    )
    on = st.session_state["ap_report"]

    # Sync URL
    current_url_report = st.query_params.get("report", "0") == "1"
    if on != current_url_report:
        params = {"author_profile": slug}   # ← was "author_preview", fix to "author_profile"
        if on:
            params["report"] = "1"
        current_url_report = st.query_params.get("report", "0") == "1"
        if on != current_url_report or st.query_params.get("author_profile", "") != slug:
            st.query_params.from_dict(params)

        link = f"{BASE_URL}/?author_profile={slug}{'&report=1' if on else ''}"
        st.caption(f"🔗 Shareable link: [{link}]({link})")

    # ── Filters + download each in their own column ──────────────────────────
    col_types, col_view = st.columns([3,2])

    with col_types:
        types = st.multiselect(
            "Publication type",
            adf["Publication type"].unique(),
            default=[],
            key="ap_types",
        )
        if types:
            adf = adf[adf["Publication type"].isin(types)].reset_index(drop=True)

    with col_view:
        view = st.radio(
            "View as:", ("Basic list", "Table", "Bibliography"),
            horizontal=True, key="ap_view",
        )


    csv = convert_df_to_csv(
        adf[["Publication type", "Title", "Abstract", "Date published",
                "Publisher", "Journal", "Link to publication", "Zotero link", "Citation"]]
        .assign(Abstract=lambda d: d["Abstract"].str.replace("\n", " "))
    )
    st.download_button(
        "⬇ Download publications", csv,
        f"{author_name}_{datetime.date.today().isoformat()}.csv",
        mime="text/csv", key="dl-ap",
    )

    # ── Report or publications list ──────────────────────────────────────────
    if on and len(adf):
        st.info(f"Report for {author_name}")
        render_report_charts(
            adf, author_name, name_replacements,
            show_themes=True, themes_df=fdc,
        )
    elif not on:
        adf = sort_radio(adf, key="ap_sort")
        if view == "Basic list":
            for i, row in adf.iterrows():
                st.write(
                    f"{i+1}) {format_entry(row, include_citation=True, reviews_map=reviews_map, base_url=BASE_URL)}"
                )
        elif view == "Table":
            st.dataframe(
                adf[["Publication type", "Title", "Date published", "FirstName2",
                     "Abstract", "Publisher", "Journal", "Citation",
                     "Link to publication", "Zotero link"]]
                .rename(columns={
                    "FirstName2": "Author(s)",
                    "Link to publication": "Publication link",
                })
            )
        elif view == "Bibliography":
            adf["zotero_item_key"] = adf["Zotero link"].str.replace(
                "https://www.zotero.org/groups/"
                "intelarchive_intelligence_studies_database/items/", ""
            )
            df_zot = pd.read_csv("zotero_citation_format.csv")
            display_bibliographies(
                pd.merge(adf, df_zot, on="zotero_item_key", how="left")
            )
    else:
        st.write("No publication type selected.")



# ── Load data ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data():
    df_dedup        = pd.read_csv("all_items.csv")
    df_dedup["parentKey"] = df_dedup["Zotero link"].str.split("/").str[-1]
    df_duplicated   = pd.read_csv("all_items_duplicated.csv")
    df_authors      = get_df_authors()
    df_book_reviews = pd.read_csv("book_reviews.csv")
    return df_dedup, df_duplicated, df_authors, df_book_reviews
    
# ── Collection profile page ──────────────────────────────────────────────────
collection_profile_key = st.query_params.get("collection", "")

# Only trigger early-exit if it's a known collection key
# (prevents conflict with search_collection() which also uses ?collection=)
if collection_profile_key and collection_profile_key in COLLECTION_KEY_MAP:
    df_dedup_cp, df_duplicated_cp, _, _ = load_data()

    if st.button("← Back to search"):
        st.query_params.clear()
        st.rerun()

    render_collection_profile(
        collection_profile_key,
        df_dedup_cp,
        df_duplicated_cp,
    )

    st.write("---")
    display_custom_license()
    st.stop()


item_key = st.query_params.get("item", "")

if item_key:
    if st.button("← Back to home"):
        st.query_params.clear()
        st.rerun()

    # Load only what's needed — no spinner, no full load_data()
    df_dedup = pd.read_csv("all_items.csv")
    df_dedup["parentKey"] = df_dedup["Zotero link"].str.split("/").str[-1]

    item = df_dedup[df_dedup["parentKey"] == item_key]
    if not item.empty:
        row = item.iloc[0]

        st.subheader(row["Title"], anchor=False)
        st.divider()

        def _safe(val):
            return "" if pd.isna(val) or str(val).strip() in ("", "nan", "NaN") else str(val).strip()

        col1, col2 = st.columns(2)
        with col1:
            authors_raw = _safe(row.get('FirstName2'))
            if authors_raw:
                author_list = [a.strip() for a in authors_raw.split(",")]
                author_links = " · ".join(
                    f"[{a}]({BASE_URL}/?author_profile={author_to_slug(_resolve_author(a))})"
                    for a in author_list if a
                )
                st.markdown(f"**Authors:** {author_links}")
            else:
                st.markdown("**Authors:** N/A")

            st.markdown(f"**Publication type:** {_safe(row.get('Publication type')) or 'N/A'}")
            date_published = _safe(row.get('Date published'))
            if date_published:
                try:
                    date_published_fmt = pd.to_datetime(date_published, utc=True).strftime("%d %B %Y")
                except Exception:
                    date_published_fmt = date_published
            else:
                date_published_fmt = 'N/A'
            st.markdown(f"**Date published:** {date_published_fmt}")
            date_added = _safe(row.get('Date added'))
            if date_added:
                try:
                    date_added_fmt = pd.to_datetime(date_added).strftime("%d %B %Y")
                except Exception:
                    date_added_fmt = date_added
                st.markdown(f"**Date added to IntelArchive:** {date_added_fmt}")
            pub_type = _safe(row.get('Publication type'))

            if pub_type == "Book chapter":
                book_title = _safe(row.get('Book_title'))
                if book_title:
                    st.markdown(f"**Book title:** {book_title}")

            if pub_type == "Thesis":
                thesis_type = _safe(row.get('Thesis_type'))
                university  = _safe(row.get('University'))
                if thesis_type:
                    st.markdown(f"**Thesis type:** {thesis_type}")
                if university:
                    st.markdown(f"**University:** {university}")

            publisher = _safe(row.get('Publisher'))
            journal   = _safe(row.get('Journal'))
            if journal:
                st.markdown(f"**Journal:** {journal}")
            elif publisher:
                st.markdown(f"**Publisher:** {publisher}")

        with col2:
            citation_val = row.get('Citation', 0)
            citation_int = 0 if pd.isna(citation_val) else int(float(citation_val))
            st.markdown(f"**Citations:** {citation_int}")
            st.markdown(f"**OA status:** {'Open Access' if row.get('OA status') else 'Not OA'}")

            # Load duplicated df to get all collections for this item
            df_dup_item = pd.read_csv("all_items_duplicated.csv")
            df_dup_item["parentKey"] = df_dup_item["Zotero link"].str.split("/").str[-1]
            item_collections = df_dup_item[df_dup_item["parentKey"] == item_key][
                ["Collection_Name", "Collection_Link", "Collection_Key"]
            ].drop_duplicates()

            if not item_collections.empty:
                collection_links = []
                for _, col_row in item_collections.iterrows():
                    col_name = str(col_row.get("Collection_Name", "")).strip()
                    col_link = str(col_row.get("Collection_Link", "")).strip()
                    col_key  = str(col_row.get("Collection_Key", "")).strip()
                    if col_name and col_name not in ("nan", ""):
                        clean_name = re.sub(r"^\d+[\.\d]*\s*", "", col_name).strip()
                        app_link   = f"{BASE_URL}/?collection={col_key}"
                        collection_links.append(f"[{clean_name}]({app_link})")
                
                st.markdown(f"**Collections:** {' | '.join(collection_links)}")
            else:
                st.markdown("**Collections:** N/A")

            # ── External links ──────────────────────────────────────────────────
            st.markdown("**External links:**")

            links_md = []

            zotero_link = _safe(row.get('Zotero link'))
            if zotero_link:
                links_md.append(f"[:red-badge[Zotero link]]({zotero_link})")

            pub_link = _safe(row.get('Link to publication'))
            if pub_link:
                links_md.append(f"[:blue-badge[Publication link]]({pub_link})")

            oa_link = _safe(row.get('OA_link')).replace(" ", "%20")
            if oa_link:
                links_md.append(f"[:green-badge[OA version]]({oa_link})")

            citation_link = _safe(row.get('Citation_list'))
            if citation_link and citation_int > 0:
                links_md.append(f"[:orange-badge[Cited by {citation_int}]]({citation_link})")

            reviews_map = load_reviews_map()
            parent_key  = _safe(row.get('parentKey'))
            if not parent_key:
                parent_key = zotero_link.rstrip("/").split("/")[-1] if zotero_link else ""

            if reviews_map and parent_key:
                review_links = reviews_map.get(parent_key, [])
                if len(review_links) == 1:
                    links_md.append(f"[:violet-badge[Book review]]({review_links[0]})")
                else:
                    for i, link in enumerate(review_links, 1):
                        links_md.append(f"[:violet-badge[Book review {i}]]({link})")

            st.markdown(" ".join(links_md))

        st.divider()
        st.markdown("**Abstract:**")
        st.info(row.get("Abstract", "No abstract available"))

        st.divider()
        st.markdown("**Cite this publication:**")

        # Get the zotero item key and look up citation format
        zotero_item_key = _safe(row.get('Zotero link')).replace(
            "https://www.zotero.org/groups/intelarchive_intelligence_studies_database/items/", ""
        )

        df_zot = pd.read_csv("zotero_citation_format.csv")
        citation_row = df_zot[df_zot["zotero_item_key"] == zotero_item_key]

        if not citation_row.empty:
            display_bibliographies(citation_row)
        else:
            st.info("Citation format not available for this item.")

    else:
        st.warning("Publication not found.")

    st.divider()
    st.markdown("**Similar publications:**")

    def get_related_publications(row, df, top_n=5):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        # Combine title and abstract for the current item
        current_text = f"{row.get('Title', '')} {row.get('Abstract', '')}".strip()
        if not current_text:
            return pd.DataFrame()

        # Combine title and abstract for all items
        df = df.copy()
        df["_text"] = (
            df["Title"].fillna("") + " " + df["Abstract"].fillna("")
        ).str.strip()

        # Remove the current item
        current_zotero = row.get("Zotero link", "")
        df = df[df["Zotero link"] != current_zotero].reset_index(drop=True)

        # Filter out items with no text
        df = df[df["_text"].str.len() > 10].reset_index(drop=True)

        if df.empty:
            return pd.DataFrame()

        # TF-IDF vectorisation
        all_texts = [current_text] + df["_text"].tolist()
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=5000,
                ngram_range=(1, 2),
            )
            tfidf_matrix = vectorizer.fit_transform(all_texts)
        except Exception:
            return pd.DataFrame()

        # Cosine similarity between current item and all others
        current_vec = tfidf_matrix[0]
        other_vecs  = tfidf_matrix[1:]
        scores = cosine_similarity(current_vec, other_vecs).flatten()

        # Get top N
        top_indices = np.argsort(scores)[::-1][:top_n]
        top_df = df.iloc[top_indices].copy()
        top_df["_score"] = scores[top_indices]
        top_df = top_df[top_df["_score"] > 0]

        return top_df

    with st.spinner("Finding related publications..."):
        df_all_items = pd.read_csv("all_items.csv")
        related = get_related_publications(row, df_all_items, top_n=5)

    if not related.empty:
        reviews_map_rel = load_reviews_map()
        for i, (_, rel_row) in enumerate(related.iterrows(), 1):
            st.write(
                f"{i}) {format_entry(rel_row, include_citation=True, reviews_map=reviews_map_rel, base_url=BASE_URL)}"
            )
    else:
        st.info("No related publications found.")

    st.write("---")
    display_custom_license()
    
    st.stop()

# ── Author profile page ─────────────────────────────────────────────────────
author_profile_slug = st.query_params.get("author_profile", "")

if author_profile_slug:
    df_dedup_ap, df_duplicated_ap, df_authors_ap, _ = load_data()

    matched_author = slug_to_author(
        author_profile_slug,
        df_authors_ap["Author_name"].unique().tolist()
    )

    if st.button("← Back to search"):
        st.query_params.clear()
        st.rerun()

    if not matched_author:
        st.warning("Author not found.")
        st.stop()

    render_author_profile(
        matched_author,
        df_dedup_ap,
        df_duplicated_ap,
        df_authors_ap,
    )

    st.write("---")
    display_custom_license()
    st.stop()



with st.spinner("Retrieving data..."):
    df_dedup, df_duplicated, df_authors, df_book_reviews = load_data()

# ── Everything below is OUTSIDE the spinner ─────────────────────────────────
col1, col2, col3 = st.columns([3, 5, 8])
with col3:
    with st.expander("Introduction"):
        st.info(intro)

df_intro = df_dedup.copy()
df_intro["Date added"] = pd.to_datetime(df_intro["Date added"])
current_date = pd.to_datetime("now", utc=True)
items_this_month = df_intro[
    (df_intro["Date added"].dt.year  == current_date.year) &
    (df_intro["Date added"].dt.month == current_date.month)
]
with col1:
    st.metric(
        label="Number of items in the library",
        value=len(df_intro),
        delta=len(items_this_month),
        help=f"**{len(items_this_month)}** items added in {current_date.strftime('%B %Y')}",
    )

st.write("The library last updated on **" + df_intro.loc[0]["Date added"].strftime("%d/%m/%Y, %H:%M") + "**")

with col2:
    with st.popover("More metrics"):
        citation_count  = df_dedup["Citation"].sum()
        non_nan_cited   = df_dedup.dropna(subset=["Citation_list"])
        citation_mean   = non_nan_cited["Citation"].mean()
        citation_median = non_nan_cited["Citation"].median()
        outlier_count   = int((df_dedup["Citation"] > 1000).sum())
        avg_wo_outliers = round(df_dedup.loc[df_dedup["Citation"] < 1000, "Citation"].mean(), 2)

        st.metric(label="Number of citations", value=int(citation_count),
                    help="Citations from [OpenAlex](https://openalex.org/).")
        st.metric(label="Average citation",
                    value=round(df_dedup["Citation"].mean(), 2),
                    help=f"**{outlier_count}** outliers >1000. Without outliers: **{avg_wo_outliers}**. Median: **{round(citation_median,1)}**.")

        ja = df_dedup[df_dedup["Publication type"] == "Journal article"]
        oa_ratio = (ja["OA status"].sum() / len(ja) * 100) if len(ja) else 0
        st.metric(label="Open access coverage", value=f"{int(oa_ratio)}%", help="Journal articles only")
        st.metric(label="Number of publication types", value=int(df_dedup["Publication type"].nunique()))

        df_no_thesis = df_dedup[df_dedup["Publication type"] != "Thesis"]
        expanded      = split_and_expand(df_no_thesis["FirstName2"])
        author_no     = len(expanded)
        item_count    = len(df_no_thesis)
        st.metric(label="Number of authors", value=int(author_no))
        st.metric(label="Author/publication ratio", value=round(author_no / item_count, 2))
        multi = df_no_thesis["FirstName2"].astype(str).apply(lambda x: "," in x).sum()
        st.metric(label="Collaboration ratio", value=f"{round(multi/item_count*100,1)}%")


sidebar_content()

tab1, tab2, tab3 = st.tabs(["📑 Publications", "📊 Dashboard", "💬 Chat"])

# ════════════════════════════════════════════════════════════════════════
# TAB 1
# ════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([6, 2])
    with col1:

        @st.fragment
        def search_options_main_menu():
            def parse_search_terms(search_term):
                """Tokenise search input into a flat list for the recursive parser."""
                tokens = []
                for tok in re.findall(r'(?:"[^"]*"|\S+)', search_term):
                    if tok.startswith("(") and len(tok) > 1:
                        tokens.append("(")
                        tokens.append(tok[1:])
                    elif tok.endswith(")") and len(tok) > 1:
                        tokens.append(tok[:-1])
                        tokens.append(")")
                    else:
                        tokens.append(tok)
                return [t for t in tokens if t]


            def apply_boolean_search(df, tokens, search_in):
                if not tokens:
                    return df

                def term_mask(df, term):
                    """Return a boolean Series: does this term match each row?"""
                    if "*" in term:
                        pattern = re.escape(term).replace(r"\*", r"\w*")
                        regex = rf"(?i)\b{pattern}"
                    else:
                        escaped = re.escape(term)
                        # Only use word boundaries if term starts/ends with word characters
                        prefix = r"\b" if re.match(r"\w", term[0]) else ""
                        suffix = r"\b" if re.match(r"\w", term[-1]) else ""
                        regex = rf"(?i){prefix}{escaped}{suffix}"

                    title_match = df["Title"].str.contains(regex, na=False, regex=True)
                    if search_in == "Title and abstract":
                        abs_match = df["Abstract"].str.contains(regex, na=False, regex=True)
                        return title_match | abs_match
                    return title_match

                def near_mask(df, term1, term2, n):
                    """Return a boolean Series: do term1 and term2 appear within n words?"""
                    cols = ["Title", "Abstract"] if search_in == "Title and abstract" else ["Title"]
                    def check(text):
                        if not isinstance(text, str):
                            return False
                        words = text.lower().split()
                        p1 = [i for i, w in enumerate(words) if re.search(rf"\b{re.escape(term1.lower())}\b", w)]
                        p2 = [i for i, w in enumerate(words) if re.search(rf"\b{re.escape(term2.lower())}\b", w)]
                        return any(abs(a - b) <= n for a in p1 for b in p2)
                    mask = df[cols[0]].apply(check)
                    for col in cols[1:]:
                        mask = mask | df[col].apply(check)
                    return mask

                import pandas as pd

                # ── Recursive descent parser ────────────────────────────────────────────
                pos = [0]  # use list so nested functions can mutate it

                def peek():
                    while pos[0] < len(tokens) and tokens[pos[0]] == "":
                        pos[0] += 1
                    return tokens[pos[0]] if pos[0] < len(tokens) else None

                def consume():
                    tok = peek()
                    pos[0] += 1
                    return tok

                def parse_expr():
                    return parse_or()

                def parse_or():
                    left = parse_and()
                    while peek() == "OR":
                        consume()
                        right = parse_and()
                        left = left | right
                    return left

                def parse_and():
                    left = parse_not()
                    while peek() not in (None, "OR", ")"):
                        if peek() == "AND":
                            consume()
                        right = parse_not()
                        left = left & right
                    return left

                def parse_not():
                    if peek() == "NOT":
                        consume()
                        operand = parse_atom()
                        return ~operand
                    return parse_atom()
                def parse_atom():
                    tok = peek()
                    if tok is None:
                        return pd.Series([False] * len(df), index=df.index)
                    if tok == "(":
                        consume()  # eat "("
                        result = parse_expr()
                        if peek() == ")":
                            consume()  # eat ")"
                        return result
                    # Check for NEAR: term1 NEAR/N term2
                    if pos[0] + 1 < len(tokens) and re.match(r"^NEAR/\d+$", tokens[pos[0] + 1] or "", re.I):
                        term1 = consume()
                        near_tok = consume()
                        n = int(re.search(r"\d+", near_tok).group())
                        term2 = consume()
                        return near_mask(df, term1, term2, n)
                    # Plain term or quoted phrase
                    consume()
                    term = tok.strip('"')
                    return term_mask(df, term)

                try:
                    mask = parse_expr()
                    return df[mask]
                except Exception:
                    return pd.DataFrame()


            st.header("Search in database", anchor=False)
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

            OPTION_MAP = {
                0: "Search keywords",
                1: "Search author",
                2: "Search collection",
                3: "Publication types",
                4: "Search journal",
                5: "Publication year",
                6: "Cited papers",
            }
            qp = st.query_params

            if qp.get("author_preview"):
                default_pill = 1
            elif qp.get("collection_preview"):
                default_pill = 2
            elif qp.get("type"):
                default_pill = 3
            elif qp.get("journal"):
                default_pill = 4
            elif qp.get("year_from") or qp.get("year_to"):
                default_pill = 5
            elif qp.get("query"):
                default_pill = 0
            else:
                default_pill = 0

            if "search_pills" not in st.session_state:
                st.session_state["search_pills"] = default_pill

            search_option = st.pills(
                "Select search option",
                options=list(OPTION_MAP.keys()),
                format_func=lambda o: OPTION_MAP[o],
                selection_mode="single",
                default=st.session_state["search_pills"],
                key="search_pills",
            )

            # ================================================================
            # 0 – KEYWORD SEARCH
            # ================================================================
            if search_option == 0:
                # Clear any non-keyword params when switching to keyword search
                non_keyword_params = {"author", "collection", "type", "journal", "year_from", "year_to"}
                if any(k in st.query_params for k in non_keyword_params):
                    st.query_params.clear()
                # Clear stale session state
                for key in ["auth_types", "auth_sort", "report_author_state",
                            "col_types", "col_sort", "type_sort",
                            "journal_sort", "year_sort", "cited_sort"]:
                    st.session_state.pop(key, None)
                st.subheader("Search keywords", anchor=False, divider="blue")


                @st.fragment
                def search_keyword():
                    reviews_map = load_reviews_map()

                    @st.dialog("Search guide")
                    def guide(_):
                        st.write("""
                            **Operators:** AND, OR, NOT  
                            **Grouping:** (cia OR mi6) AND humint  
                            **Exact phrase:** "covert action"  
                            **Wildcard:** intel* (matches intelligence, intelligent...)  
                            **Proximity:** cia NEAR/5 humint (within 5 words)  
                            **Example:** (cia OR mi6) AND NOT "cold war"
                        """)

                    if "guide" not in st.session_state:
                        if st.button("Search guide"):
                            guide("Search guide")

                    def update_search_params():
                        st.session_state.search_term = st.session_state.search_term_input
                        params = {
                            "search_in": st.session_state.search_in,
                            "query":     st.session_state.search_term,
                        }
                        if st.session_state.get("report_keyword_state", False):
                            params["report"] = "1"
                        st.query_params.from_dict(params)

                    for k, default in [("search_term",       qp.get("query",     "")),
                                        ("search_in",         qp.get("search_in", "Title")),
                                        ("search_term_input", qp.get("query",     ""))]:
                        if k not in st.session_state:
                            st.session_state[k] = default

                    search_options = ["Title", "Title and abstract"]
                    try:
                        si_index = search_options.index(qp.get("search_in", "Title"))
                    except ValueError:
                        si_index = 0

                    cols, cola = st.columns([2, 6])
                    with cols:
                        st.session_state.search_in = st.selectbox(
                            "Search in", search_options, index=si_index,
                            on_change=update_search_params,
                        )
                    with cola:
                        st.text_input(
                            "Search keywords in titles or abstracts",
                            st.session_state.search_term_input,
                            key="search_term_input",
                            placeholder="Type your keyword(s)",
                            on_change=update_search_params,
                        )

                    search_term = st.session_state.search_term.strip()

     
                    if not search_term:
                        st.session_state.pop("report_keyword_state", None)  # ← reset on clear
                        st.info("Please enter a keyword to search in title or abstract.")
                        return

                    with st.status(f"Searching publications for '**{search_term}**'...", expanded=True) as status:
                        tokens      = parse_search_terms(search_term)
                        df_csv      = df_duplicated.copy()
                        filtered_df = apply_boolean_search(df_csv, tokens, st.session_state.search_in)
                        filtered_df_for_collections = filtered_df.copy()
                        filtered_df = filtered_df.drop_duplicates()

                        if not filtered_df.empty and "Date published" in filtered_df.columns:
                            filtered_df["Date published"] = parse_date_column(filtered_df["Date published"])
                            filtered_df["Date published"] = filtered_df["Date published"].fillna("")
                            filtered_df = sort_by_date(filtered_df).sort_values(
                                ["No date flag", "Date published"], ascending=[True, True])

                        types       = filtered_df["Publication type"].dropna().unique()
                        collections = filtered_df["Collection_Name"].dropna().unique()

                        cs1, cs2, cs3, cs4 = st.columns(4)
                        with cs1:
                            c_metric = st.container()
                        with cs2:
                            with st.popover("More metrics"):
                                c_cit      = st.container()
                                c_cit_avg  = st.container()
                                c_oa       = st.container()
                                c_type     = st.container()
                                c_auth_no  = st.container()
                                c_auth_rat = st.container()
                                c_collab   = st.container()
                        with cs3:
                            with st.popover("Relevant themes"):
                                st.markdown("##### Top relevant publication themes")
                                fdc = filtered_df_for_collections[
                                    ["Zotero link","Collection_Key","Collection_Name","Collection_Link"]
                                ].copy()
                                fdc2 = fdc["Collection_Name"].value_counts().reset_index().head(5)
                                fdc2.columns = ["Collection_Name","Number_of_Items"]
                                fdc  = pd.merge(fdc2, fdc, on="Collection_Name", how="left").drop_duplicates("Collection_Name").reset_index(drop=True)
                                fdc["Collection_Name"] = fdc["Collection_Name"].apply(remove_numbers)
                                for i, row in fdc.iterrows():
                                    st.caption(f"{i+1}) [{row['Collection_Name']}]({row['Collection_Link']}) {row['Number_of_Items']} items")
                        with cs4:
                            with st.popover("Filters and more"):
                                types2       = st.multiselect("Publication types", types, key="kw_types")
                                collections2 = st.multiselect("Collection", collections, key="kw_collections")
                                c_dl         = st.container()
                                display_abstracts = st.checkbox("Display abstracts")
                                only_cited   = st.checkbox("Show cited items only")
                                view         = st.radio("View as:", ("Basic list","Table","Bibliography"), horizontal=True)

                        if types2:
                            filtered_df = filtered_df[filtered_df["Publication type"].isin(types2)]
                        if collections2:
                            filtered_df = filtered_df[filtered_df["Collection_Name"].isin(collections2)]
                        if only_cited:
                            filtered_df = filtered_df[(filtered_df["Citation"].notna()) & (filtered_df["Citation"] != 0)]
                        filtered_df = filtered_df.drop_duplicates(subset=["Zotero link"], keep="first")
                        num_items   = len(filtered_df)

                        if num_items:
                            render_metrics(
                                filtered_df,
                                container_metric=c_metric,
                                container_citation=c_cit,
                                container_citation_average=c_cit_avg,
                                container_oa=c_oa,
                                container_type=c_type,
                                container_author_no=c_auth_no,
                                container_author_pub_ratio=c_auth_rat,
                                container_publication_ratio=c_collab,
                            )
                            csv = convert_df_to_csv(
                                filtered_df[["Publication type","Title","Abstract","Date published",
                                            "Publisher","Journal","Link to publication","Zotero link","Citation"]]
                                .assign(Abstract=lambda d: d["Abstract"].str.replace("\n"," "))
                                .reset_index(drop=True)
                            )
                            c_dl.download_button(
                                "Download search", csv,
                                f"search-result-{datetime.date.today().isoformat()}.csv",
                                mime="text/csv", key="dl-kw", icon=":material/download:",
                            )
                            if "report_keyword" not in st.session_state:
                                st.session_state["report_keyword"] = st.query_params.get("report", "0") == "1"

                            st.toggle(
                                ":material/monitoring: Generate report",
                                key="report_keyword",
                            )

                            on = st.session_state["report_keyword"]

                            # Sync session state tracker
                            st.session_state["report_keyword_state"] = on

                            # Update URL without triggering extra rerun by checking against URL directly
                            current_url_report = st.query_params.get("report", "0") == "1"
                            if on != current_url_report:
                                params = {"search_in": st.session_state.search_in, "query": st.session_state.search_term}
                                if on:
                                    params["report"] = "1"
                                st.query_params.from_dict(params)

                            link = (
                                f"https://intelligence.streamlit.app/"
                                f"?search_in={st.session_state.search_in}"
                                f"&query={st.session_state.search_term.replace(' ', '+')}"
                                f"{'&report=1' if on else ''}"
                            )
                            st.caption(f"🔗 Shareable link: [{link}]({link})")

                            if on:
                                st.info(f"Dashboard for: {search_term}")
                                render_report_charts(filtered_df, search_term, name_replacements,
                                                    show_themes=True, themes_df=fdc)
                            else:
                                filtered_df = sort_radio(filtered_df, key="kw_sort")
                                if view == "Basic list":
                                    articles = [format_entry(row, include_citation=True, reviews_map=reviews_map, base_url=BASE_URL) for _, row in filtered_df.iterrows()]
                                    abstracts = [row["Abstract"] if pd.notnull(row["Abstract"]) else "N/A" for _, row in filtered_df.iterrows()]
                                    render_paginated_list(filtered_df, articles, abstracts,
                                                        display_abstracts=display_abstracts,
                                                        search_tokens=tokens,
                                                        search_in=st.session_state.search_in)
                                elif view == "Table":
                                    st.dataframe(
                                        filtered_df[["Publication type","Title","Date published","FirstName2",
                                                    "Abstract","Publisher","Journal","Collection_Name",
                                                    "Link to publication","Zotero link"]]
                                        .rename(columns={"FirstName2":"Author(s)","Collection_Name":"Collection",
                                                        "Link to publication":"Publication link"})
                                    )
                                elif view == "Bibliography":
                                    filtered_df["zotero_item_key"] = filtered_df["Zotero link"].str.replace(
                                        "https://www.zotero.org/groups/intelarchive_intelligence_studies_database/items/","")
                                    df_zot = pd.read_csv("zotero_citation_format.csv")
                                    display_bibliographies(pd.merge(filtered_df, df_zot, on="zotero_item_key", how="left"))
                        else:
                            c_metric.metric(label="Number of items found", value=0)
                            st.write("No articles found with the given keyword/phrase.")

                        status.update(
                            label=f"Search found **{num_items}** {'matching source' if num_items == 1 else 'matching sources'} for '**{search_term}**'.",
                            state="complete", expanded=True,
                        )

                search_keyword()

            # ================================================================
            # 1 – AUTHOR SEARCH
            # ================================================================

            elif search_option == 1:
                st.subheader("Search author", anchor=False, divider="blue")

                # Clear stale params when switching to author search
                for key in ["search_term", "search_term_input", "search_in", "report_keyword_state",
                            "col_types", "col_sort", "type_sort",
                            "journal_sort", "year_sort", "cited_sort"]:
                    st.session_state.pop(key, None)
                    
                @st.fragment
                def search_author():
                    reviews_map    = load_reviews_map()
                    pub_counts     = df_authors["Author_name"].value_counts().to_dict()
                    sorted_authors = sorted(df_authors["Author_name"].unique(),
                                            key=lambda a: pub_counts.get(a, 0), reverse=True)
                    options = [""] + [f"{a} ({pub_counts.get(a,0)})" for a in sorted_authors]

                    # ── Only read from URL on first load ──────────────────────────
                    if "author_selectbox" not in st.session_state:
                        default_slug  = st.query_params.get("author_preview", "")
                        default_index = 0
                        if default_slug:
                            matched = slug_to_author(default_slug, [o.split(" (")[0] for o in options if o])
                            if matched:
                                default_index = next(
                                    (i for i, o in enumerate(options) if o.startswith(matched + " (")), 0
                                )
                        st.session_state["author_selectbox"] = options[default_index]

                    selected_display = st.selectbox(
                        "Select author", options,
                        key="author_selectbox",
                    )
                    selected_author = selected_display.split(" (")[0] if selected_display else None

                    if not selected_author:
                        st.write("Select an author to see items")
                        return

                    slug = author_to_slug(selected_author)

                    if "report_author_state" not in st.session_state:
                        st.session_state["report_author_state"] = st.query_params.get("report", "0") == "1"

                    # ── Only update URL if slug has changed ───────────────────────
                    if st.query_params.get("author_preview", "") != slug:
                        st.query_params.from_dict({"author_preview": slug})

                    preview_link = f"{BASE_URL}/?author_preview={slug}"
                    st.caption(f"🔗 Shareable link: [{preview_link}]({preview_link})")

                    adf = df_authors[df_authors["Author_name"] == selected_author].copy()
                    adf["Date published"] = parse_date_column(adf["Date published"])
                    adf["Date published"] = adf["Date published"].fillna("")
                    adf = sort_by_date(adf).sort_values(["No date flag","Date published"], ascending=[True,True])

                    with st.expander("Click to expand", expanded=True):
                        st.subheader(f"Publications by {selected_author}", anchor=False, divider="blue")
                        st.write("*This database **may not show** all research outputs of the author.*")

                        profile_link = f"{BASE_URL}/?author_profile={slug}"
                        st.link_button("👤 View full profile", profile_link)

                        # ── Quick stats ──────────────────────────────────────────────────────
                        total_pubs   = len(adf)
                        total_cit    = int(adf["Citation"].sum()) if "Citation" in adf.columns else 0
                        top_type     = adf["Publication type"].value_counts().idxmax() if total_pubs else "N/A"

                        qs1, qs2, qs3 = st.columns(3)
                        qs1.metric("Publications", total_pubs)
                        qs2.metric("Total citations", total_cit)
                        qs3.metric("Most common type", top_type)

                        # ── Top 3 themes ─────────────────────────────────────────────────────
                        st.markdown("**Top themes:**")
                        fdc  = pd.merge(df_duplicated, adf[["Zotero link"]], on="Zotero link")
                        fdc  = fdc[["Zotero link", "Collection_Key", "Collection_Name", "Collection_Link"]]
                        fdc2 = fdc["Collection_Name"].value_counts().reset_index().head(4)
                        fdc2.columns = ["Collection_Name", "Number_of_Items"]
                        fdc2 = fdc2[fdc2["Collection_Name"] != "01 Intelligence history"].head(3)
                        fdc  = pd.merge(fdc2, fdc, on="Collection_Name", how="left") \
                                .drop_duplicates("Collection_Name").reset_index(drop=True)
                        fdc["Collection_Name"] = fdc["Collection_Name"].apply(remove_numbers)
                        theme_links = []
                        for _, row in fdc.iterrows():
                            col_key = str(row.get("Collection_Key", "")).strip()
                            app_link = f"{BASE_URL}/?collection={col_key}" if col_key else row['Collection_Link']
                            theme_links.append(f"[{row['Collection_Name']}]({app_link})")
                        st.caption(" | ".join(theme_links))

                        # ── 5 most recent publications ───────────────────────────────────────
                        st.markdown("**5 most recent publications:**")
                        recent = adf.copy()
                        recent["_sort_date"] = pd.to_datetime(
                            recent["Date published"], errors="coerce", utc=True
                        )
                        recent = recent.sort_values("_sort_date", ascending=False) \
                                    .drop(columns=["_sort_date"]).head(5)
                        for i, (_, row) in enumerate(recent.iterrows(), 1):
                            st.write(
                                f"{i}) {format_entry(row, include_citation=True, reviews_map=reviews_map, base_url=BASE_URL)}"
                            )

                        st.divider()

                search_author()

            # ================================================================
            # 2 – COLLECTION SEARCH
            # ================================================================
            elif search_option == 2:

                for key in ["search_term", "search_term_input", "search_in"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.subheader("Search collection", anchor=False, divider="blue")

                @st.fragment
                def search_collection():
                    reviews_map  = load_reviews_map()
                    df_csv_col   = df_duplicated.copy()
                    df_csv_col["Collection_Name"] = df_csv_col["Collection_Name"].apply(remove_numbers)
                    excluded     = {"KCL intelligence","Events","Journals",""}
                    col_counts   = df_csv_col["Collection_Name"].value_counts()
                    sorted_cols  = [c for c in col_counts.index if c not in excluded]
                    options      = [""] + [f"{c} [{col_counts[c]} items]" for c in sorted_cols]

                    key_to_option = {}
                    for c in sorted_cols:
                        # Find the collection key for this collection name
                        match = df_csv_col[df_csv_col["Collection_Name"] == c]["Collection_Key"].iloc[0] if len(df_csv_col[df_csv_col["Collection_Name"] == c]) > 0 else None
                        if match:
                            key_to_option[match] = f"{c} [{col_counts[c]} items]"

                    if "collection_selectbox" not in st.session_state:
                        default_key = st.query_params.get("collection_preview", "")
                        default_col_index = 0
                        if default_key and default_key in key_to_option:
                            target_option = key_to_option[default_key]
                            default_col_index = next(
                                (i for i, o in enumerate(options) if o == target_option), 0
                            )
                        st.session_state["collection_selectbox"] = options[default_col_index]

                    sel_display  = st.selectbox(
                        "Select a collection", options,
                        key="collection_selectbox",
                    )
                    selected_col = sel_display.rsplit(" [", 1)[0] if sel_display else None

                    if selected_col:
                        col_key = df_csv_col[df_csv_col["Collection_Name"] == selected_col]["Collection_Key"].iloc[0]
                        if st.query_params.get("collection_preview", "") != col_key:
                            st.query_params.from_dict({"collection_preview": col_key})
                    else:
                        st.query_params.clear()

                    if not selected_col:
                        st.write("Pick a collection to see items")
                        return

                    cdf = df_csv_col[df_csv_col["Collection_Name"] == selected_col].copy()
                    cdf["Date published"] = parse_date_column(cdf["Date published"])
                    cdf["Date published"] = cdf["Date published"].fillna("")
                    cdf = sort_by_date(cdf).sort_values(["No date flag","Date published"], ascending=[True,True])
                    collection_link = cdf["Collection_Link"].iloc[0]

                    with st.expander("Click to expand", expanded=True):
                        st.markdown(f"#### Collection theme: {selected_col}")
                        st.write(f"*See the collection in [Zotero]({collection_link})*")

                        # ── Link to full profile ──────────────────────────────
                        profile_link = f"{BASE_URL}/?collection={col_key}"
                        st.link_button("📁 View full collection profile", profile_link)

                        # ── Quick stats ───────────────────────────────────────
                        total_items  = len(cdf)
                        total_cit    = int(cdf["Citation"].sum()) if "Citation" in cdf.columns else 0
                        top_type     = cdf["Publication type"].value_counts().idxmax() if total_items else "N/A"

                        qs1, qs2, qs3 = st.columns(3)
                        qs1.metric("Items", total_items)
                        qs2.metric("Total citations", total_cit)
                        qs3.metric("Most common type", top_type)

                        # ── Top 3 authors ──────────────────────────────────────
                        st.markdown("**Top authors:**")
                        top_authors = (
                            cdf["FirstName2"].str.split(", ")
                            .explode().str.strip()
                            .apply(_resolve_author)  # ← resolve canonical name
                            .value_counts().head(3)
                        )
                        author_links = []
                        for author, count in top_authors.items():
                            slug = author_to_slug(author)
                            author_links.append(f"[{author}]({BASE_URL}/?author_profile={slug}) ({count})")
                        st.caption(" | ".join(author_links))

                        # ── 5 most recent publications ────────────────────────
                        st.markdown("**5 most recent publications:**")
                        recent = cdf.copy()
                        recent["_sort_date"] = pd.to_datetime(
                            recent["Date published"], errors="coerce", utc=True
                        )
                        recent = recent.sort_values("_sort_date", ascending=False) \
                                       .drop(columns=["_sort_date"]).head(5)
                        reviews_map_col = load_reviews_map()
                        for i, (_, row) in enumerate(recent.iterrows(), 1):
                            st.write(
                                f"{i}) {format_entry(row, include_citation=True, reviews_map=reviews_map_col, base_url=BASE_URL)}"
                            )

                        st.divider()
                        st.caption(f"🔗 Shareable link: [{BASE_URL}/?collection_preview={col_key}]({BASE_URL}/?collection_preview={col_key})")

                search_collection()

            # ================================================================
            # 3 – PUBLICATION TYPES
            # ================================================================
            elif search_option == 3:
                for key in ["type_selectbox"]:
                    st.session_state.pop(key, None)
                st.subheader("Publication types", anchor=False, divider="blue")

                @st.fragment
                def type_selection():
                    reviews_map  = load_reviews_map()
                    unique_types = [""] + list(df_authors["Publication type"].unique())

                    # Pre-select from URL if ?type= is present
                    if "type_selectbox" not in st.session_state:
                        default_type       = st.query_params.get("type", "").replace("+", " ")
                        default_type_index = 0
                        if default_type and default_type in unique_types:
                            default_type_index = unique_types.index(default_type)
                        st.session_state["type_selectbox"] = unique_types[default_type_index]

                    selected_type = st.selectbox(
                        "Select a publication type", unique_types,
                        key="type_selectbox",
                    )

                    if selected_type:
                        st.query_params.from_dict({"type": selected_type})
                        encoded = selected_type.replace(" ", "+")
                        link    = f"https://intelligence.streamlit.app/?type={encoded}"
                        st.caption(f"🔗 Shareable link: [{link}]({link})")
                    else:
                        st.query_params.clear()

                    if not selected_type:
                        st.write("Pick a publication type to see items")
                        return

                    # Only runs after selected_type is confirmed
                    tdf = df_dedup[df_dedup["Publication type"] == selected_type].copy()
                    tdf["Date published"] = parse_date_column(tdf["Date published"])
                    tdf["Date published"] = tdf["Date published"].fillna("")
                    tdf = sort_by_date(tdf).sort_values(["No date flag","Date published"], ascending=[True,True])

                    with st.expander("Click to expand", expanded=True):
                        st.subheader(f"Publication type: {selected_type}", anchor=False, divider="blue")
                        if selected_type == "Thesis":
                            st.warning("Links to PhD theses may not work due to the [British Library cyber incident](https://www.bl.uk/cyber-incident/).")

                        ct1, ct2, ct3, ct4 = st.columns(4)
                        with ct1: c_m = st.container()
                        with ct2:
                            with st.popover("More metrics"):
                                c_cit      = st.container()
                                c_oa       = st.container()
                                c_collab   = st.container()
                                c_auth_no  = st.container()
                                c_auth_rat = st.container()
                        with ct3:
                            with st.popover("Relevant themes"):
                                st.markdown("##### Top relevant publication themes")
                                fdc  = pd.merge(df_duplicated, tdf[["Zotero link"]], on="Zotero link")
                                fdc  = fdc[["Zotero link","Collection_Key","Collection_Name","Collection_Link"]]
                                fdc2 = fdc["Collection_Name"].value_counts().reset_index().head(10)
                                fdc2.columns = ["Collection_Name","Number_of_Items"]
                                fdc  = pd.merge(fdc2, fdc, on="Collection_Name", how="left").drop_duplicates("Collection_Name").reset_index(drop=True)
                                fdc["Collection_Name"] = fdc["Collection_Name"].apply(remove_numbers)
                                for i, row in fdc.iterrows():
                                    st.caption(f"{i+1}) [{row['Collection_Name']}]({row['Collection_Link']}) {row['Number_of_Items']} items")
                        with ct4:
                            with st.popover("Filters and more"):
                                c_dl = st.container()
                                if selected_type == "Thesis":
                                    thesis_types = [""] + list(tdf["Thesis_type"].unique())
                                    sel_thesis   = st.selectbox("Select a thesis type", thesis_types)
                                    if sel_thesis:
                                        tdf = tdf[tdf["Thesis_type"] == sel_thesis]
                                    unis    = [""] + sorted(tdf["University"].astype(str).unique().tolist())
                                    sel_uni = st.selectbox("Select a university", unis)
                                    if sel_uni:
                                        tdf = tdf[tdf["University"] == sel_uni]
                                view = st.radio("View as:", ("Basic list","Table","Bibliography"), horizontal=True)

                        render_metrics(tdf, container_metric=c_m, container_citation=c_cit,
                                        container_citation_average=st.container(),
                                        container_oa=c_oa, container_author_no=c_auth_no,
                                        container_author_pub_ratio=c_auth_rat,
                                        container_publication_ratio=c_collab)

                        csv = convert_df_to_csv(
                            tdf[["Publication type","Title","Abstract","Date published",
                                    "Publisher","Journal","Link to publication","Zotero link","Citation"]]
                            .assign(Abstract=lambda d: d["Abstract"].str.replace("\n"," "))
                            .reset_index(drop=True)
                        )
                        c_dl.download_button(
                            "Download", csv,
                            f"{selected_type}_{datetime.date.today().isoformat()}.csv",
                            mime="text/csv", key="dl-type", icon=":material/download:",
                        )

                        on = st.toggle(":material/monitoring: Generate report")
                        if on and len(tdf):
                            st.info(f"Report for {selected_type}")
                            render_report_charts(tdf, selected_type, name_replacements,
                                                    show_themes=True, themes_df=fdc)
                        else:
                            tdf = sort_radio(tdf, key="type_sort")
                            if len(tdf) > 20 and st.checkbox("Show only first 20 items (untick to see all)", value=True):
                                tdf = tdf.head(20)
                            if view == "Basic list":
                                for i, row in tdf.iterrows():
                                    st.write(f"{i+1}) {format_entry(row, include_citation=True, reviews_map=reviews_map)}")
                            elif view == "Table":
                                st.dataframe(
                                    tdf[["Publication type","Title","Date published","FirstName2",
                                            "Abstract","Link to publication","Zotero link"]]
                                    .rename(columns={"FirstName2":"Author(s)","Link to publication":"Publication link"})
                                )
                            elif view == "Bibliography":
                                tdf["zotero_item_key"] = tdf["Zotero link"].str.replace(
                                    "https://www.zotero.org/groups/intelarchive_intelligence_studies_database/items/","")
                                df_zot = pd.read_csv("zotero_citation_format.csv")
                                display_bibliographies(pd.merge(tdf, df_zot, on="zotero_item_key", how="left"))

                type_selection()

            # ================================================================
            # 4 – JOURNAL SEARCH
            # ================================================================
            elif search_option == 4:
                st.subheader("Search journal", anchor=False, divider="blue")

                @st.fragment
                def search_journal():
                    df_ja   = df_dedup[df_dedup["Publication type"] == "Journal article"].copy()
                    jcounts = df_ja["Journal"].value_counts()
                    all_journals = jcounts.index.tolist()

                    # Pre-select from URL if ?journal= GUID is present
                    if "journal_selectbox" not in st.session_state:
                        default_guid = st.query_params.get("journal", "")
                        default_journal = guid_to_journal(default_guid, all_journals) if default_guid else ""
                        default_idx = (all_journals.index(default_journal) + 1) if default_journal in all_journals else 0
                        st.session_state["journal_selectbox"] = ([""] + all_journals)[default_idx]

                    selected_journal = st.selectbox(
                        "Select a journal",
                        [""] + all_journals,
                        key="journal_selectbox",
                    )

                    if selected_journal:
                        guid = journal_to_guid(selected_journal)
                        if st.query_params.get("journal", "") != guid:
                            st.query_params.from_dict({"journal": guid})
                        link = f"{BASE_URL}/?journal={guid}"
                        st.caption(f"🔗 Shareable link: [{link}]({link})")
                    else:
                        st.query_params.clear()

                    if not selected_journal:
                        st.write("Pick a journal name to see items")
                        return

                    journals = [selected_journal]

                    jdf = df_ja[df_ja["Journal"].isin(journals)].copy()
                    jdf["Date published"] = parse_date_column(jdf["Date published"])
                    jdf["Date published"] = jdf["Date published"].fillna("")
                    jdf = sort_by_date(jdf).sort_values(["No date flag","Date published"], ascending=[True,True])

                    with st.expander("Click to expand", expanded=True):
                        if len(journals) == 1:
                            st.markdown(f"#### Selected Journal: {journals[0]}")
                        else:
                            st.markdown("#### Selected Journals: " + ", ".join(journals))

                        cj1, cj2, cj3, cj4 = st.columns(4)
                        with cj1: c_m = st.container()
                        with cj2:
                            with st.popover("More metrics"):
                                c_cit      = st.container()
                                c_oa       = st.container()
                                c_collab   = st.container()
                                c_auth_no  = st.container()
                                c_auth_rat = st.container()
                                c_jcit_df  = st.container()
                        with cj3:
                            with st.popover("Relevant themes"):
                                st.markdown("##### Top 5 relevant themes")
                                fdc  = pd.merge(df_duplicated, jdf[["Zotero link"]], on="Zotero link")
                                fdc  = fdc[["Zotero link","Collection_Key","Collection_Name","Collection_Link"]]
                                fdc2 = fdc["Collection_Name"].value_counts().reset_index().head(10)
                                fdc2.columns = ["Collection_Name","Number_of_Items"]
                                fdc2 = fdc2[fdc2["Collection_Name"] != "01 Intelligence history"]
                                fdc  = pd.merge(fdc2, fdc, on="Collection_Name", how="left").drop_duplicates("Collection_Name").reset_index(drop=True)
                                fdc["Collection_Name"] = fdc["Collection_Name"].apply(remove_numbers)
                                for i, row in fdc.iterrows():
                                    st.caption(f"{i+1}) [{row['Collection_Name']}]({row['Collection_Link']}) {row['Number_of_Items']} items")
                        with cj4:
                            with st.popover("Filters and more"):
                                c_dl = st.container()
                                view = st.radio("View as:", ("Basic list","Table","Bibliography"), horizontal=True)

                        render_metrics(jdf, container_metric=c_m, container_citation=c_cit,
                                        container_citation_average=st.container(),
                                        container_oa=c_oa, container_author_no=c_auth_no,
                                        container_author_pub_ratio=c_auth_rat,
                                        container_publication_ratio=c_collab)

                        if len(journals) > 1:
                            c_jcit_df.dataframe(jdf.groupby("Journal")["Citation"].sum())

                        csv = convert_df_to_csv(
                            jdf[["Publication type","Title","Abstract","Date published",
                                    "Publisher","Journal","Link to publication","Zotero link","Citation"]]
                            .assign(Abstract=lambda d: d["Abstract"].str.replace("\n"," "))
                            .reset_index(drop=True)
                        )
                        c_dl.download_button(
                            "Download", csv,
                            f"selected_journal_{datetime.date.today().isoformat()}.csv",
                            mime="text/csv", key="dl-journal", icon=":material/download:",
                        )

                        on = st.toggle(":material/monitoring: Generate report")
                        if on and len(jdf):
                            st.info(f"Report for {journals}")
                            non_nan_id = jdf["ID"].count()
                            if non_nan_id != 0:
                                citation_count = jdf["Citation"].sum()
                                num_items      = len(jdf)
                                colcite1, colcite2, colcite3 = st.columns(3)
                                with colcite1:
                                    st.metric(label="Citation average", value=round(citation_count / num_items))
                                with colcite2:
                                    st.metric(label="Citation median", value=round(jdf["Citation"].median()))
                                with colcite3:
                                    year_diff_mean = jdf["Year_difference"].mean()
                                    st.metric(label="First citation occurence (avg year)",
                                                value=round(year_diff_mean) if not pd.isna(year_diff_mean) else "N/A")
                            render_report_charts(jdf, str(journals), name_replacements,
                                                    show_themes=True, themes_df=fdc)
                            jdf_copy = jdf.copy()
                            jdf_copy["Year"] = pd.to_datetime(jdf_copy["Date published"]).dt.year
                            pub_by_year = jdf_copy.groupby(["Year","Journal"]).size().unstack().fillna(0).cumsum()
                            st.plotly_chart(px.line(pub_by_year, x=pub_by_year.index, y=pub_by_year.columns,
                                                    title="Cumulative Publications Over Years"),
                                            use_container_width=True)
                            if len(journals) > 1:
                                jcit = jdf.groupby("Journal")["Citation"].sum().reset_index()
                                jcit = jcit[jcit["Citation"] > 0].sort_values("Citation", ascending=False)
                                st.plotly_chart(px.bar(jcit, x="Journal", y="Citation",
                                                        title="Citations per Journal"), use_container_width=True)
                        else:
                            jdf = sort_radio(jdf, key="journal_sort")
                            if len(jdf) > 20 and st.checkbox("Show only first 20 items (untick to see all)", value=True):
                                jdf = jdf.head(20)
                            if view == "Basic list":
                                for i, row in jdf.iterrows():
                                    st.write(f"{i+1}) {format_entry(row)}")
                            elif view == "Table":
                                st.dataframe(
                                    jdf[["Publication type","Title","Journal","Date published","FirstName2",
                                            "Abstract","Link to publication","Zotero link"]]
                                    .rename(columns={"FirstName2":"Author(s)","Link to publication":"Publication link"})
                                )
                            elif view == "Bibliography":
                                jdf["zotero_item_key"] = jdf["Zotero link"].str.replace(
                                    "https://www.zotero.org/groups/intelarchive_intelligence_studies_database/items/","")
                                df_zot = pd.read_csv("zotero_citation_format.csv")
                                display_bibliographies(pd.merge(jdf, df_zot, on="zotero_item_key", how="left"))

                search_journal()

            # ================================================================
            # 5 – PUBLICATION YEAR
            # ================================================================
            elif search_option == 5:
                st.subheader("Items by publication year", anchor=False, divider="blue")

                @st.fragment
                def search_pub_year():
                    reviews_map = load_reviews_map()
                    with st.expander("Click to expand", expanded=True):
                        df_all = df_dedup.copy()
                        df_all["Date published"] = parse_date_column(df_all["Date published"])
                        df_all["Date year"]      = pd.to_numeric(df_all["Date published"].str[:4], errors="coerce")
                        numeric_years = df_all["Date year"].dropna()
                        min_y, max_y  = int(numeric_years.min()), int(numeric_years.max())
                        df_all["Date published"] = df_all["Date published"].fillna("")
                        df_all = sort_by_date(df_all).sort_values("Date published", ascending=False)

                        if "year_slider" not in st.session_state:
                            current_year = date.today().year
                            default_from = int(st.query_params.get("year_from", current_year))
                            default_to   = int(st.query_params.get("year_to",   current_year + 1))
                            default_from = max(min_y, min(default_from, max_y))
                            default_to   = max(min_y, min(default_to,   max_y))
                            st.session_state["year_slider"] = (default_from, default_to)

                        years = st.slider(
                            "Publication years between:", min_y, max_y,
                            key="year_slider",
                        )
                        st.query_params.from_dict({"year_from": str(years[0]), "year_to": str(years[1])})
                        link = f"https://intelligence.streamlit.app/?year_from={years[0]}&year_to={years[1]}"
                        st.caption(f"🔗 Shareable link: [{link}]({link})")
                        df_all = df_all[(df_all["Date year"] >= years[0]) & (df_all["Date year"] <= years[1])]

                        cy1, cy2, cy3, cy4 = st.columns(4)
                        with cy1: c_m = st.container()
                        with cy2:
                            with st.popover("More metrics"):
                                c_cit      = st.container()
                                c_cit_avg  = st.container()
                                c_oa       = st.container()
                                c_type     = st.container()
                                c_auth_no  = st.container()
                                c_auth_rat = st.container()
                                c_collab   = st.container()
                        with cy3:
                            with st.popover("Relevant themes"):
                                st.markdown("##### Top relevant themes")
                                c_themes = st.container()
                        with cy4:
                            with st.popover("Filters and more"):
                                st.warning("Items without a publication date are not listed here!")
                                sel_types = st.multiselect("Filter by publication type:", df_all["Publication type"].unique())
                                if sel_types:
                                    df_all = df_all[df_all["Publication type"].isin(sel_types)]
                                df_all = df_all.reset_index(drop=True)
                                c_dl = st.container()
                                view = st.radio("View as:", ("Basic list","Table","Bibliography"), horizontal=True)

                        render_metrics(df_all, container_metric=c_m, container_citation=c_cit,
                                        container_citation_average=c_cit_avg, container_oa=c_oa,
                                        container_type=c_type, container_author_no=c_auth_no,
                                        container_author_pub_ratio=c_auth_rat,
                                        container_publication_ratio=c_collab,
                                        label=f"#Sources {years[0]}-{years[1]}")

                        fdc  = pd.merge(df_duplicated, df_all[["Zotero link"]], on="Zotero link")
                        fdc  = fdc[["Zotero link","Collection_Key","Collection_Name","Collection_Link"]]
                        fdc2 = fdc["Collection_Name"].value_counts().reset_index().head(10)
                        fdc2.columns = ["Collection_Name","Number_of_Items"]
                        fdc2 = fdc2[fdc2["Collection_Name"] != "01 Intelligence history"]
                        fdc  = pd.merge(fdc2, fdc, on="Collection_Name", how="left").drop_duplicates("Collection_Name").reset_index(drop=True)
                        fdc["Collection_Name"] = fdc["Collection_Name"].apply(remove_numbers)
                        for i, row in fdc.iterrows():
                            c_themes.caption(f"{i+1}) [{row['Collection_Name']}]({row['Collection_Link']}) {row['Number_of_Items']} items")

                        csv = convert_df_to_csv(
                            df_all[["Publication type","Title","Abstract","FirstName2",
                                    "Link to publication","Zotero link","Date published","Citation"]]
                            .rename(columns={"FirstName2":"Author(s)"})
                            .assign(Abstract=lambda d: d["Abstract"].str.replace("\n"," "))
                        )
                        label_str = f"{years[0]}-{years[1]}"
                        c_dl.download_button(
                            "Download selected items", csv,
                            f"intelligence-bibliography-items-between-{label_str}.csv",
                            mime="text/csv", key="dl-year", icon=":material/download:",
                        )

                        on = st.toggle(":material/monitoring: Generate report", key="year_report")

                        current_url_report = st.query_params.get("report", "0") == "1"
                        if on != current_url_report:
                            params = {"year_from": str(years[0]), "year_to": str(years[1])}
                            if on:
                                params["report"] = "1"
                            st.query_params.from_dict(params)

                        link = f"https://intelligence.streamlit.app/?year_from={years[0]}&year_to={years[1]}{'&report=1' if on else ''}"
                        st.caption(f"🔗 Shareable link: [{link}]({link})")

                        if on and len(df_all):
                            st.info(f"Report for {label_str}")
                            render_report_charts(df_all, label_str, name_replacements,
                                                    show_themes=True, themes_df=fdc)
                        else:
                            df_all = sort_radio(df_all, key="year_sort")
                            if len(df_all) > 20 and st.checkbox("Show only first 20 items (untick to see all)", value=True, key="all_items"):
                                df_all = df_all.head(20)
                            if view == "Basic list":
                                articles  = [format_entry(row, include_citation=True, reviews_map=reviews_map) for _, row in df_all.iterrows()]
                                abstracts = [row["Abstract"] if pd.notnull(row["Abstract"]) else "N/A" for _, row in df_all.iterrows()]
                                render_paginated_list(df_all, articles, abstracts)
                            elif view == "Table":
                                st.dataframe(
                                    df_all[["Publication type","Title","Date published","FirstName2",
                                            "Abstract","Link to publication","Zotero link"]]
                                    .rename(columns={"FirstName2":"Author(s)","Link to publication":"Publication link"})
                                )
                            elif view == "Bibliography":
                                df_all["zotero_item_key"] = df_all["Zotero link"].str.replace(
                                    "https://www.zotero.org/groups/intelarchive_intelligence_studies_database/items/","")
                                df_zot = pd.read_csv("zotero_citation_format.csv")
                                display_bibliographies(pd.merge(df_all, df_zot, on="zotero_item_key", how="left"))

                search_pub_year()

            # ================================================================
            # 6 – CITED PAPERS
            # ================================================================
            elif search_option == 6:
                st.query_params.clear()
                st.subheader("Cited items in the library", anchor=False, divider="blue")

                @st.fragment
                def search_cited_papers():
                    reviews_map = load_reviews_map()
                    with st.expander("Click to expand", expanded=True):
                        c_md              = st.container()
                        df_cited          = df_dedup[df_dedup["Citation"].notna()].copy().reset_index(drop=True)
                        df_cited_for_mean = df_dedup.copy()
                        non_nan_id        = df_dedup["ID"].count()

                        cc1, cc2, cc3 = st.columns(3)
                        with cc1: c_m = st.container()
                        with cc2:
                            with st.popover("More metrics"):
                                c_cit      = st.container()
                                c_cit_avg  = st.container()
                                c_oa       = st.container()
                                c_auth_no  = st.container()
                                c_auth_rat = st.container()
                                c_collab   = st.container()
                        with cc3:
                            with st.popover("Filters and more"):
                                st.warning("Citation data from [OpenAlex](https://openalex.org/).")
                                citation_type = st.radio(
                                    "Select:", ("All citations","Trends","Citations without outliers"), horizontal=True,
                                )
                                c_slider = st.container()
                                c_dl     = st.container()
                                view     = st.radio("View as:", ("Basic list","Table","Bibliography"), horizontal=True)

                        c_md.markdown(f"#### {citation_type}")
                        current_year = datetime.datetime.now().year

                        if citation_type == "Trends":
                            df_cited = df_cited[
                                (df_cited["Last_citation_year"].isin([current_year, current_year-1])) &
                                (df_cited["Publication_year"].isin([current_year, current_year-1]))
                            ]
                        elif citation_type == "Citations without outliers":
                            df_cited          = df_cited[df_cited["Citation"] < 1000]
                            df_cited_for_mean = df_cited_for_mean[df_cited_for_mean["Citation"] < 1000]

                        max_cit   = int(df_cited["Citation"].max()) if len(df_cited) else 1
                        sel_range = c_slider.slider("Select a citation range:", 1, max_cit, (1, max_cit))
                        df_cited  = df_cited[(df_cited["Citation"] >= sel_range[0]) & (df_cited["Citation"] <= sel_range[1])]

                        df_cited["Date published"] = parse_date_column(df_cited["Date published"])
                        df_cited["Date published"] = df_cited["Date published"].fillna("")
                        df_cited = sort_by_date(df_cited).sort_values("Date published", ascending=False).reset_index(drop=True)

                        render_metrics(df_cited, container_metric=c_m, container_citation=c_cit,
                                        container_citation_average=c_cit_avg, container_oa=c_oa,
                                        container_author_no=c_auth_no, container_author_pub_ratio=c_auth_rat,
                                        container_publication_ratio=c_collab,
                                        label="Number of cited publications")

                        if citation_type == "Trends":
                            st.info(f"Shows citations in {current_year-1}-{current_year} to papers from the same period.")
                        elif citation_type == "Citations without outliers":
                            outlier_count = int((df_dedup["Citation"] > 1000).sum())
                            st.info(f"**{outlier_count}** items with >1000 citations are excluded.")

                        csv = convert_df_to_csv(
                            df_cited[["Publication type","Title","Abstract","FirstName2",
                                        "Link to publication","Zotero link","Date published","Citation"]]
                            .rename(columns={"FirstName2":"Author(s)"})
                            .assign(Abstract=lambda d: d["Abstract"].str.replace("\n"," "))
                        )
                        c_dl.download_button(
                            "Download selected items", csv, "cited-items.csv",
                            mime="text/csv", key="dl-cited", icon=":material/download:",
                        )

                        on = st.toggle(":material/monitoring: Generate report")
                        if on and len(df_cited):
                            st.markdown("#### Report for cited items in the library")
                            non_nan_c       = df_cited.dropna(subset=["Citation_list"])
                            citation_mean   = non_nan_c["Citation"].mean()
                            citation_median = non_nan_c["Citation"].median()
                            colcite1, colcite2, colcite3 = st.columns(3)
                            with colcite1:
                                st.metric(label="Citation average", value=round(citation_mean, 2))
                            with colcite2:
                                st.metric(label="Citation median", value=round(citation_median, 2))
                            with colcite3:
                                mean_first = df_cited["Year_difference"].mean()
                                st.metric(label="First citation occurence (avg year)", value=round(mean_first))

                            citation_dist = df_cited["Citation"].value_counts().sort_index().reset_index()
                            citation_dist.columns = ["Number of Citations","Number of Articles"]
                            fig = px.scatter(citation_dist, x="Number of Citations", y="Number of Articles",
                                                title="Distribution of Citations Across Articles")
                            fig.update_traces(marker=dict(color="red", size=7, opacity=0.5))
                            st.plotly_chart(fig)

                            fig2 = go.Figure(data=go.Scatter(
                                x=df_cited["Year_difference"], y=[0]*len(df_cited), mode="markers"))
                            fig2.update_layout(title="First citation occurence (years after publication)",
                                                xaxis_title="Year Difference", yaxis_title="")
                            st.plotly_chart(fig2)

                            render_report_charts(df_cited, "cited items", name_replacements)
                        else:
                            df_cited = sort_radio(df_cited, key="cited_sort")
                            if len(df_cited) > 20 and st.checkbox("Show only first 20 items (untick to see all)", value=True, key="all_items"):
                                df_cited = df_cited.head(20)
                            if view == "Basic list":
                                for i, row in df_cited.iterrows():
                                    st.markdown(f"{i+1}. {format_entry(row, include_citation=True, reviews_map=reviews_map)}", unsafe_allow_html=True)
                            elif view == "Table":
                                st.dataframe(
                                    df_cited[["Publication type","Title","Date published","FirstName2",
                                                "Abstract","Journal","Link to publication","Zotero link","Citation"]]
                                    .rename(columns={"FirstName2":"Author(s)","Link to publication":"Publication link"})
                                )
                            elif view == "Bibliography":
                                df_cited["zotero_item_key"] = df_cited["Zotero link"].str.replace(
                                    "https://www.zotero.org/groups/intelarchive_intelligence_studies_database/items/","")
                                df_zot = pd.read_csv("zotero_citation_format.csv")
                                display_bibliographies(pd.merge(df_cited, df_zot, on="zotero_item_key", how="left"))

                search_cited_papers()

            # ── Overview ────────────────────────────────────────────────────
            st.header("Overview", anchor=False)

            @st.fragment
            def overview():
                tab11, tab12, tab13 = st.tabs(["Recently added items","Recently published items","Top cited items"])

                with tab11:
                    st.markdown("#### Recently added or updated items")
                    reviews_map = load_reviews_map()
                    df_ov = df_dedup.sort_values("Date added", ascending=False).head(10).copy()
                    df_ov["Date published"] = parse_date_column(df_ov["Date published"], fmt="%d-%m-%Y")
                    df_ov["Date published"] = df_ov["Date published"].fillna("No date")
                    df_ov["Abstract"]       = df_ov["Abstract"].fillna("No abstract")
                    display = st.checkbox("Display abstract")
                    for i, (_, row) in enumerate(df_ov.iterrows(), 1):
                        st.markdown(f"{i}) {format_entry(row, include_citation=True, reviews_map=reviews_map)}", unsafe_allow_html=True)
                        if display and row["Abstract"]:
                            st.markdown(f"**Abstract:** {row['Abstract']}")

                with tab12:
                    st.markdown("#### Recently published items")
                    display2 = st.checkbox("Display abstracts", key="recently_published")
                    df_ov2   = df_dedup.copy()
                    df_ov2["Date published"] = pd.to_datetime(df_ov2["Date published"], utc=True, errors="coerce").dt.tz_convert("Europe/London")
                    now      = datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=1)))
                    df_ov2   = df_ov2[df_ov2["Date published"] <= now]
                    df_ov2["Date published"] = df_ov2["Date published"].dt.strftime("%Y-%m-%d").fillna("")
                    df_ov2   = df_ov2.sort_values("Date published", ascending=False).head(10).reset_index(drop=True)
                    for i, row in df_ov2.iterrows():
                        st.write(f"{i+1}) {format_entry(row, include_citation=True)}")
                        if display2:
                            st.caption(row["Abstract"])

                with tab13:
                    st.markdown("#### Top cited items")
                    display3 = st.checkbox("Display abstracts", key="top_cited")

                    @st.cache_resource(ttl=5000)
                    def _top_cited():
                        df_t = df_dedup.copy()
                        df_t["Date published"] = parse_date_column(df_t["Date published"])
                        return df_t.sort_values("Citation", ascending=False).reset_index(drop=True)

                    df_top = _top_cited().head(10)
                    for i, row in df_top.iterrows():
                        st.write(f"{i+1}) {format_entry(row)}")
                        if display3:
                            st.caption(row["Abstract"])

            overview()

            # ── All items ───────────────────────────────────────────────────
            st.header("All items in database", anchor=False)
            with st.expander("Click to expand", expanded=False):
                st.write("""
                The entire dataset is available on Zenodo (updated quarterly):

                Ozkan, Yusuf A. 'Intelligence Studies Network Dataset'. Zenodo, 15 August 2024.
                https://doi.org/10.5281/zenodo.13325698.
                """)
                df_added = df_dedup.copy()
                df_added["Date added"]  = pd.to_datetime(df_added["Date added"])
                df_added["YearMonth"]   = df_added["Date added"].dt.to_period("M").astype(str)
                monthly    = df_added.groupby("YearMonth").size().rename("Number of items added")
                cumulative = monthly.cumsum()
                chart = (
                    alt.Chart(pd.DataFrame({"YearMonth": cumulative.index, "Total items": cumulative}))
                    .mark_bar()
                    .encode(x="YearMonth", y="Total items", tooltip=["YearMonth","Total items"])
                    .properties(width=500, height=600, title="Total Number of Items Added")
                )
                st.subheader("Growth of the library", anchor=False, divider="blue")
                st.altair_chart(chart, use_container_width=True)

        search_options_main_menu()
    # ── Right sidebar col ───────────────────────────────────────────────
    with col2:
        st.info("Join the [mailing list](https://groups.google.com/g/intelarchive)")

        @st.fragment
        def collection_buttons():
            SIDEBAR_COLLECTIONS = [
                ("Intelligence history",              "01_CONTAINER"),
                ("Intelligence studies",              "HCN8YFI8"),
                ("Intelligence analysis",             "CZJ36V8L"),
                ("Intelligence organisations",        "CK5MNYPQ"),
                ("Intelligence failures",             "D7XFV7JL"),
                ("Accountability, oversight, ethics", "DVEM4H4W"),
                ("Intelligence collection",           "07_CONTAINER"),
                ("Counterintelligence",               "RHJFPRAI"),
                ("Covert action",                     "B6RJNLTK"),
                ("Intelligence and cybersphere",      "8XXD789V"),
                ("Global intelligence",               "AZ3BZ9BR"),
                ("Special collections",               "98_CONTAINER"),
            ]
            with st.expander("Collections", expanded=True):
                for label, key in SIDEBAR_COLLECTIONS:
                    if st.button(label):
                        st.query_params.from_dict({"collection": key})
                        st.rerun()

        collection_buttons()

        with st.expander("Events & conferences", expanded=True):
            for info in evens_conferences():
                st.write(info)

        with st.expander("Digest", expanded=True):
            st.write("See our dynamic [digest](https://intelligence.streamlit.app/Digest) for the latest updates!")

# ════════════════════════════════════════════════════════════════════════
# TAB 2 – DASHBOARD
# ════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("Dashboard", anchor=False)
    on_main_dashboard = st.toggle(":material/dashboard: Display dashboard")

    if on_main_dashboard:
        df_csv           = df_duplicated.copy()
        df_collections_2 = df_csv.copy()
        df_csv           = df_dedup.copy().reset_index(drop=True)

        df_csv["Date published"] = parse_date_column(df_csv["Date published"])
        df_csv["Date year"]      = df_csv["Date published"].str[:4].fillna("No date")

        df_year = df_csv["Date year"].value_counts().reset_index()
        df_year.columns = ["Publication year","Count"]
        df_year.drop(df_year[df_year["Publication year"] == "No date"].index, inplace=True)
        df_year = df_year.sort_values("Publication year").reset_index(drop=True)
        max_y   = int(df_year["Publication year"].max())
        min_y   = int(df_year["Publication year"].min())

        df_collections_2["Date published"] = parse_date_column(df_collections_2["Date published"])
        df_collections_2["Date year"]      = df_collections_2["Date published"].str[:4].fillna("No date")

        with st.expander("**Select filters**", expanded=False):
            types = st.multiselect("Publication type",
                                    df_csv["Publication type"].unique(),
                                    df_csv["Publication type"].unique())
            df_journals_dash = df_dedup[df_dedup["Publication type"] == "Journal article"]
            journals_dash    = st.multiselect("Select a journal",
                                                df_journals_dash["Journal"].value_counts().index.tolist(),
                                                key="big_dashboard_journals")
            years = st.slider("Publication years between:", min_y, max_y + 1, (min_y, max_y + 1), key="years2")

            if st.button("Update dashboard"):
                df_csv = df_csv[df_csv["Publication type"].isin(types)]
                if journals_dash:
                    df_csv = df_csv[df_csv["Journal"].isin(journals_dash)]
                df_csv = df_csv[df_csv["Date year"] != "No date"]
                df_csv = df_csv[(df_csv["Date year"].astype(int) >= years[0]) &
                                (df_csv["Date year"].astype(int) < years[1])]
                df_year = df_csv["Date year"].value_counts().reset_index()
                df_year.columns = ["Publication year","Count"]
                df_year.drop(df_year[df_year["Publication year"] == "No date"].index, inplace=True)
                df_year = df_year.sort_values("Publication year").reset_index(drop=True)

                df_collections_2 = df_collections_2[df_collections_2["Publication type"].isin(types)]
                if journals_dash:
                    df_collections_2 = df_collections_2[df_collections_2["Journal"].isin(journals_dash)]
                df_collections_2 = df_collections_2[df_collections_2["Date year"] != "No date"]
                df_collections_2 = df_collections_2[
                    (df_collections_2["Date year"].astype(int) >= years[0]) &
                    (df_collections_2["Date year"].astype(int) < years[1])
                ]

        if not df_csv["Title"].any():
            st.warning("No data to visualise. Select a correct parameter.")
        else:
            # ── Collections ────────────────────────────────────────────
            st.subheader("Publications by collection", anchor=False, divider="blue")

            @st.fragment
            def collection_chart():
                df_col21 = df_collections_2["Collection_Name"].value_counts().reset_index()
                df_col21.columns = ["Collection_Name","Number_of_Items"]

                col1, col2 = st.columns(2)
                with col1:
                    colallcol1, colallcol2 = st.columns([2, 3])
                    with colallcol1:
                        show_legend = st.checkbox("Show legend", key="collection_bar_legend_check")
                        last_5_col  = st.checkbox("Limit to last 5 years", key="last5yearscollections")
                        if last_5_col:
                            df_col21 = df_collections_2[df_collections_2["Date year"] != "No date"].copy()
                            df_col21["Date year"] = df_col21["Date year"].astype(int)
                            df_col21 = df_col21[df_col21["Date year"] > (datetime.datetime.now().year - 5)]
                            df_col21 = df_col21["Collection_Name"].value_counts().reset_index()
                            df_col21.columns = ["Collection_Name","Number_of_Items"]
                    with colallcol2:
                        number0 = st.slider("Select a number of collections", 3, len(df_col21), 10, key="slider01")
                    plot = df_col21.head(number0 + 1)
                    plot = plot[plot["Collection_Name"] != "01 Intelligence history"]
                    fig  = px.bar(plot, x="Collection_Name", y="Number_of_Items", color="Collection_Name")
                    fig.update_xaxes(tickangle=-65)
                    fig.update_traces(width=0.6)
                    fig.update_layout(autosize=False, width=600, height=600, showlegend=show_legend,
                                        title=f"Top {number0} collections in the library")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    colcum1, colcum2, colcum3 = st.columns(3)
                    with colcum1: hide_legend = st.checkbox("Hide legend", key="collection_line_legend_check")
                    with colcum2: last_5_cum  = st.checkbox("Limit to last 5 years", key="last5yearscollectioncummulative")
                    with colcum3: top_5_only  = st.checkbox("Show top 5 collections", key="top5collections")

                    df_col22 = df_collections_2.copy()
                    if last_5_cum:
                        df_col22 = df_col22[df_col22["Date year"] != "No date"].copy()
                        df_col22["Date year"] = df_col22["Date year"].astype(int)
                        df_col22 = df_col22[df_col22["Date year"] > (datetime.datetime.now().year - 5)]

                    col_counts = df_col22.groupby(["Date year","Collection_Name"]).size().unstack().fillna(0)
                    col_counts = col_counts.reset_index()
                    col_counts.iloc[:, 1:] = col_counts.iloc[:, 1:].cumsum()
                    top_cols   = df_col22["Collection_Name"].value_counts().head(5).index.tolist() if top_5_only \
                                    else df_col22["Collection_Name"].unique().tolist()
                    col_filt   = col_counts[["Date year"] + top_cols]
                    col_filt["Date year"] = pd.to_numeric(col_filt["Date year"], errors="coerce")
                    col_filt   = col_filt.sort_values("Date year")
                    fig = px.line(col_filt, x="Date year", y=top_cols, markers=True,
                                    title="Cumulative changes in collection over years")
                    fig.update_layout(showlegend=not hide_legend)
                    st.plotly_chart(fig, use_container_width=True)

            collection_chart()

            st.divider()
            st.subheader("Publications by type and year", anchor=False, divider="blue")

            @st.fragment
            def types_pubyears():
                df_types   = df_csv["Publication type"].value_counts().reset_index()
                df_types.columns = ["Publication type","Count"]
                chart_type = st.radio("Choose visual type", ["Bar chart","Pie chart"], horizontal=True)

                col1, col2 = st.columns(2)
                with col1:
                    coltype1, coltype2 = st.columns(2)
                    with coltype1:
                        last_5_types = st.checkbox("Limit to last 5 years", key="last5yearsitemtypes")
                    with coltype2:
                        log0 = st.checkbox("Show in log scale", key="log0", disabled=(chart_type == "Pie chart"))
                    if last_5_types:
                        df_csv2 = df_csv[df_csv["Date year"] != "No date"].copy()
                        df_csv2["Date year"] = df_csv2["Date year"].astype(int)
                        df_csv2 = df_csv2[df_csv2["Date year"] > (datetime.datetime.now().year - 5)]
                        df_types = df_csv2["Publication type"].value_counts().reset_index()
                        df_types.columns = ["Publication type","Count"]
                    if chart_type == "Bar chart":
                        fig = px.bar(df_types, x="Publication type", y="Count", color="Publication type",
                                        log_y=log0, title="Item types" + (" (log scale)" if log0 else ""))
                        fig.update_traces(width=0.6)
                        fig.update_xaxes(tickangle=-70)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.pie(df_types, values="Count", names="Publication type", title="Item types")
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    coly1, coly2 = st.columns(2)
                    df_year_dash = df_year.copy()
                    df_year_dash["Publication year"] = df_year_dash["Publication year"].astype(int)
                    with coly1:
                        last_10 = st.checkbox("Limit to last 10 years", value=False)
                        if last_10:
                            cur = datetime.datetime.now().year
                            df_year_dash = df_year_dash[df_year_dash["Publication year"] >= cur - 9]
                    with coly2:
                        min_yr = int(df_year_dash["Publication year"].min())
                        max_yr = int(df_year_dash["Publication year"].max())
                        if min_yr == max_yr:
                            st.warning(f"All publications are from {min_yr}.")
                            yr_range = (min_yr, max_yr)
                        else:
                            yr_range = st.slider("Publication years:", min_yr, max_yr, (min_yr, max_yr), key="years3")
                    df_year_dash = df_year_dash[(df_year_dash["Publication year"] >= yr_range[0]) &
                                                (df_year_dash["Publication year"] <= yr_range[1])]
                    if not df_year_dash.empty:
                        fig = px.bar(df_year_dash, x="Publication year", y="Count",
                                        title=f"All items by publication year {yr_range[0]}-{yr_range[1]}")
                        fig.update_xaxes(tickangle=-70, type="category")
                        st.plotly_chart(fig, use_container_width=True)

            types_pubyears()

            st.divider()
            st.subheader("Publications by author", anchor=False, divider="blue")

            @st.fragment
            def author_chart():
                df_auth  = df_csv.copy()
                df_auth2 = df_csv.copy()
                num_authors = st.slider("Select number of authors to display:", 5, 30, 20, key="author2")

                col1, col2 = st.columns(2)
                with col1:
                    colauth1, colauth2 = st.columns(2)
                    with colauth1:
                        table_view = st.radio("Choose visual type", ["Bar chart","Table view"], key="author", horizontal=True)
                    with colauth2:
                        last_5_auth = st.checkbox("Limit to last 5 years", key="last5yearsauthorsall")
                    if last_5_auth:
                        df_auth = df_csv[df_csv["Date year"] != "No date"].copy()
                        df_auth["Date year"] = df_auth["Date year"].astype(int)
                        df_auth = df_auth[df_auth["Date year"] > (datetime.datetime.now().year - 5)]
                    df_auth_exp = df_auth.copy()
                    df_auth_exp["Author_name"] = df_auth_exp["FirstName2"].apply(
                        lambda x: x.split(", ") if isinstance(x, str) and x else [])
                    df_auth_exp = df_auth_exp.explode("Author_name")
                    df_auth_exp["Author_name"] = df_auth_exp["Author_name"].map(name_replacements).fillna(df_auth_exp["Author_name"])
                    df_auth_exp = df_auth_exp[df_auth_exp["Author_name"] != "nan"]
                    top_auth    = df_auth_exp["Author_name"].value_counts().head(num_authors).reset_index()
                    top_auth.columns = ["Author","Number of Publications"]
                    if table_view == "Bar chart":
                        fig = px.bar(top_auth, x="Author", y="Number of Publications",
                                        title=f"Top {num_authors} Authors (all items)")
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig)
                    else:
                        st.markdown(f"###### Top {num_authors} Authors (all items)")
                        st.dataframe(top_auth.rename(columns={"Author":"Author name","Number of Publications":"Publication count"}))

                with col2:
                    colauth11, colauth12 = st.columns(2)
                    with colauth11:
                        sel_type_auth = st.radio("Select a publication type",
                                                    ["Journal article","Book","Book chapter"], horizontal=True)
                    with colauth12:
                        last_5_auth2 = st.checkbox("Limit to last 5 years", key="last5yearsauthorsallspecified")
                    df_auth_t = df_csv[df_csv["Publication type"] == sel_type_auth].copy()
                    if last_5_auth2:
                        df_auth_t = df_auth_t[df_auth_t["Date year"] != "No date"]
                        df_auth_t["Date year"] = df_auth_t["Date year"].astype(int)
                        df_auth_t = df_auth_t[df_auth_t["Date year"] > (datetime.datetime.now().year - 5)]
                    if len(df_auth_t):
                        df_auth_t["Author_name"] = df_auth_t["FirstName2"].apply(
                            lambda x: x.split(", ") if isinstance(x, str) and x else [])
                        df_auth_t = df_auth_t.explode("Author_name")
                        df_auth_t["Author_name"] = df_auth_t["Author_name"].map(name_replacements).fillna(df_auth_t["Author_name"])
                        df_auth_t = df_auth_t[df_auth_t["Author_name"] != "nan"]
                        top_t = df_auth_t["Author_name"].value_counts().head(num_authors).reset_index()
                        top_t.columns = ["Author","Number of Publications"]
                        if table_view == "Bar chart":
                            fig = px.bar(top_t, x="Author", y="Number of Publications",
                                            title=f"Top {num_authors} Authors ({sel_type_auth})")
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig)
                        else:
                            st.markdown(f"###### Top {num_authors} Authors ({sel_type_auth})")
                            st.dataframe(top_t.rename(columns={"Author":"Author name","Number of Publications":"Publication count"}))
                    else:
                        st.write("No data to visualize")

                st.markdown("##### Single vs Multiple authored publications",
                            help="Theses excluded as they are inherently single-authored.")
                col1, col2 = st.columns([3, 1])
                with col1:
                    df_auth2["multiple_authors"] = df_auth2["FirstName2"].apply(
                        lambda x: isinstance(x, str) and "," in x)
                    df_auth3  = df_auth2[df_auth2["Publication type"] != "Thesis"].copy()
                    grouped3  = df_auth3.groupby("Date year")
                    total3    = grouped3.size().reset_index(name="Total Publications")
                    multi3    = grouped3["multiple_authors"].apply(lambda x: (x==True).sum()).reset_index(name="# Multiple Authored Publications")
                    df_multi3 = pd.merge(total3, multi3, on="Date year")
                    df_multi3["# Single Authored Publications"] = df_multi3["Total Publications"] - df_multi3["# Multiple Authored Publications"]

                    df_auth2["Date year"] = pd.to_numeric(df_auth2["Date year"], errors="coerce")
                    grouped   = df_auth2.groupby("Date year")
                    total     = grouped.size().reset_index(name="Total Publications")
                    multi     = grouped["multiple_authors"].apply(lambda x: (x==True).sum()).reset_index(name="# Multiple Authored Publications")
                    df_multi  = pd.merge(total, multi, on="Date year")
                    df_multi["# Single Authored Publications"]      = df_multi["Total Publications"] - df_multi["# Multiple Authored Publications"]
                    df_multi["% Multiple Authored Publications"]    = round(df_multi["# Multiple Authored Publications"] / df_multi["Total Publications"], 3) * 100
                    df_multi["% Single Authored Publications"]      = 100 - df_multi["% Multiple Authored Publications"]
                    df_multi  = df_multi[df_multi["Date year"] <= datetime.datetime.now().year]
                    last_20   = df_multi[df_multi["Date year"] >= (df_multi["Date year"].max() - 20)]

                    see_number = st.toggle("See number of publications")
                    fig1 = go.Figure()
                    fig1.add_trace(go.Scatter(x=last_20["Date year"], y=last_20["# Multiple Authored Publications"],
                                                mode="lines+markers", name="# Multiple Authored", line=dict(color="goldenrod")))
                    fig1.add_trace(go.Scatter(x=last_20["Date year"], y=last_20["# Single Authored Publications"],
                                                mode="lines+markers", name="# Single Authored", line=dict(color="green")))
                    fig1.update_layout(title="# Single vs Multiple Authored Publications",
                                        xaxis_title="Year", yaxis_title="Number of Publications")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=last_20["Date year"], y=last_20["% Multiple Authored Publications"],
                                                mode="lines+markers", name="% Multiple Authored", line=dict(color="goldenrod")))
                    fig2.add_trace(go.Scatter(x=last_20["Date year"], y=last_20["% Single Authored Publications"],
                                                mode="lines+markers", name="% Single Authored", line=dict(color="green")))
                    fig2.update_layout(title="% Single vs Multiple Authored Publications",
                                        xaxis_title="Publication Year", yaxis_title="% Publications")
                    st.plotly_chart(fig1 if see_number else fig2, use_container_width=True)

                with col2:
                    last_5_pie = st.checkbox("Limit to last 5 years", key="last5yearsauthor")
                    df_pie     = df_multi3.copy()
                    if last_5_pie:
                        df_pie = df_pie[df_pie["Date year"] >= (df_pie["Date year"].max() - 5)]
                    fig = px.pie(
                        values=[df_pie["# Multiple Authored Publications"].sum(),
                                df_pie["# Single Authored Publications"].sum()],
                        names=["Multiple Authored","Single Authored"],
                        title="Single vs Multiple Authored Papers",
                        color_discrete_sequence=["goldenrod","green"],
                    )
                    st.plotly_chart(fig)

            author_chart()

            st.divider()
            st.subheader("Publishers and Journals", anchor=False, divider="blue")
            col1, col2 = st.columns(2)

            with col1:
                @st.fragment
                def publisher_chart():
                    number  = st.slider("Select a number of publishers", 0, 30, 10)
                    df_pub  = df_csv["Publisher"].value_counts().reset_index()
                    df_pub.columns = ["Publisher","Count"]
                    df_pub  = df_pub.sort_values("Count", ascending=False).head(number)
                    log1    = st.checkbox("Show in log scale", key="log1")
                    leg1    = st.checkbox("Disable legend", key="leg1")
                    tbl_pub = st.checkbox("Table view")
                    if tbl_pub:
                        st.dataframe(df_pub)
                    elif not df_pub.empty:
                        fig = px.bar(df_pub, x="Publisher", y="Count", color="Publisher", log_y=log1,
                                        title=f"Top {number} publishers" + (" (log scale)" if log1 else ""))
                        fig.update_traces(width=0.6)
                        fig.update_layout(autosize=False, width=1200, height=700, showlegend=not leg1)
                        fig.update_xaxes(tickangle=-70)
                        st.plotly_chart(fig, use_container_width=True)
                publisher_chart()

            with col2:
                @st.fragment
                def journal_chart():
                    number2  = st.slider("Select a number of journals", 0, 30, 10)
                    df_jour  = df_csv[df_csv["Publication type"] == "Journal article"]["Journal"].value_counts().reset_index()
                    df_jour.columns = ["Journal","Count"]
                    df_jour  = df_jour.sort_values("Count", ascending=False).head(number2)
                    log2     = st.checkbox("Show in log scale", key="log2")
                    leg2     = st.checkbox("Disable legend", key="leg2")
                    tbl_jour = st.checkbox("Table view", key="journal")
                    if tbl_jour:
                        st.dataframe(df_jour)
                    elif not df_jour.empty:
                        fig = px.bar(df_jour, x="Journal", y="Count", color="Journal", log_y=log2,
                                        title=f"Top {number2} journals" + (" (log scale)" if log2 else ""))
                        fig.update_traces(width=0.6)
                        fig.update_layout(autosize=False, width=1200, height=700, showlegend=not leg2)
                        fig.update_xaxes(tickangle=-70)
                        st.plotly_chart(fig, use_container_width=True)
                journal_chart()

            st.divider()
            st.subheader("Publications by open access status", anchor=False, divider="blue")

            df_dedup_oa = df_collections_2.drop_duplicates(subset="Zotero link").copy()
            df_dedup_oa["Date year"] = pd.to_numeric(df_dedup_oa["Date year"], errors="coerce")
            df_dedup_v2 = df_dedup_oa.dropna(subset=["OA status"]).copy()
            df_dedup_v2["Citation status"] = df_dedup_v2["Citation"].apply(
                lambda x: False if pd.isna(x) or x == 0 else True)
            filtered_oa  = df_dedup_v2[(df_dedup_v2["Citation status"]) & (df_dedup_v2["OA status"] == True)]
            filtered_oa2 = df_dedup_v2[df_dedup_v2["Citation status"]]
            df_cited_oa  = filtered_oa.groupby("Date year")["OA status"].count().reset_index()
            df_cited_oa.columns = ["Date year","Cited OA papers"]

            @st.fragment
            def oa_charts():
                df_cited_p = filtered_oa2.groupby("Date year")["OA status"].count().reset_index()
                df_cited_p.columns = ["Date year","Cited papers"]
                df_cited_p = pd.merge(df_cited_p, df_cited_oa, on="Date year", how="left")
                df_cited_p["Cited OA papers"]     = df_cited_p["Cited OA papers"].fillna(0)
                df_cited_p["Cited non-OA papers"] = df_cited_p["Cited papers"] - df_cited_p["Cited OA papers"]
                df_cited_p["%Cited OA papers"]     = round(df_cited_p["Cited OA papers"] / df_cited_p["Cited papers"], 3) * 100
                df_cited_p["%Cited non-OA papers"] = 100 - df_cited_p["%Cited OA papers"]

                grouped_oa = df_dedup_v2.groupby("Date year")
                total_oa   = grouped_oa.size().reset_index(name="Total Publications")
                open_oa    = grouped_oa["OA status"].apply(lambda x: (x==True).sum()).reset_index(name="#OA Publications")
                df_oa_time = pd.merge(total_oa, open_oa, on="Date year")
                df_oa_time["#Non-OA Publications"]    = df_oa_time["Total Publications"] - df_oa_time["#OA Publications"]
                df_oa_time["OA publication ratio"]    = round(df_oa_time["#OA Publications"] / df_oa_time["Total Publications"], 3) * 100
                df_oa_time["Non-OA publication ratio"] = 100 - df_oa_time["OA publication ratio"]
                df_oa_time = pd.merge(df_oa_time, df_cited_p, on="Date year")

                col1, col2 = st.columns([3, 1])
                with col1:
                    last_20_oa = df_oa_time[df_oa_time["Date year"] >= (df_oa_time["Date year"].max() - 20)]
                    cit_ratio  = st.checkbox("Add citation ratio")
                    fig = px.bar(last_20_oa, x="Date year",
                                    y=["OA publication ratio","Non-OA publication ratio"],
                                    title="Open Access Publications Ratio (last 20 years)",
                                    color_discrete_map={"OA publication ratio":"green","Non-OA publication ratio":"#D3D3D3"},
                                    barmode="stack",
                                    hover_data=["#OA Publications","#Non-OA Publications"])
                    if cit_ratio:
                        fig.add_scatter(x=last_20_oa["Date year"], y=last_20_oa["%Cited OA papers"],
                                        mode="lines+markers", name="%Cited OA", line=dict(color="blue"))
                        fig.add_scatter(x=last_20_oa["Date year"], y=last_20_oa["%Cited non-OA papers"],
                                        mode="lines+markers", name="%Cited non-OA", line=dict(color="red"))
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    last_5_oa  = st.checkbox("Limit to last 5 years", key="last5years0")
                    df_pie_oa  = df_oa_time[df_oa_time["Date year"] >= (df_oa_time["Date year"].max() - 5)] if last_5_oa else df_oa_time
                    fig = px.pie(
                        values=[df_pie_oa["#OA Publications"].sum(), df_pie_oa["#Non-OA Publications"].sum()],
                        names=["OA Publications","Non-OA Publications"],
                        title="OA vs Non-OA" + (" (last 5 years)" if last_5_oa else " (all items)"),
                        color_discrete_sequence=["green","#D3D3D3"],
                    )
                    st.plotly_chart(fig)

            oa_charts()

            st.divider()
            st.subheader("Publications by citation status", anchor=False, divider="blue")

            @st.fragment
            def cited_status_charts():
                df_cit_sum  = df_dedup_v2.groupby("Date year")["Citation"].sum().reset_index()
                grouped_cit = df_dedup_v2.groupby("Date year")
                total_cit   = grouped_cit.size().reset_index(name="Total Publications")
                cited_cit   = grouped_cit["Citation status"].apply(lambda x: (x==True).sum()).reset_index(name="Cited Publications")
                df_cit_time = pd.merge(total_cit, cited_cit, on="Date year")
                df_cit_time = pd.merge(df_cit_time, df_cit_sum, on="Date year")
                df_cit_time["Non-cited Publications"]  = df_cit_time["Total Publications"] - df_cit_time["Cited Publications"]
                df_cit_time["%Cited Publications"]     = round(df_cit_time["Cited Publications"] / df_cit_time["Total Publications"], 3) * 100
                df_cit_time["%Non-Cited Publications"] = 100 - df_cit_time["%Cited Publications"]

                col1, col2 = st.columns(2)
                with col1:
                    last_20_cit = df_cit_time[df_cit_time["Date year"] >= (df_cit_time["Date year"].max() - 20)]
                    add_count   = st.checkbox("Add citation count")
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=last_20_cit["Date year"], y=last_20_cit["%Cited Publications"],
                                            name="%Cited", marker_color="#17becf"))
                    fig.add_trace(go.Bar(x=last_20_cit["Date year"], y=last_20_cit["%Non-Cited Publications"],
                                            name="%Non-Cited", marker_color="#D3D3D3"))
                    if add_count:
                        fig.add_trace(go.Scatter(x=last_20_cit["Date year"], y=last_20_cit["Citation"],
                                                    name="#Citations", mode="lines+markers",
                                                    marker=dict(color="green"), yaxis="y2"))
                    fig.update_layout(
                        title="Cited papers ratio (last 20 Years)",
                        barmode="stack",
                        yaxis2=dict(title="#Citations", overlaying="y", side="right") if add_count else {},
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    last_5_cit = st.checkbox("Limit to last 5 years", key="last5years1")
                    df_pie_cit = df_cit_time[df_cit_time["Date year"] >= (df_cit_time["Date year"].max() - 5)] if last_5_cit else df_cit_time
                    fig = px.pie(
                        values=[df_pie_cit["Cited Publications"].sum(), df_pie_cit["Non-cited Publications"].sum()],
                        names=["Cited","Non-cited"],
                        title="Cited vs Non-cited" + (" (last 5 years)" if last_5_cit else " (all items)"),
                        color_discrete_sequence=["green","#D3D3D3"],
                    )
                    st.plotly_chart(fig)

                with col2:
                    df_oa_cit  = filtered_oa.groupby("Date year")["Citation"].sum().reset_index()
                    df_oa_cit.columns = ["Date year","#Citations (OA papers)"]
                    df_all_cit = filtered_oa2.groupby("Date year")["Citation"].sum().reset_index()
                    df_all_cit.columns = ["Date year","#Citations (all)"]
                    df_cit_oa  = pd.merge(df_all_cit, df_oa_cit, on="Date year", how="left")
                    df_cit_oa["#Citations (OA papers)"].fillna(0, inplace=True)
                    df_cit_oa["#Citations (non-OA papers)"]       = df_cit_oa["#Citations (all)"] - df_cit_oa["#Citations (OA papers)"]
                    df_cit_oa["%Citation count (OA papers)"]      = round(df_cit_oa["#Citations (OA papers)"] / df_cit_oa["#Citations (all)"], 3) * 100
                    df_cit_oa["%Citation count (non-OA papers)"]  = 100 - df_cit_oa["%Citation count (OA papers)"]

                    last_20_coa = df_cit_oa[df_cit_oa["Date year"] >= (df_cit_oa["Date year"].max() - 20)]
                    line_show   = st.toggle("Citation count graph")
                    if line_show:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=last_20_coa["Date year"], y=last_20_coa["#Citations (OA papers)"],
                                                    mode="lines+markers", name="#Citations (OA)", line=dict(color="goldenrod")))
                        fig.add_trace(go.Scatter(x=last_20_coa["Date year"], y=last_20_coa["#Citations (non-OA papers)"],
                                                    mode="lines+markers", name="#Citations (non-OA)", line=dict(color="#D3D3D3")))
                        fig.update_layout(title="Citation Counts OA vs non-OA (last 20 Years)")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.bar(last_20_coa, x="Date year",
                                        y=["%Citation count (OA papers)","%Citation count (non-OA papers)"],
                                        title="OA vs non-OA Citation Count Ratio (last 20 Years)",
                                        color_discrete_map={"%Citation count (OA papers)":"goldenrod",
                                                            "%Citation count (non-OA papers)":"#D3D3D3"},
                                        barmode="stack",
                                        hover_data=["#Citations (OA papers)","#Citations (non-OA papers)"])
                        st.plotly_chart(fig, use_container_width=True)

                    last_5_coa = st.checkbox("Limit to last 5 years", key="last5years2")
                    df_pie_coa = df_cit_oa[df_cit_oa["Date year"] >= (df_cit_oa["Date year"].max() - 5)] if last_5_coa else df_cit_oa
                    fig = px.pie(
                        values=[df_pie_coa["#Citations (OA papers)"].sum(), df_pie_coa["#Citations (non-OA papers)"].sum()],
                        names=["#Citations (OA)","#Citations (non-OA)"],
                        title="OA vs non-OA citations" + (" (last 5 years)" if last_5_coa else " (all items)"),
                        color_discrete_sequence=["#D3D3D3","goldenrod"],
                    )
                    st.plotly_chart(fig)

            cited_status_charts()

            st.divider()
            st.subheader("Country mentions in titles", anchor=False, divider="blue")

            # Load and prepare data once
            df_countries = pd.read_csv("countries.csv")
            df_countries["Country"] = df_countries["Country"].replace("UK", "United Kingdom")
            df_countries = df_countries.groupby("Country", as_index=False).sum()

            # Get coordinates
            def get_coordinates(country_name):
                try:
                    return CountryInfo(country_name).info().get("latlng", (None, None))
                except KeyError:
                    return None, None

            df_countries[["Latitude", "Longitude"]] = df_countries["Country"].apply(
                lambda x: pd.Series(get_coordinates(x))
            )
            df_countries["size"] = df_countries["Count"] * 500 + 100000
            df_countries = df_countries.dropna(subset=["Latitude", "Longitude"])
            df_countries = df_countries.sort_values("Count", ascending=False).reset_index(drop=True)
            df_countries = df_countries.rename(columns={"Count": "# Mentions"})

            # Build map
            chart = pdk.Deck(
                layers=[pdk.Layer(
                    "ScatterplotLayer",
                    data=df_countries,
                    get_position=["Longitude", "Latitude"],
                    get_radius="size",
                    get_fill_color="[255, 140, 0, 160]",
                    pickable=True,
                    auto_highlight=True,
                    id="country-mentions-layer",
                )],
                initial_view_state=pdk.ViewState(latitude=20, longitude=0, zoom=1, pitch=30),
                tooltip={"text": "{Country}\nMentions: {# Mentions}"},
                map_style="light",
                height=800,  # ← increase this to make the map taller
            )

            col1, col2 = st.columns([8, 3])
            with col1:
                st.pydeck_chart(chart, use_container_width=True)
            with col2:
                st.dataframe(df_countries[["Country", "# Mentions"]], height=600, hide_index=True, use_container_width=True)
            

            st.divider()
            st.subheader("Locations, People, and Organisations", anchor=False, divider="blue")
            st.info("Named Entity Recognition (NER) retrieves locations, people, and organisations from titles and abstracts. [What is NER?](https://medium.com/mysuperai/what-is-named-entity-recognition-ner-and-how-can-i-use-it-2b68cf6f545d)")

            col1, col2, col3 = st.columns(3)
            with col1:
                gpe = pd.read_csv("gpe.csv")
                st.plotly_chart(px.bar(gpe.head(15), x="GPE", y="count", height=600,
                                        title="Top 15 locations").update_xaxes(tickangle=-65), use_container_width=True)
            with col2:
                per = pd.read_csv("person.csv")
                st.plotly_chart(px.bar(per.head(15), x="PERSON", y="count", height=600,
                                        title="Top 15 persons").update_xaxes(tickangle=-65), use_container_width=True)
            with col3:
                org = pd.read_csv("org.csv")
                st.plotly_chart(px.bar(org.head(15), x="ORG", y="count", height=600,
                                        title="Top 15 organisations").update_xaxes(tickangle=-65), use_container_width=True)

            st.write("---")
            st.subheader("Wordcloud", anchor=False, divider="blue")
            wordcloud_opt = st.radio("Wordcloud of:", ("Titles","Abstracts"), horizontal=True)
            df_wc     = df_csv.copy()
            df_abs_no = df_wc.dropna(subset=["Abstract"])
            if wordcloud_opt == "Abstracts":
                st.warning(f"Not all items have an abstract. Items with an abstract: {len(df_abs_no)}.")
                df_wc["Title"] = df_wc["Abstract"].astype(str)
            render_wordcloud(df_wc, title=f"Top words in {'abstracts' if wordcloud_opt == 'Abstracts' else 'titles'}")

        st.divider()
        st.subheader("Item inclusion history", anchor=False, divider="blue")

        @st.fragment
        def fragment_item_inclusion():
            st.write("This part shows the number of items added to the bibliography over time.")
            df_inc = df_dedup.copy()
            df_inc["Date added"] = pd.to_datetime(df_inc["Date added"])
            time_interval = st.selectbox("Select time interval:", ["Yearly","Monthly"])
            col11, col12 = st.columns(2)

            df_inc["YearMonth"] = df_inc["Date added"].dt.to_period("M").astype(str)
            monthly_inc = df_inc.groupby("YearMonth").size().rename("Number of items added")

            with col11:
                if time_interval == "Yearly":
                    df_inc["Year"] = df_inc["Date added"].dt.to_period("Y").astype(str)
                    yearly_inc = df_inc.groupby("Year").size().rename("Number of items added")
                    bar = (alt.Chart(yearly_inc.reset_index())
                            .mark_bar()
                            .encode(x="Year", y="Number of items added", tooltip=["Year","Number of items added"])
                            .properties(width=600, title="Number of Items Added per Year"))
                    st.altair_chart(bar, use_container_width=True)
                else:
                    bar = (alt.Chart(monthly_inc.reset_index())
                            .mark_bar()
                            .encode(x="YearMonth", y="Number of items added", tooltip=["YearMonth","Number of items added"])
                            .properties(width=600, title="Number of Items Added per Month"))
                    st.altair_chart(bar, use_container_width=True)

            with col12:
                if time_interval == "Monthly":
                    cum = monthly_inc.cumsum()
                    line = (alt.Chart(pd.DataFrame({"YearMonth": cum.index, "Cumulative": cum}))
                            .mark_line()
                            .encode(x="YearMonth", y="Cumulative", tooltip=["YearMonth","Cumulative"])
                            .properties(width=600, title="Cumulative Number of Items Added"))
                    st.altair_chart(line, use_container_width=True)
                else:
                    yearly_inc = df_inc.groupby("Year").size().rename("Number of items added")
                    cum_y = yearly_inc.cumsum()
                    line = (alt.Chart(pd.DataFrame({"Year": cum_y.index, "Cumulative": cum_y}))
                            .mark_line()
                            .encode(x="Year", y="Cumulative", tooltip=["Year","Cumulative"])
                            .properties(width=600, title="Cumulative Number of Items Added"))
                    st.altair_chart(line, use_container_width=True)

        fragment_item_inclusion()
    else:
        st.info("Toggle to see the dashboard!")


with tab3:
    st.header("Chat with IntelArchive", anchor=False)
    st.info("Ask questions about the intelligence studies database. Powered by Claude AI.")

    # ── API key input ───────────────────────────────────────────────────────
    with st.expander("🔑 Enter your Claude API key", expanded="user_api_key" not in st.session_state):
        st.markdown("""
        To use the chat feature you need a Claude API key from Anthropic.
        1. Go to [platform.anthropic.com](https://platform.anthropic.com)
        2. Sign up and add billing credits (minimum $5)
        3. Go to **API Keys** → **Create Key**
        4. Paste your key below
        
        Your key is stored only in your browser session and never saved anywhere.
        """)
        api_key_input = st.text_input(
            "Claude API key",
            type="password",
            placeholder="sk-ant-...",
            key="api_key_input_field"
        )
        if st.button("Save API key"):
            if api_key_input.startswith("sk-ant-"):
                st.session_state["user_api_key"] = api_key_input
                st.success("API key saved for this session.")
                st.rerun()
            else:
                st.error("Invalid API key format. It should start with 'sk-ant-'")

    if "user_api_key" not in st.session_state:
        st.warning("Please enter your Claude API key above to use the chat feature.")

    else:
        # ── Everything below only renders when API key is set ───────────────
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🔑 Clear API key"):
                del st.session_state["user_api_key"]
                st.rerun()
        with col2:
            st.caption("API key active for this session ✓")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the database..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Build context from database
        with st.spinner("Searching database..."):
            stop_words = {"can", "you", "find", "any", "publications", "containing",
                        "in", "the", "title", "about", "with", "a", "an", "and",
                        "or", "is", "are", "what", "who", "how", "many", "show",
                        "me", "please", "list", "give", "tell", "do", "have",
                        "has", "been", "that", "this", "for", "of", "to", "did",
                        "had", "good", "bad", "well", "better", "best", "was",
                        "were", "there", "their", "its", "his", "her", "our"}

            keywords = [
                w.strip("'\"?,.")
                for w in prompt.split()
                if w.lower().strip("'\"?.,") not in stop_words
                and len(w.strip("'\"?.,")) > 3
            ]

            if keywords:
                df_search = df_dedup.copy()
                df_search["_title"]    = df_search["Title"].fillna("").str.lower()
                df_search["_abstract"] = df_search["Abstract"].fillna("").str.lower()
                df_search["_combined"] = df_search["_title"] + " " + df_search["_abstract"]

                # Score each row by how many keywords it matches
                def score_row(text):
                    return sum(1 for k in keywords if k.lower() in text)

                df_search["_score"] = df_search["_combined"].apply(score_row)

                # Only keep rows that match at least 2 keywords, sorted by score
                relevant = df_search[df_search["_score"] >= 2].sort_values(
                    "_score", ascending=False
                ).head(30)

                # If nothing matches 2+, fall back to 1 keyword match
                if relevant.empty:
                    relevant = df_search[df_search["_score"] >= 1].sort_values(
                        "_score", ascending=False
                    ).head(30)
            else:
                relevant = pd.DataFrame()

            if relevant.empty:
                context = f"No publications found matching keywords: {keywords}. Database has {len(df_dedup)} total publications."
            else:
                context = f"Found {len(relevant)} relevant publications (ranked by relevance):\n\n"
                for _, row in relevant.iterrows():
                    context += f"Title: {row['Title']}\n"
                    context += f"Authors: {row.get('FirstName2', 'N/A')}\n"
                    context += f"Date: {row.get('Date published', 'N/A')}\n"
                    context += f"Type: {row.get('Publication type', 'N/A')}\n"
                    context += f"Journal/Publisher: {row.get('Journal') or row.get('Publisher', 'N/A')}\n"
                    abstract = str(row.get('Abstract', ''))
                    if abstract and abstract != 'nan':
                        context += f"Abstract: {abstract[:500]}{'...' if len(abstract) > 500 else ''}\n"
                    context += "\n"

        # Call Claude API
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    import anthropic
                    client = anthropic.Anthropic(api_key=st.session_state["user_api_key"])

                    response = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=1024,
                        system="""You are an assistant for IntelArchive, an intelligence studies bibliography database containing over 8,000 publications. 
                        You will be given database context containing relevant publications found by searching the database.
                        If publications are provided in the context, list them specifically with their titles and authors.
                        If the context says publications were found, report them — do not say you cannot find them.
                        Be specific and always cite exact titles and authors from the context provided.
                        Do not make up publications or authors not in the context.""",
                        messages=[
                            {
                                "role": "user",
                                "content": f"""Database context (relevant publications):
        {context}

        User question: {prompt}"""
                            }
                        ]
                    )
                    answer = response.content[0].text
                    st.markdown(answer)
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})

                except anthropic.AuthenticationError:
                    st.error("Invalid API key. Please check your key and try again.")
                    del st.session_state["user_api_key"]
                except anthropic.RateLimitError:
                    st.error("Rate limit reached. Please wait a moment and try again.")
                except Exception as e:
                    st.error(f"Error calling Claude API: {e}")

        if st.session_state.chat_messages:
            if st.button("Clear chat"):
                st.session_state.chat_messages = []
                st.rerun()

st.write("---")
with st.expander("Acknowledgements"):
    st.subheader("Acknowledgements", anchor=False)
    st.write("""
    The following sources are used to collate some of the items and events in this website:
    1. [King's Centre for the Study of Intelligence (KCSI) digest](https://kcsi.uk/kcsi-digests) compiled by Kayla Berg
    2. [International Association for Intelligence Education (IAIE) digest](https://www.iafie.org/Login.aspx) compiled by Filip Kovacevic

    Contributors with comments and sources:
    1. Daniela Richterove
    2. Steven Wagner
    3. Sophie Duroy

    Proudly sponsored by the [King's Centre for the Study of Intelligence](https://kcsi.uk/)
    """)

display_custom_license()


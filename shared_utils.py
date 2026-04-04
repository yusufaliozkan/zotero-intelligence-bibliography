"""
shared_utils.py  –  extracted helpers for the IntelArchive Streamlit app.
Import everything from here instead of repeating inline.
"""
from typing import Optional
import re

import datetime
from datetime import date

import numpy as np
import pandas as pd
import nltk
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def author_to_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

def slug_to_author(slug: str, author_list: list) -> str:
    return next((a for a in author_list if author_to_slug(a) == slug), "")
    
# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def parse_date_column(series: pd.Series, fmt: str = "%Y-%m-%d") -> pd.Series:
    """Convert a raw date string Series to London-localised dates, formatted as `fmt`."""
    return (
        series
        .astype(str)
        .str.strip()
        .apply(lambda x: pd.to_datetime(x, utc=True, errors="coerce").tz_convert("Europe/London"))
        .dt.strftime(fmt)
    )


def sort_by_date(df: pd.DataFrame, date_col: str = "Date published") -> pd.DataFrame:
    """Add a no-date flag and sort so dated items come first (most-recent last)."""
    df = df.copy()
    df["No date flag"] = df[date_col].isnull().astype(np.uint8)
    return df.sort_values(["No date flag", date_col], ascending=[True, True])


# ---------------------------------------------------------------------------
# Reviews map
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_reviews_map(path: str = "book_reviews.csv") -> dict:
    """Return {parentKey_upper: [url, ...]} from the book-reviews CSV."""
    try:
        df = pd.read_csv(path, dtype=str)
        df = df.dropna(subset=["parentKey", "url"]).copy()
        df["parentKey"] = df["parentKey"].str.strip().str.upper()
        return df.groupby("parentKey")["url"].apply(list).to_dict()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Text / NLP helpers  (previously copy-pasted in every report block)
# ---------------------------------------------------------------------------

_STOPWORDS_EXTRA = [
    "york", "intelligence", "security", "pp", "war", "world", "article",
    "twitter", "nan", "new", "isbn", "book", "also", "yet", "matter",
    "erratum", "commentary", "studies", "volume", "paper", "study",
    "question", "editorial", "welcome", "introduction", "reader",
    "university", "followed", "particular", "based", "press", "examine",
    "show", "may", "result", "explore", "examines", "become", "used",
    "journal", "london", "review",
]

def build_stopwords(extra: Optional[list] = None) -> list:
    stopword = nltk.corpus.stopwords.words("english")
    stopword.extend(_STOPWORDS_EXTRA)
    if extra:
        stopword.extend(extra)
    return stopword


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"[0-9_]", " ", text)
    text = re.sub(r"[^a-z_]", " ", text)
    return text


def tokenize(text: str) -> list:
    return re.split(r"\W+", text)


def remove_stopwords(tokens: list, stopword: list) -> list:
    return [w for w in tokens if w and w not in stopword]


def lemmatize(tokens: list) -> list:
    wn = nltk.WordNetLemmatizer()
    return [wn.lemmatize(w) for w in tokens]


def prepare_title_tokens(df: pd.DataFrame, stopword: list) -> pd.Series:
    """Full NLP pipeline on the Title column; returns a Series of token lists."""
    cleaned = df["Title"].apply(clean_text)
    cleaned = cleaned.apply(lambda x: " ".join(w for w in x.split() if len(w) > 2))
    tokens = cleaned.apply(tokenize)
    tokens = tokens.apply(lambda t: remove_stopwords(t, stopword))
    return tokens.apply(lemmatize)


def render_wordcloud(df: pd.DataFrame, title: str, extra_stopwords: list | None = None):
    """Build and render a WordCloud from df['Title']."""
    stopword = build_stopwords(extra_stopwords)
    token_series = prepare_title_tokens(df, stopword)
    all_tokens = [tok for sublist in token_series for tok in sublist]
    text = pd.Series(all_tokens).str.cat(sep=" ")
    wc = WordCloud(
        stopwords=stopword, width=1500, height=750,
        background_color="white", collocations=False, colormap="magma",
    ).generate(str(text))
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.imshow(wc)
    ax.axis("off")
    ax.set_title(title)
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def split_and_expand(authors_series: pd.Series) -> pd.Series:
    """Explode comma-separated author strings into individual rows."""
    def _split(x):
        if isinstance(x, str):
            return pd.Series([a.strip() for a in x.split(",")])
        return pd.Series([x])
    return authors_series.apply(_split).stack().reset_index(level=1, drop=True)


def remove_numbers(name: str) -> str:
    """Strip leading numeric prefixes from collection names."""
    return re.sub(r"^\d+(\.\d+)*\s*", "", name)


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# ---------------------------------------------------------------------------
# Shared metrics rendering
# ---------------------------------------------------------------------------

def render_metrics(
    df: pd.DataFrame,
    *,
    container_metric,
    container_citation,
    container_citation_average,
    container_oa,
    container_type=None,
    container_author_no=None,
    container_author_pub_ratio=None,
    container_publication_ratio=None,
    label: str = "Number of items",
):
    """Render the standard set of st.metric cards into pre-created containers."""
    num_items = len(df)
    publications_by_type = df["Publication type"].value_counts()
    breakdown_string = ", ".join(f"{k}: {v}" for k, v in publications_by_type.items())
    container_metric.metric(label=label, value=int(num_items), help=breakdown_string)

    # Citations
    citation_count = df["Citation"].sum()
    non_nan = df.dropna(subset=["Citation_list"])
    citation_mean = non_nan["Citation"].mean() if len(non_nan) else 0
    citation_median = non_nan["Citation"].median() if len(non_nan) else 0
    container_citation.metric(
        label="Number of citations",
        value=int(citation_count),
        help=f"Citation per publication: **{round(citation_mean, 1)}**, median: **{round(citation_median, 1)}**",
    )

    outlier_count = int((df["Citation"] > 1000).sum())
    avg_wo_outliers = round(df.loc[df["Citation"] < 1000, "Citation"].mean(), 2) if num_items else 0
    citation_average = round(df["Citation"].mean(), 2) if num_items else 0
    container_citation_average.metric(
        label="Average citation",
        value=citation_average,
        help=f"**{outlier_count}** outlier(s) >1000 citations. Without outliers: **{avg_wo_outliers}**.",
    )

    # OA (journal articles only)
    ja = df[df["Publication type"] == "Journal article"]
    oa_ratio = (ja["OA status"].sum() / len(ja) * 100) if len(ja) else 0
    container_oa.metric(label="Open access coverage", value=f"{int(oa_ratio)}%", help="Journal articles only")

    if container_type is not None:
        container_type.metric(label="Number of publication types", value=int(df["Publication type"].nunique()))

    if container_author_no is not None or container_author_pub_ratio is not None:
        expanded = split_and_expand(df["FirstName2"])
        author_no = len(expanded)
        author_pub_ratio = round(author_no / num_items, 2) if num_items else 0
        if container_author_no is not None:
            container_author_no.metric(label="Number of authors", value=int(author_no))
        if container_author_pub_ratio is not None:
            container_author_pub_ratio.metric(
                label="Author/publication ratio", value=author_pub_ratio,
                help="Average author count per publication",
            )

    if container_publication_ratio is not None:
        df = df.copy()
        df["FirstName2"] = df["FirstName2"].astype(str)
        multi = df["FirstName2"].apply(lambda x: "," in x).sum()
        collab = round(multi / num_items * 100, 1) if num_items else 0
        container_publication_ratio.metric(
            label="Collaboration ratio", value=f"{collab}%",
            help="Ratio of multiple-authored papers",
        )


# ---------------------------------------------------------------------------
# Shared report charts  (previously copy-pasted in every "Generate report" block)
# ---------------------------------------------------------------------------

def render_report_charts(
    df: pd.DataFrame,
    label: str,
    name_replacements: dict,
    show_themes: bool = False,
    themes_df: Optional[pd.DataFrame] = None,
):
    """Render the standard suite of report charts for any filtered dataframe."""
    # Publications by type
    by_type = df["Publication type"].value_counts()
    st.plotly_chart(
        px.bar(by_type, x=by_type.index, y=by_type.values,
               labels={"x": "Publication Type", "y": "Count"},
               title=f"Publications by Type ({label})"),
        use_container_width=True,
    )

    # Publications by year
    df = df.copy()
    df["Year"] = pd.to_datetime(df["Date published"]).dt.year
    by_year = df["Year"].value_counts().sort_index()
    fig_year = px.bar(
        by_year, x=by_year.index, y=by_year.values,
        labels={"x": "Year", "y": "Count"},
        title=f"Publications by Year ({label})",
    )
    fig_year.update_xaxes(type="category", tickangle=-45)
    st.plotly_chart(fig_year, use_container_width=True)

    # Themes radar (optional)
    if show_themes and themes_df is not None and not themes_df.empty:
        fig = px.line_polar(
            themes_df, r="Number_of_Items", theta="Collection_Name",
            line_close=True, title=f"Top Publication Themes ({label})",
        )
        fig.update_traces(fill="toself")
        st.plotly_chart(fig, use_container_width=True)

    # Top 10 authors by publication count
    author_df = df.copy()
    author_df["Author_name"] = author_df["FirstName2"].apply(
        lambda x: x.split(", ") if isinstance(x, str) and x else []
    )
    author_df = author_df.explode("Author_name")
    author_df["Author_name"] = author_df["Author_name"].map(name_replacements).fillna(author_df["Author_name"])
    top_authors = author_df["Author_name"].value_counts().head(10)
    fig_auth = px.bar(
        top_authors, x=top_authors.index, y=top_authors.values,
        title=f"Top 10 Authors ({label})",
        labels={"x": "Author", "y": "Publications"},
    )
    fig_auth.update_layout(xaxis_tickangle=-45, xaxis_type="category")
    st.plotly_chart(fig_auth, use_container_width=True)

    # Word cloud
    render_wordcloud(df, title=f"Word Cloud for Titles ({label})")


# ---------------------------------------------------------------------------
# Shared bibliography display
# ---------------------------------------------------------------------------

def display_bibliographies(df: pd.DataFrame):
    df = df.copy()
    df["bibliography"] = df["bibliography"].fillna("").astype(str)
    html = "<p><p>".join(df["bibliography"].tolist())
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Shared sort-by radio + apply
# ---------------------------------------------------------------------------

def apply_sort(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    if sort_by == "Publication date :arrow_down:":
        return df.sort_values("Date published", ascending=False).reset_index(drop=True)
    elif sort_by == "Citation":
        return df.sort_values("Citation", ascending=False).reset_index(drop=True)
    elif sort_by == "Date added :arrow_down:":
        return df.sort_values("Date added", ascending=False).reset_index(drop=True)
    return df


def sort_radio(df: pd.DataFrame, key: str = "sort_radio") -> pd.DataFrame:
    sort_by = st.radio(
        "Sort by:",
        ("Publication date :arrow_down:", "Citation", "Date added :arrow_down:"),
        horizontal=True,
        key=key,
    )
    if sort_by == "Citation" and df["Citation"].sum() == 0:
        sort_by = "Publication date :arrow_down:"
    return apply_sort(df, sort_by)


# ---------------------------------------------------------------------------
# Paginated list display
# ---------------------------------------------------------------------------

def render_paginated_list(
    df: pd.DataFrame,
    articles_list: list,
    abstracts_list: Optional[list] = None,
    display_abstracts: bool = False,
    search_tokens: Optional[list] = None,
    search_in: str = "Title",
):
    """Display articles_list with optional pagination and abstract toggle."""

    def _highlight(text):
        if not search_tokens:
            return text
        boolean_ops = {"AND", "OR", "NOT"}
        urls = re.findall(r"https?://\S+", text)
        for i, u in enumerate(urls):
            text = text.replace(u, f"___URL_{i}___")
        pattern = re.compile(
            "|".join(rf"\b{re.escape(t)}\b" for t in search_tokens if t not in boolean_ops),
            flags=re.IGNORECASE,
        )
        text = pattern.sub(lambda m: f'<span style="background-color:#FF8581;">{m.group(0)}</span>', text)
        for i, u in enumerate(urls):
            text = text.replace(f"___URL_{i}___", u)
        return text

    num_items = len(articles_list)
    if num_items == 0:
        st.write("No articles found.")
        return

    if num_items <= 20:
        for i, article in enumerate(articles_list, 1):
            st.markdown(f"{i}. {_highlight(article)}", unsafe_allow_html=True)
            if display_abstracts and abstracts_list:
                abstract = abstracts_list[i - 1]
                highlighted_abstract = _highlight(abstract) if search_in == "Title and abstract" else abstract
                st.caption(f"Abstract: {highlighted_abstract}", unsafe_allow_html=True)
        return

    show_first_20 = st.checkbox("Show only first 20 items (untick to see all)", value=True)
    if show_first_20:
        items_to_show = articles_list[:20]
        abstracts_to_show = abstracts_list[:20] if abstracts_list else []
    else:
        items_to_show = articles_list
        abstracts_to_show = abstracts_list or []

    num_tabs = (len(items_to_show) // 20) + (1 if len(items_to_show) % 20 else 0)
    if num_tabs == 1:
        for i, article in enumerate(items_to_show, 1):
            st.markdown(f"{i}. {_highlight(article)}", unsafe_allow_html=True)
            if display_abstracts and abstracts_to_show:
                st.caption(f"Abstract: {_highlight(abstracts_to_show[i-1]) if search_in == 'Title and abstract' else abstracts_to_show[i-1]}", unsafe_allow_html=True)
    else:
        tab_titles = [
            f"Results {i*20+1}–{min((i+1)*20, len(items_to_show))}"
            for i in range(num_tabs)
        ]
        for tab_index, tab in enumerate(st.tabs(tab_titles)):
            with tab:
                start, end = tab_index * 20, min((tab_index + 1) * 20, len(items_to_show))
                for i in range(start, end):
                    st.markdown(f"{i+1}. {_highlight(items_to_show[i])}", unsafe_allow_html=True)
                    if display_abstracts and abstracts_to_show:
                        st.caption(
                            f"Abstract: {_highlight(abstracts_to_show[i]) if search_in == 'Title and abstract' else abstracts_to_show[i]}",
                            unsafe_allow_html=True,
                        )
"""Post newly added book reviews to Bluesky.

A book review is a linked-URL attachment whose title starts with "Book review"
(e.g. "Book review by Jane Smith"), attached to a book in the Zotero group library.

Post format:
    Book review for <Book title> by <Reviewer>

    <review URL>
"""
import os
import re
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import pytz
import requests
from atproto import Client
from bs4 import BeautifulSoup
from pyzotero import zotero

STATE_FILE = "bluesky_posts/last_posted_reviews.json"
REQUEST_TIMEOUT = 10  # seconds
MAX_POST_LENGTH = 300
LIBRARY_ID = "2514686"
LIBRARY_TYPE = "group"


# ── State ────────────────────────────────────────────────────────────────────
def load_last_posted():
    try:
        with open(STATE_FILE, "r") as f:
            return pd.to_datetime(json.load(f)["last_posted_date_added"], utc=True)
    except (FileNotFoundError, KeyError):
        # First ever run: only look back 1 hour so old reviews aren't all posted
        return datetime.now(pytz.UTC) - timedelta(hours=1)


def save_last_posted(ts):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_posted_date_added": ts.isoformat()}, f, indent=2)


# ── Zotero ───────────────────────────────────────────────────────────────────
def fetch_recent_reviews(zot, since) -> pd.DataFrame:
    """Return book-review attachments added after `since`, oldest first."""
    attachments = zot.items(itemType="attachment", sort="dateAdded", direction="desc", limit=100)

    rows = []
    for att in attachments:
        d = att["data"]
        title = (d.get("title") or "").strip()
        if d.get("linkMode") != "linked_url" or not title.lower().startswith("book review"):
            continue
        if not d.get("parentItem") or not d.get("url"):
            continue
        added = pd.to_datetime(d.get("dateAdded"), utc=True, errors="coerce")
        if pd.isna(added) or added <= since:
            continue
        rows.append({
            "parentKey": d["parentItem"],
            "linkTitle": title,
            "url": d["url"],
            "dateAdded": added,
        })

    df = pd.DataFrame(rows, columns=["parentKey", "linkTitle", "url", "dateAdded"])
    if df.empty:
        return df

    # Look up each parent book's title (one call per distinct book)
    titles = {}
    for key in df["parentKey"].unique():
        try:
            parent = zot.item(key)["data"]
            titles[key] = parent.get("title") or parent.get("shortTitle") or ""
        except Exception as e:
            print(f"Warning: could not fetch parent item {key}: {e}")
            titles[key] = ""
    df["bookTitle"] = df["parentKey"].map(titles)

    # "Book review by Jane Smith" -> "Jane Smith"
    df["reviewer"] = (
        df["linkTitle"]
        .str.replace(r"^\s*Book review\s*(by)?\s*", "", regex=True, case=False)
        .str.strip(" ()")
    )
    return df.sort_values("dateAdded").reset_index(drop=True)


# ── Post text ────────────────────────────────────────────────────────────────
def build_post_text(book_title: str, reviewer: str, url: str) -> str:
    book_title = (book_title or "Untitled").strip()
    by = f" by {reviewer}" if reviewer else ""
    tail = f"{by}\n\n{url}"
    max_title = MAX_POST_LENGTH - len("Book review for ") - len(tail)
    if len(book_title) > max_title:
        book_title = book_title[: max(max_title - 3, 0)].rstrip() + "..."
    return f"Book review for {book_title}{tail}"


# ── Bluesky helpers (same approach as the publications bot) ──────────────────
def fetch_link_metadata(url: str) -> Dict:
    try:
        r = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ZoteroBiblioBot/1.0)"},
        )
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Warning: could not fetch metadata for {url}: {e}")
        return {"title": "", "description": "", "image": "", "url": url}

    soup = BeautifulSoup(r.text, "html.parser")
    get = lambda prop: (soup.find("meta", property=prop) or {}).get("content", "")
    return {"title": get("og:title"), "description": get("og:description"),
            "image": get("og:image"), "url": url}


def create_link_card_embed(client, url: str) -> Optional[Dict]:
    meta = fetch_link_metadata(url)
    if not meta["title"] and not meta["description"]:
        return None
    thumb = None
    if meta["image"]:
        try:
            img = requests.get(meta["image"], timeout=REQUEST_TIMEOUT)
            img.raise_for_status()
            thumb = client.upload_blob(img.content)["blob"]
        except Exception as e:
            print(f"Warning: could not attach image for {url}: {e}")
    return {
        "$type": "app.bsky.embed.external",
        "external": {"uri": meta["url"], "title": meta["title"],
                     "description": meta["description"], "thumb": thumb},
    }


def url_facets(text: str) -> List[Dict]:
    """Make URLs in the post clickable (Bluesky needs byte offsets)."""
    url_regex = rb"[$|\W](https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*[-a-zA-Z0-9@%_\+~#//=])?)"
    return [
        {
            "index": {"byteStart": m.start(1), "byteEnd": m.end(1)},
            "features": [{"$type": "app.bsky.richtext.facet#link", "uri": m.group(1).decode("UTF-8")}],
        }
        for m in re.finditer(url_regex, text.encode("UTF-8"))
    ]


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    zot = zotero.Zotero(LIBRARY_ID, LIBRARY_TYPE)
    last_posted = load_last_posted()
    reviews = fetch_recent_reviews(zot, last_posted)
    print(f"{len(reviews)} new book review(s) since {last_posted.isoformat()}")
    if reviews.empty:
        return

    client = Client(base_url="https://bsky.social")
    client.login("intelarchive.io", os.getenv("BLUESKY_PASSWORD"))

    for _, rev in reviews.iterrows():
        text = build_post_text(rev["bookTitle"], rev["reviewer"], rev["url"])
        try:
            client.send_post(
                text=text,
                facets=url_facets(text),
                embed=create_link_card_embed(client, rev["url"]),
            )
            print(f"Posted: {text.splitlines()[0]}")
        except Exception as e:
            print(f"Failed to post review for {rev['bookTitle']}: {e}")
            # Stop here so this review is retried next run instead of being skipped
            break
        save_last_posted(rev["dateAdded"])


if __name__ == "__main__":
    main()

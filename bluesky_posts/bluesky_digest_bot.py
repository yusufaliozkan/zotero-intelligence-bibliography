"""Share this week's digest (as archived on Google Groups) on Bluesky.

Runs in the Weekly Digest workflow after digest_link.py, which saves the
digest's Google Groups link in digest_post.json.
"""
import json
import os
import sys

import requests
from atproto import Client

GROUP = "intelarchive"
GROUP_URL = f"https://groups.google.com/g/{GROUP}"
HANDOFF_FILE = "digest_post.json"
LOGO_URL = ("https://raw.githubusercontent.com/yusufaliozkan/zotero-intelligence-bibliography/"
            "main/images/01_logo/IntelArchive_Digital_Logo_Colour-Negative.png")
TIMEOUT = 15


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def build_text(items: int, reviews: int):
    """Return (text, link_text). link_text is made clickable via a facet."""
    parts = []
    if items:
        parts.append(plural(items, "new item"))
    if reviews:
        parts.append(plural(reviews, "book review"))
    summary = " and ".join(parts)
    link_text = "Read the digest"
    text = (f"📬 This week's IntelArchive digest is out: {summary}, "
            f"plus upcoming events and calls for papers.\n\n{link_text} →")
    return text, link_text


def link_facet(text: str, link_text: str, url: str):
    b = text.encode("utf-8")
    start = b.index(link_text.encode("utf-8"))
    return [{
        "index": {"byteStart": start, "byteEnd": start + len(link_text.encode("utf-8"))},
        "features": [{"$type": "app.bsky.richtext.facet#link", "uri": url}],
    }]


def link_card(client, url: str, title: str, description: str):
    thumb = None
    try:
        img = requests.get(LOGO_URL, timeout=TIMEOUT)
        img.raise_for_status()
        thumb = client.upload_blob(img.content)["blob"]
    except Exception as e:
        print(f"Warning: could not attach logo: {e}")
    return {
        "$type": "app.bsky.embed.external",
        "external": {"uri": url, "title": title, "description": description, "thumb": thumb},
    }


def main():
    if not os.path.exists(HANDOFF_FILE):
        print("No digest was sent this run. Nothing to post.")
        return
    with open(HANDOFF_FILE) as f:
        info = json.load(f)

    url = info.get("link") or GROUP_URL
    text, link_text = build_text(info.get("item_count", 0), info.get("review_count", 0))

    client = Client(base_url="https://bsky.social")
    client.login("intelarchive.io", os.getenv("BLUESKY_PASSWORD"))
    client.send_post(
        text=text,
        facets=link_facet(text, link_text, url),
        embed=link_card(client, url, info.get("subject", "IntelArchive Weekly Digest"),
                        "New publications, book reviews, events and calls for papers in intelligence studies."),
    )
    print(f"Posted to Bluesky: {text!r} -> {url}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Failed to post digest to Bluesky: {e}")
        sys.exit(1)
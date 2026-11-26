"""Share this week's digest (as archived on Google Groups) on Bluesky.

Runs right after send_digest.py in the Weekly Digest workflow. send_digest.py
writes digest_post.json with the email's Message-ID; Google Groups serves every
archived message at https://groups.google.com/d/msgid/<group>/<Message-ID>.
"""
import json
import os
import sys
import time
from urllib.parse import quote

import requests
from atproto import Client

GROUP = "intelarchive"
GROUP_URL = f"https://groups.google.com/g/{GROUP}"
HANDOFF_FILE = "digest_post.json"
LOGO_URL = ("https://raw.githubusercontent.com/yusufaliozkan/zotero-intelligence-bibliography/"
            "main/images/01_logo/IntelArchive_Digital_Logo_Colour-Negative.png")
CHECK_ATTEMPTS = 6        # how many times to check Google Groups has archived the email
CHECK_INTERVAL = 5 * 60   # seconds between checks (6 x 5 min = up to 30 min)
TIMEOUT = 15


def digest_link(message_id: str) -> str:
    """Wait until Google Groups has the message; fall back to the group page."""
    url = f"https://groups.google.com/d/msgid/{GROUP}/{quote(message_id, safe='')}"
    for attempt in range(1, CHECK_ATTEMPTS + 1):
        try:
            r = requests.get(url, timeout=TIMEOUT, allow_redirects=True,
                             headers={"User-Agent": "Mozilla/5.0 (compatible; IntelArchiveBot/1.0)"})
            if r.status_code == 200:
                print(f"Digest archived: {r.url}")
                # Prefer the clean thread URL if Groups redirected to it
                return r.url if "/c/" in r.url else url
            print(f"Attempt {attempt}: Groups returned {r.status_code}")
        except requests.RequestException as e:
            print(f"Attempt {attempt}: {e}")
        if attempt < CHECK_ATTEMPTS:
            time.sleep(CHECK_INTERVAL)
    print("Could not confirm the archived message; linking to the group page instead.")
    return GROUP_URL


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

    url = digest_link(info["message_id"])
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

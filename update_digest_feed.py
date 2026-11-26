"""Add this week's digest to the RSS feed at feeds/digest.xml.

Runs after digest_link.py. The feed is public at
https://raw.githubusercontent.com/yusufaliozkan/zotero-intelligence-bibliography/main/feeds/digest.xml
and is what Zapier watches to post the digest to the IntelArchive LinkedIn page.
"""
import json
import os
import xml.etree.ElementTree as ET
from email.utils import format_datetime
from datetime import datetime, timezone

HANDOFF_FILE = "digest_post.json"
FEED_FILE = "feeds/digest.xml"
MAX_ITEMS = 20


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def summary(info) -> str:
    parts = []
    if info.get("item_count"):
        parts.append(plural(info["item_count"], "new item"))
    if info.get("review_count"):
        parts.append(plural(info["review_count"], "book review"))
    what = " and ".join(parts) or "new additions"
    return (f"This week's IntelArchive digest is out: {what}, "
            f"plus upcoming events and calls for papers.")


def new_feed():
    rss = ET.Element("rss", version="2.0")
    ch = ET.SubElement(rss, "channel")
    ET.SubElement(ch, "title").text = "IntelArchive Weekly Digest"
    ET.SubElement(ch, "link").text = "https://groups.google.com/g/intelarchive"
    ET.SubElement(ch, "description").text = (
        "Weekly digest of new publications, book reviews, events and calls for papers "
        "in intelligence studies from IntelArchive.")
    ET.SubElement(ch, "language").text = "en-gb"
    return rss


def main():
    if not os.path.exists(HANDOFF_FILE):
        print("No digest was sent this run. Feed unchanged.")
        return
    with open(HANDOFF_FILE) as f:
        info = json.load(f)

    rss = ET.parse(FEED_FILE).getroot() if os.path.exists(FEED_FILE) else new_feed()
    ch = rss.find("channel")

    guid_text = info["message_id"]
    if any(g.text == guid_text for g in ch.iter("guid")):
        print("This digest is already in the feed.")
        return

    now = format_datetime(datetime.now(timezone.utc))
    item = ET.Element("item")
    ET.SubElement(item, "title").text = info.get("subject", "IntelArchive Weekly Digest")
    ET.SubElement(item, "link").text = info["link"]
    ET.SubElement(item, "description").text = summary(info)
    ET.SubElement(item, "guid", isPermaLink="false").text = guid_text
    ET.SubElement(item, "pubDate").text = now

    # Newest first, straight after the channel's own details
    items = ch.findall("item")
    insert_at = list(ch).index(items[0]) if items else len(ch)
    ch.insert(insert_at, item)
    for old in ch.findall("item")[MAX_ITEMS:]:
        ch.remove(old)
    lbd = ch.find("lastBuildDate")
    if lbd is None:
        lbd = ET.SubElement(ch, "lastBuildDate")
    lbd.text = now

    os.makedirs(os.path.dirname(FEED_FILE), exist_ok=True)
    ET.indent(rss)
    ET.ElementTree(rss).write(FEED_FILE, encoding="utf-8", xml_declaration=True)
    print(f"Added to feed: {info.get('subject')} -> {info['link']}")


if __name__ == "__main__":
    main()

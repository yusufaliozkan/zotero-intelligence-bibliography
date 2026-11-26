from pyzotero import zotero
import pandas as pd

# === RETRIEVING BOOK REVIEWS FROM ZOTERO LIBRARY ===
# A book review is a linked-URL attachment titled "Book review ..." on a library item.
# Fetches all attachments in a few paged calls (instead of one call per item),
# so this is quick enough to run with the hourly database sync.
library_id   = '2514686'
library_type = 'group'   # 'user' for personal library, 'group' for shared libraries
api_key      = None      # Only needed for private libraries/groups

zot = zotero.Zotero(library_id, library_type, api_key) if api_key else zotero.Zotero(library_id, library_type)

# Parent titles, keyed by item key
top_items = zot.everything(zot.top())
titles = {
    item["key"]: item["data"].get("title") or item["data"].get("shortTitle") or ""
    for item in top_items
}

# All attachments in the library
attachments = zot.everything(zot.items(itemType="attachment"))

rows = []
for child in attachments:
    d = child["data"]
    parent_key = d.get("parentItem")
    if not parent_key or parent_key not in titles:
        continue
    if d.get("linkMode") == "linked_url":
        rows.append({
            "attachmentKey": child["key"],
            "parentKey": parent_key,
            "parentTitle": titles[parent_key],
            "linkTitle": d.get("title"),
            "url": d.get("url"),
            "dateAdded": d.get("dateAdded"),
        })

# Build DataFrame
df = pd.DataFrame(rows, columns=["attachmentKey", "parentKey", "parentTitle", "linkTitle", "url", "dateAdded"])

df = df[df["linkTitle"].str.startswith("Book review", na=False)]
df = df.sort_values(["parentKey", "dateAdded"]).reset_index(drop=True)

print(f"{len(df)} book reviews found")
df.to_csv('book_reviews.csv')
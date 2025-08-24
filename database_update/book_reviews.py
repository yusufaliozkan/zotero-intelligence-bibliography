from pyzotero import zotero
import pandas as pd

# === RETRIEVING ITEMS FROM ZOTERO LIBRARY ===
library_id   = '2514686'
library_type = 'group'   # 'user' for personal library, 'group' for shared libraries
api_key      = None      # Only needed for private libraries/groups

zot = zotero.Zotero(library_id, library_type, api_key) if api_key else zotero.Zotero(library_id, library_type)

# ✅ Fetch *all* top-level items, not just 25
top_items = zot.everything(zot.top())

rows = []
for parent in top_items:
    parent_key   = parent["key"]
    parent_title = parent["data"].get("title") or parent["data"].get("shortTitle") or ""
    
    # Fetch children for this parent
    children = zot.children(parent_key)
    for child in children:
        d = child["data"]
        if d.get("itemType") == "attachment" and d.get("linkMode") == "linked_url":
            rows.append({
                "attachmentKey": child["key"],
                "parentKey": parent_key,
                "parentTitle": parent_title,
                "linkTitle": d.get("title"),
                "url": d.get("url")
            })

# Build DataFrame
df = pd.DataFrame(rows, columns=["attachmentKey", "parentKey", "parentTitle", "linkTitle", "url"])

df = df[df["linkTitle"].str.startswith("Book review", na=False)]

df.to_csv('book_reviews.csv')
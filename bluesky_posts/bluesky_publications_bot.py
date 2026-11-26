import os
from atproto import Client
import pandas as pd
from pyzotero import zotero
import numpy as np
import requests
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from grapheme import length as grapheme_length
from datetime import datetime, timedelta
import pytz
import re
import json

client = Client(base_url='https://bsky.social')
bluesky_password = os.getenv("BLUESKY_PASSWORD")
client.login('intelarchive.io', bluesky_password)

### POST ITEMS

STATE_FILE = "bluesky_posts/last_posted.json"
REQUEST_TIMEOUT = 10  # seconds, used for all outbound HTTP calls


def load_last_posted():
    try:
        with open(STATE_FILE, "r") as f:
            ts = json.load(f)["last_posted_date_added"]
            return pd.to_datetime(ts, utc=True)
    except (FileNotFoundError, KeyError):
        # first ever run — fall back to "1 hour ago" so it doesn't post the whole library
        return datetime.now(pytz.UTC) - timedelta(hours=1)


def save_last_posted(ts):
    with open(STATE_FILE, "w") as f:
        json.dump({"last_posted_date_added": ts.isoformat()}, f, indent=2)


def fetch_link_metadata(url: str, timeout: int = REQUEST_TIMEOUT) -> Dict:
    """Fetch OpenGraph metadata for a URL. Never raises — returns empty
    metadata on any network failure so a single bad link can't kill the run."""
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (compatible; ZoteroBiblioBot/1.0)"},
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Warning: could not fetch metadata for {url}: {e}")
        return {"title": "", "description": "", "image": "", "url": url}

    soup = BeautifulSoup(response.text, 'html.parser')

    title = soup.find("meta", property="og:title")
    description = soup.find("meta", property="og:description")
    image = soup.find("meta", property="og:image")

    return {
        "title": title["content"] if title else "",
        "description": description["content"] if description else "",
        "image": image["content"] if image else "",
        "url": url,
    }


def upload_image_to_bluesky(client, image_url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[Dict]:
    try:
        response = requests.get(image_url, timeout=timeout)
        response.raise_for_status()
        image_blob = client.upload_blob(response.content)
        return image_blob['blob']  # Assuming `blob` is the key where the blob reference is stored
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None
    except Exception as e:
        print(f"Error uploading image to Bluesky: {e}")
        return None


def create_link_card_embed(client, url: str) -> Optional[Dict]:
    metadata = fetch_link_metadata(url)

    # If we couldn't get any usable metadata, skip the embed entirely
    # rather than posting a blank/broken-looking card.
    if not metadata["title"] and not metadata["description"]:
        return None

    image_blob = None
    if metadata["image"]:
        image_blob = upload_image_to_bluesky(client, metadata["image"])

    embed = {
        '$type': 'app.bsky.embed.external',
        'external': {
            'uri': metadata['url'],
            'title': metadata['title'],
            'description': metadata['description'],
            'thumb': image_blob,  # This can be None if the image was invalid/missing
        },
    }
    return embed


def parse_mentions(text: str) -> List[Dict]:
    spans = []
    mention_regex = rb"[$|\W](@([a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)"
    text_bytes = text.encode("UTF-8")
    for m in re.finditer(mention_regex, text_bytes):
        spans.append({
            "start": m.start(1),
            "end": m.end(1),
            "handle": m.group(1)[1:].decode("UTF-8")
        })
    return spans


def parse_urls(text: str) -> List[Dict]:
    spans = []
    url_regex = rb"[$|\W](https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*[-a-zA-Z0-9@%_\+~#//=])?)"
    text_bytes = text.encode("UTF-8")
    for m in re.finditer(url_regex, text_bytes):
        spans.append({
            "start": m.start(1),
            "end": m.end(1),
            "url": m.group(1).decode("UTF-8"),
        })
    return spans


def parse_facets(text: str) -> List[Dict]:
    facets = []
    for m in parse_mentions(text):
        try:
            resp = requests.get(
                "https://bsky.social/xrpc/com.atproto.identity.resolveHandle",
                params={"handle": m["handle"]},
                timeout=REQUEST_TIMEOUT,
            )
        except requests.exceptions.RequestException as e:
            print(f"Warning: could not resolve handle {m['handle']}: {e}")
            continue
        if resp.status_code == 400:
            continue
        did = resp.json()["did"]
        facets.append({
            "index": {
                "byteStart": m["start"],
                "byteEnd": m["end"],
            },
            "features": [{"$type": "app.bsky.richtext.facet#mention", "did": did}],
        })
    for u in parse_urls(text):
        facets.append({
            "index": {
                "byteStart": u["start"],
                "byteEnd": u["end"],
            },
            "features": [
                {
                    "$type": "app.bsky.richtext.facet#link",
                    "uri": u["url"],
                }
            ],
        })
    return facets


def parse_facets_and_embed(text: str, client) -> Dict:
    facets = parse_facets(text)
    embed = None

    for facet in facets:
        if 'features' in facet and facet['features'][0]['$type'] == 'app.bsky.richtext.facet#link':
            url = facet['features'][0]['uri']
            embed = create_link_card_embed(client, url)
            break  # Only handle the first link

    return {
        'facets': facets,
        'embed': embed,
    }


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to fit within the max_length, considering full graphemes."""
    if len(text) <= max_length:
        return text
    else:
        return text[:max_length - 3] + "..."  # Reserve space for the ellipsis


library_id = '2514686'
library_type = 'group'
api_key = ''  # api_key is only needed for private groups and libraries

zot = zotero.Zotero(library_id, library_type)


def zotero_data(library_id, library_type):
    items = zot.top(limit=50)
    items = sorted(items, key=lambda x: x['data']['dateAdded'], reverse=True)
    data = []
    columns = ['Title', 'Publication type', 'Link to publication', 'Abstract', 'Zotero link', 'Date added',
               'Date published', 'Date modified', 'Col key', 'Authors', 'Pub_venue', 'Book_title',
               'Thesis_type', 'University']

    for item in items:
        creators = item['data']['creators']
        creators_str = ", ".join([
            creator.get('firstName', '') + ' ' + creator.get('lastName', '')
            if 'firstName' in creator and 'lastName' in creator
            else creator.get('name', '')
            for creator in creators
        ])
        data.append((item['data']['title'],
                     item['data']['itemType'],
                     item['data']['url'],
                     item['data']['abstractNote'],
                     item['links']['alternate']['href'],
                     item['data']['dateAdded'],
                     item['data'].get('date'),
                     item['data']['dateModified'],
                     item['data']['collections'],
                     creators_str,
                     item['data'].get('publicationTitle'),
                     item['data'].get('bookTitle'),
                     item['data'].get('thesisType', ''),
                     item['data'].get('university', '')
                     ))
    df = pd.DataFrame(data, columns=columns)
    return df


df = zotero_data(library_id, library_type)
df['Abstract'] = df['Abstract'].replace(r'^\s*$', np.nan, regex=True)  # To replace '' with NaN.
df['Abstract'] = df['Abstract'].fillna('No abstract')

split_df = pd.DataFrame(df['Col key'].tolist())
df = pd.concat([df, split_df], axis=1)
df['Authors'] = df['Authors'].fillna('null')

# Change type name
type_map = {
    'thesis': 'Thesis',
    'journalArticle': 'Journal article',
    'book': 'Book',
    'bookSection': 'Book chapter',
    'blogPost': 'Blog post',
    'videoRecording': 'Video',
    'podcast': 'Podcast',
    'magazineArticle': 'Magazine article',
    'webpage': 'Webpage',
    'newspaperArticle': 'Newspaper article',
    'report': 'Report',
    'forumPost': 'Forum post',
    'conferencePaper': 'Conference paper',
    'audioRecording': 'Podcast',
    'preprint': 'Preprint',
    'document': 'Document',
    'computerProgram': 'Computer program',
    'dataset': 'Dataset'
}

mapping_thesis_type = {
    "MA Thesis": "Master's Thesis",
    "PhD Thesis": "PhD Thesis",
    "Master Thesis": "Master's Thesis",
    "Thesis": "Master's Thesis",  # Assuming 'Thesis' refers to Master's Thesis here, adjust if necessary
    "Ph.D.": "PhD Thesis",
    "Master's Dissertation": "Master's Thesis",
    "Undergraduate Theses": "Undergraduate Thesis",
    "MPhil": "MPhil Thesis",
    "A.L.M.": "Master's Thesis",  # Assuming A.L.M. (Master of Liberal Arts) maps to Master's Thesis
    "doctoralThesis": "PhD Thesis",
    "PhD": "PhD Thesis",
    "Masters": "Master's Thesis",
    "PhD thesis": "PhD Thesis",
    "phd": "PhD Thesis",
    "doctoral": "PhD Thesis",
    "Doctoral": "PhD Thesis",
    "Master of Arts Dissertation": "Master's Thesis",
    "": 'Unclassified'
}
df['Thesis_type'] = df['Thesis_type'].replace(mapping_thesis_type)
df['Publication type'] = df['Publication type'].replace(type_map)
df['Date published'] = (
    df['Date published']
    .str.strip()
    .apply(lambda x: pd.to_datetime(x, utc=True, errors='coerce').tz_convert('Europe/London'))
)
df['Date published'] = df['Date published'].dt.strftime('%d-%m-%Y')
df['Date published'] = df['Date published'].fillna('No date')

df['Date added'] = pd.to_datetime(df['Date added'], errors='coerce', utc=True)

last_posted = load_last_posted()
df = df[df['Date added'] > last_posted]
df = df[['Title', 'Publication type', 'Link to publication', 'Zotero link', 'Date added',
         'Date published', 'Date modified', 'Authors']]

header = 'New addition\n\n'


def format_authors(authors_raw: str, max_authors: int = 2) -> str:
    if pd.isna(authors_raw) or not str(authors_raw).strip() or str(authors_raw).strip().lower() == "null":
        return ""

    # Your creators_str uses ", " as the separator
    authors = [a.strip() for a in str(authors_raw).split(",") if a.strip()]

    if len(authors) <= max_authors:
        return ", ".join(authors)
    else:
        return f"{authors[0]} et al."


for index, row in df.iterrows():
    publication_type = row['Publication type']
    title = row['Title']
    publication_date = row['Date published']
    link = row['Link to publication']
    author_name = format_authors(row['Authors'])  # Extract the author name

    # Calculate maximum title length without truncating the link or additional info
    max_title_length = 300 - len(header) - len(f"{publication_type}: (published {publication_date})\n\n{link}") - len(author_name) - 10  # Reserve space for formatting
    truncated_title = truncate_text(title, max_title_length)

    # Assemble the post text
    post_text = f"{header}{publication_type}: {truncated_title} by {author_name} (published {publication_date})\n\n{link}"

    # Ensure the final text fits within 300 characters
    if len(post_text) > 300:
        print(f"Post text exceeded 300 characters after adjustments: {post_text}")
        post_text = post_text[:300]  # This should rarely happen now

    # Parse facets and embed (never raises — bad links just skip the embed)
    parsed = parse_facets_and_embed(post_text, client)

    post_payload = {
        "$type": "app.bsky.feed.post",
        "text": post_text,
        "facets": parsed['facets'],
        "embed": parsed['embed'],  # Include the embed if present
        "createdAt": pd.Timestamp.now('UTC').isoformat().replace('+00:00', 'Z'),
    }

    try:
        post = client.send_post(
            text=post_payload["text"],
            facets=post_payload["facets"],
            embed=post_payload.get("embed"),  # Pass the embed if it exists
        )
    except Exception as e:
        print(f"Failed to post: {e}")

if not df.empty:
    save_last_posted(df['Date added'].max())
import os
from atproto import Client
import pandas as pd
from pyzotero import zotero
import numpy as np
import requests
from typing import List, Dict
from bs4 import BeautifulSoup
from grapheme import length as grapheme_length
from datetime import datetime, timedelta
import pytz
import re 

client = Client(base_url='https://bsky.social')
bluesky_password = os.getenv("BLUESKY_PASSWORD")
client.login('intelarchive.app', bluesky_password)

### POST ITEMS

def fetch_link_metadata(url: str) -> Dict:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    title = soup.find("meta", property="og:title")
    description = soup.find("meta", property="og:description")
    image = soup.find("meta", property="og:image")

    metadata = {
        "title": title["content"] if title else "",
        "description": description["content"] if description else "",
        "image": image["content"] if image else "",
        "url": url,
    }
    return metadata

def upload_image_to_bluesky(client, image_url: str) -> str:
    try:
        response = requests.get(image_url)
        image_blob = client.upload_blob(response.content)
        return image_blob['blob']  # Assuming `blob` is the key where the blob reference is stored
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image: {e}")
        return None
    except Exception as e:
        print(f"Error uploading image to Bluesky: {e}")
        return None


def create_link_card_embed(client, url: str) -> Dict:
    metadata = fetch_link_metadata(url)
    
    # Check if the image URL is valid
    if metadata["image"]:
        try:
            image_blob = upload_image_to_bluesky(client, metadata["image"])
        except requests.exceptions.MissingSchema:
            print(f"Invalid image URL: {metadata['image']}")
            image_blob = None
    else:
        image_blob = None

    embed = {
        '$type': 'app.bsky.embed.external',
        'external': {
            'uri': metadata['url'],
            'title': metadata['title'],
            'description': metadata['description'],
            'thumb': image_blob,  # This can be None if the image was invalid
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
        resp = requests.get(
            "https://bsky.social/xrpc/com.atproto.identity.resolveHandle",
            params={"handle": m["handle"]},
        )
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
        return text[:max_length-3] + "..."  # Reserve space for the ellipsis

library_id = '2514686'
library_type = 'group'
api_key = '' # api_key is only needed for private groups and libraries

zot = zotero.Zotero(library_id, library_type)
def zotero_data(library_id, library_type):
    items = zot.top(limit=50)
    items = sorted(items, key=lambda x: x['data']['dateAdded'], reverse=True)
    data=[]
    columns = ['Title','Publication type', 'Link to publication', 'Abstract', 'Zotero link', 'Date added', 'Date published', 'Date modified', 'Col key', 'Authors', 'Pub_venue', 'Book_title', 'Thesis_type', 'University']

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
df['Abstract'] = df['Abstract'].replace(r'^\s*$', np.nan, regex=True) # To replace '' with NaN. Otherwise the code below do not understand the value is nan.
df['Abstract'] = df['Abstract'].fillna('No abstract')

split_df= pd.DataFrame(df['Col key'].tolist())
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
    'conferencePaper' : 'Conference paper',
    'audioRecording' : 'Podcast',
    'preprint':'Preprint',
    'document':'Document',
    'computerProgram':'Computer program',
    'dataset':'Dataset'
}

mapping_thesis_type ={
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
    "":'Unclassified'
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
# df['Date published'] = df['Date published'].map(lambda x: x.strftime('%d/%m/%Y') if x else 'No date')

# df['Date added'] = pd.to_datetime(df['Date added'], errors='coerce')
# df['Date added'] = df['Date added'].dt.strftime('%d-%m-%Y')
df['Date added'] = pd.to_datetime(df['Date added'], errors='coerce', utc=True)

# df['Date modified'] = pd.to_datetime(df['Date modified'], errors='coerce')
# df['Date modified'] = df['Date modified'].dt.strftime('%d/%m/%Y, %H:%M')

# today = datetime.now(pytz.UTC).date()
# days_ago = today - timedelta(days=3)

# df = df[df['Date added'].dt.date >= days_ago]

now = datetime.now(pytz.UTC)
last_hours = now - timedelta(hours=1)
df = df[df['Date added'] >= last_hours]
df = df[['Title', 'Publication type', 'Link to publication', 'Zotero link', 'Date added', 'Date published', 'Date modified', 'Authors']]

header='New addition\n\n'

for index, row in df.iterrows():
    publication_type = row['Publication type']
    title = row['Title']
    publication_date = row['Date published']
    link = row['Link to publication']
    author_name = row['Authors']  # Extract the author name

    post_text = f"{header}{publication_type}: {title} by {author_name} (published {publication_date})\n\n{link}"

    if len(post_text) > 300:
        max_title_length = 300 - len(f"{publication_type}: \n{link}") - len(f" (published {publication_date})")
        truncated_title = truncate_text(title, max_title_length)
        post_text = f"{header}{publication_type}: {truncated_title} (published {publication_date})\n{link}"

    # Make sure the entire post_text fits within 300 graphemes
    post_text = truncate_text(post_text, 300)

    parsed = parse_facets_and_embed(post_text, client)
    
    post_payload = {
        "$type": "app.bsky.feed.post",
        "text": post_text,
        "facets": parsed['facets'],
        "embed": parsed['embed'],  # Include the embed if present
        "createdAt": pd.Timestamp.utcnow().isoformat() + "Z"
    }

    try:
        post = client.send_post(
            text=post_payload["text"],  
            facets=post_payload["facets"],  
            embed=post_payload.get("embed"),  # Pass the embed if it exists
        )
    except Exception as e:
        print(f"Failed to post: {e}")

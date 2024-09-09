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
from urllib.parse import quote

client = Client(base_url='https://bsky.social')
bluesky_password = os.getenv("BLUESKY_PASSWORD")
client.login('intelarchive.app', bluesky_password)

### POST ITEMS

def fetch_link_metadata(url: str) -> Dict:
    # URL Encode the URL to handle special characters properly
    encoded_url = quote(url, safe=':/')
    response = requests.get(encoded_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    title = soup.find("meta", property="og:title")
    description = soup.find("meta", property="og:description")
    image = soup.find("meta", property="og:image")

    metadata = {
        "title": title["content"] if title else "",
        "description": description["content"] if description else "",
        "image": image["content"] if image else "",
        "url": url,  # Use the original URL for display purposes
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
    # Use encoded URL when fetching metadata
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
            'uri': metadata['url'],  # Use the original URL here
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
    # Updated regex to capture URLs with commas and other special characters
    url_regex = rb"(https?://[^\s]+)"
    text_bytes = text.encode("UTF-8")
    for m in re.finditer(url_regex, text_bytes):
        url = m.group(1).decode("UTF-8")
        encoded_url = quote(url, safe=':/')  # Encode the URL properly
        spans.append({
            "start": m.start(1),
            "end": m.end(1),
            "url": encoded_url,  # Use the encoded URL
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

events_sheet_url = "https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/export?format=csv&gid=1941981997"

# Read the Google Sheet into a DataFrame
df_forms = pd.read_csv(events_sheet_url)

df_forms = df_forms.rename(columns={'Event name':'event_name', 'Event organiser':'organiser','Link to the event':'link','Date of event':'date', 'Event venue':'venue', 'Details':'details'})
df_forms['date'] = pd.to_datetime(df_forms['date'])
df_forms['date_new'] = df_forms['date'].dt.strftime('%Y-%m-%d')
# Calculate the date range: today + 2 days
start_date = pd.to_datetime('today').normalize()
end_date = start_date + pd.Timedelta(days=2)
end_date

# Filter the DataFrame to include only events within the date range
# df_forms = df_forms[(df_forms['date'] >= start_date) & (df_forms['date'] <= end_date)]
df_forms = df_forms[df_forms['date'] == end_date]
df_forms = df_forms.drop_duplicates(subset=['event_name', 'link', 'date'], keep='first')

conf_sheet_url_1 = "https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/export?format=csv&gid=939232836"
df_con = pd.read_csv(conf_sheet_url_1)
df_con['date'] = pd.to_datetime(df_con['date'], errors='coerce')
df_con['date_new'] = df_con['date'].dt.strftime('%Y-%m-%d')
df_con = df_con[df_con['date'] == end_date]
df_con = df_con.rename(columns={'conference_name':'event_name'})

conf_sheet_url_1 = "https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/export?format=csv&gid=312814443"
df_con_2 = pd.read_csv(conf_sheet_url_1)
df_con_2['date'] = pd.to_datetime(df_con_2['date'], errors='coerce')
df_con_2['date_new'] = df_con_2['date'].dt.strftime('%Y-%m-%d')
df_con_2 = df_con_2[df_con_2['date'] == end_date]
df_con_2 = df_con_2.rename(columns={'conference_name':'event_name'})
df_con = pd.concat([df_con, df_con_2])
df_con = df_con.drop_duplicates(subset='link')

# df_con = df_con[['conference_name', 'organiser', 'link', 'venue', 'date_new']]
# df_con = df_con.rename(columns={'conference_name':'event_name'})

df_forms = pd.concat([df_forms, df_con])


for index, row in df_forms.iterrows():
    event_name = row['event_name']
    organiser = row['organiser']
    event_date = row['date_new']
    link = row['link']
    venue = row['venue']  # Extract the author name

    post_text = f"{venue}\n\n{event_name} by {organiser} (on {event_date})\n\n{link}"

    if len(post_text) > 300:
        max_title_length = 300 - len(f"{venue}: \n{link}") - len(f" (on {event_date})")
        truncated_title = truncate_text(event_name, max_title_length)
        post_text = f"{venue}\n\n{event_name} (on {event_date})\n{link}"

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
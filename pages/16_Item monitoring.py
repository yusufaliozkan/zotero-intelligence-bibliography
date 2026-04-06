from pyzotero import zotero
import pandas as pd
import streamlit as st
from IPython.display import HTML
import streamlit.components.v1 as components
import numpy as np
from datetime import date, timedelta, datetime  # Combined datetime-related imports
import pytz  # Import pytz to work with timezone information
import re
from sidebar_content import sidebar_content
import requests
from rss_feed import df_podcast, df_magazines
from events import evens_conferences
import xml.etree.ElementTree as ET
from fuzzywuzzy import fuzz
from atproto import Client
import os
from bs4 import BeautifulSoup
from grapheme import length as grapheme_length
from typing import List, Dict
from st_keyup import st_keyup
from streamlit_gsheets import GSheetsConnection
import gspread
from copyright import display_custom_license
from urllib.parse import quote
from sidebar_content import sidebar_content, set_page_config
from streamlit_theme import st_theme


set_page_config()

theme = st_theme()
# Set the image path based on the theme
if theme and theme.get('base') == 'dark':
    image_path = 'images/01_logo/IntelArchive_Digital_Logo_Colour-Negative.svg'
else:
    image_path = 'images/01_logo/IntelArchive_Digital_Logo_Colour-Positive.svg'

# Read and display the SVG image
with open(image_path, 'r') as file:
    svg_content = file.read()
    st.image(svg_content, width=200)  # Adjust the width as needed
    
st.header('Item monitoring', anchor=False)

image = 'https://images.pexels.com/photos/315918/pexels-photo-315918.png'
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

# ### RETRIEVING CITATION COUNT AND OA STATUS FROM OPENALEX
# df = pd.read_csv('all_items.csv')
# df_doi = pd.read_csv('all_items.csv')

# df_doi = df_doi[['Zotero link', 'DOI']].dropna()
# df_doi = df_doi.drop(df_doi[df_doi['DOI'] == ''].index)
# df_doi = df_doi.reset_index(drop=True)
# df_doi['DOI'] = df_doi['DOI'].str.replace('https://doi.org/', '')

# def fetch_article_metadata(doi):
#     base_url = 'https://api.openalex.org/works/https://doi.org/'
#     response = requests.get(base_url + doi)
#     if response.status_code == 200:
#         data = response.json()
#         counts_by_year = data.get('counts_by_year', [])
#         if counts_by_year:
#             first_citation_year = min(entry.get('year') for entry in data['counts_by_year'])
#         else:
#             first_citation_year = None
#         if data.get('counts_by_year'):
#             last_citation_year = max(entry.get('year') for entry in data['counts_by_year'])
#         else:
#             last_citation_year = None

#         article_metadata = {
#             'ID': data.get('id'),
#             'Citation': data.get('cited_by_count'),
#             'OA status': data.get('open_access', {}).get('is_oa'),
#             'Citation_list': data.get('cited_by_api_url'),
#             'First_citation_year': first_citation_year,
#             'Last_citation_year': last_citation_year,
#             'Publication_year': data.get('publication_year'),
#             'OA_link': data.get('open_access', {}).get('oa_url')
#         }
#         return article_metadata
#     else:
#         return {
#             'ID': None,
#             'Citation': None,
#             'OA status': None,
#             'First_citation_year': None,
#             'Last_citation_year': None,
#             'Publication_year': None,
#             'OA_link': None
#         }

# df_doi['ID'] = None
# df_doi['Citation'] = None
# df_doi['OA status'] = None
# df_doi['Citation_list'] = None
# df_doi['First_citation_year'] = None
# df_doi['Last_citation_year'] = None
# df_doi['Publication_year'] = None
# df_doi['OA_link'] = None

# # Iterate over each row in the DataFrame
# for index, row in df_doi.iterrows():
#     doi = row['DOI']
#     article_metadata = fetch_article_metadata(doi)
#     if article_metadata:
#         # Update DataFrame with fetched information
#         df_doi.at[index, 'ID'] = article_metadata['ID']
#         df_doi.at[index, 'Citation'] = article_metadata['Citation']
#         df_doi.at[index, 'OA status'] = article_metadata['OA status']
#         df_doi.at[index, 'First_citation_year'] = article_metadata['First_citation_year']
#         df_doi.at[index, 'Last_citation_year'] = article_metadata['Last_citation_year']
#         df_doi.at[index, 'Citation_list'] = article_metadata.get('Citation_list', None)
#         df_doi.at[index, 'Publication_year'] = article_metadata['Publication_year']
#         df_doi.at[index, 'OA_link'] = article_metadata.get('OA_link', None)

# # Calculate the difference between First_citation_year and Publication_year
# df_doi['Year_difference'] = df_doi['First_citation_year'] - df_doi['Publication_year']

# df_doi = df_doi.drop(columns=['DOI'])
# df = pd.merge(df, df_doi, on='Zotero link', how='left')
# df
# total_citations = df['Citation'].sum()
# total_citations

# st.stop()

with st.sidebar:

    sidebar_content()

### Bluesky posting functions start here
client = Client(base_url='https://bsky.social')
bluesky_password = st.secrets["bluesky_password"]
client.login('intelarchive.io', bluesky_password)

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
### Bluesky posting functions end here

password_input = st.text_input("Enter the password to access admin dashboard:", type="password")

if not password_input:
    st.info('Enter the password')
else:
    if password_input == st.secrets['item_monitoring_password']:
        st.success('Wellcome to the admin dashboard')

        admin_task = st.radio('Select an option', ['Item monitoring', 'Post publications', 'Post events'], horizontal=True)

        if admin_task=='Post publications':
            @st.fragment
            def post_pubs():
                st.subheader('Post publications on Bluesky', anchor=False)

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
                last_hours = now - timedelta(hours=120)
                df = df[df['Date added'] >= last_hours]
                df['Include?'] = False
                last_column = df.columns[-1]
                df = df[[last_column] + list(df.columns[:-1])]
                df = df[['Include?', 'Title', 'Publication type', 'Link to publication', 'Zotero link', 'Date added', 'Date published', 'Date modified', 'Authors']]
                st.markdown('##### Recently added items')
                st.write('''
                Pick item(s) from the 'Include?' column.
                The selected items will be appear in the 'Items to be posted' table below.
                ''')
                select_all = st.checkbox("Select all")
                if select_all:
                    df['Include?']= True
                df = st.data_editor(df)
                df = df[df['Include?']==True]
                df = df.reset_index(drop=True)

                df_db = pd.read_csv('all_items.csv')
                df_db['Date published'] = pd.to_datetime(df_db['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
                df_db['Date published'] = df_db['Date published'].dt.strftime('%Y-%m-%d')
                df_db['Date published'] = df_db['Date published'].fillna('')
                df_db = df_db.sort_values(by=['Date published'], ascending=False)
                df_db = df_db.drop(columns=['Unnamed: 0'])
                df_db = df_db.reset_index(drop=True)
                df_db['Include?'] = False
                last_column = df_db.columns[-1]
                df_db = df_db[[last_column] + list(df_db.columns[:-1])]            
                name = st_keyup("Enter keywords to search in title", key='name', placeholder='Search keyword(s)', debounce=500)
                if name:
                    df_db = df_db[df_db.Title.str.lower().str.contains(name.lower(), na=False)]
                df_db = df_db[['Include?', 'Title', 'Publication type', 'Link to publication', 'Zotero link', 'Date added', 'Date published', 'FirstName2']]
                df_db = df_db.rename(columns={'FirstName2':'Authors'})
                st.markdown('##### All items')
                st.write('''
                Pick item(s) from the 'Include?' column.
                The selected items will be appear in the 'Items to be posted' table below.
                ''')
                df_db = st.data_editor(df_db)
                df_db = df_db[df_db['Include?']==True]
                df_db = df_db.reset_index(drop=True)
                df = pd.concat([df, df_db])
                df = df.reset_index(drop=True)
                df
                
                item_header = st.radio('Select a header', ['New addition', 'Recently published', 'Custom'], horizontal=True)
                if item_header=='New addition':
                    header='New addition\n\n'
                elif item_header=='Recently published':
                    header ='Recently published\n\n'
                else:
                    header = st.text_input('Write a custom header')
                    if not header:
                        header = ''
                    else:
                        header = f'{header}\n\n'
                # limit = st.number_input('Limit to:', min_value=0, max_value=100, value=0, step=1, format="%d")

                # if limit==0:
                #     st.info('Enter a value to limit the number of items.')
                # else:
                #     df = df.head(limit)
                #     df

                post_bluesky = st.button('Post items on Bluesky')
                if post_bluesky:

                    # Iterate through the dataframe and create posts with link cards

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
            post_pubs()
        elif admin_task=='Post events':

            @st.fragment
            def post_events():
                st.subheader('Post events on Bluesky', anchor=False)
                conn = st.connection("gsheets", type=GSheetsConnection)
                df_forms = conn.read(spreadsheet='https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/edit#gid=1941981997')
                
                
                # df_forms = df_forms.rename(columns={'Event name':'event_name', 'Event organiser':'organiser','Link to the event':'link','Date of event':'date', 'Event venue':'venue', 'Details':'details'})
                # df_forms['date'] = pd.to_datetime(df_forms['date'])
                # df_forms['date_new'] = df_forms['date'].dt.strftime('%Y-%m-%d')
                # # Calculate the date range: today + 4 days
                # start_date = pd.to_datetime('today').normalize()
                # end_date = start_date + pd.Timedelta(days=2)
                # end_date
                # # Filter the DataFrame to include only events within the date range
                # # df_forms = df_forms[(df_forms['date'] >= start_date) & (df_forms['date'] <= end_date)]
                # df_forms = df_forms[df_forms['date'] == end_date]
                # df_forms['month'] = df_forms['date'].dt.strftime('%m')
                # df_forms['year'] = df_forms['date'].dt.strftime('%Y')
                # df_forms['month_year'] = df_forms['date'].dt.strftime('%Y-%m')
                # df_forms.sort_values(by='date', ascending=True, inplace=True)
                # df_forms = df_forms.drop_duplicates(subset=['event_name', 'link', 'date'], keep='first')
                # df_forms = df_forms.reset_index(drop=True)
                # df_forms['Include?'] = False
                # last_column = df_forms.columns[-1]
                # df_forms = df_forms[[last_column] + list(df_forms.columns[:-1])]
                # st.markdown('##### Events')
                # st.write('''
                # Pick item(s) from the 'Include?' column.
                # The selected items will appear in the 'Items to be posted' table below.
                # ''')
                # df_forms = st.data_editor(df_forms)              
                
                
                df_forms = df_forms.rename(columns={'Event name':'event_name', 'Event organiser':'organiser','Link to the event':'link','Date of event':'date', 'Event venue':'venue', 'Details':'details'})
                df_forms['date'] = pd.to_datetime(df_forms['date'])
                df_forms['date_new'] = df_forms['date'].dt.strftime('%Y-%m-%d')
                df_forms['month'] = df_forms['date'].dt.strftime('%m')
                df_forms['year'] = df_forms['date'].dt.strftime('%Y')
                df_forms['month_year'] = df_forms['date'].dt.strftime('%Y-%m')
                df_forms.sort_values(by='date', ascending=True, inplace=True)
                df_forms = df_forms.drop_duplicates(subset=['event_name', 'link', 'date'], keep='first')
                df_forms = df_forms[df_forms['date_new'] >= pd.to_datetime('today').strftime('%Y-%m-%d')]
                df_forms = df_forms.reset_index(drop=True)
                df_forms['Include?'] = False
                last_column = df_forms.columns[-1]
                df_forms = df_forms[[last_column] + list(df_forms.columns[:-1])]
                st.markdown('##### Events')
                st.write('''
                Pick item(s) from the 'Include?' column.
                The selected items will be appear in the 'Items to be posted' table below.
                ''')
                df_forms = st.data_editor(df_forms)
        
                df_forms = df_forms[df_forms['Include?']==True]
                df_forms = df_forms.reset_index(drop=True)
                df_forms = df_forms[['event_name', 'organiser', 'link', 'venue', 'date_new']]

                df_con = conn.read(spreadsheet='https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/edit#gid=939232836')
                df_con['date'] = pd.to_datetime(df_con['date'])
                df_con['date_new'] = df_con['date'].dt.strftime('%Y-%m-%d')
                df_con['date_new'] = pd.to_datetime(df_con['date'], dayfirst = True).dt.strftime('%Y-%m-%d')
                df_con = df_con[df_con['date_new'] >= pd.to_datetime('today').strftime('%Y-%m-%d')]
                df_con = df_con.reset_index(drop=True)

                df_con_2 = conn.read(spreadsheet='https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/edit#gid=312814443')
                df_con_2['date'] = pd.to_datetime(df_con_2['date'])
                df_con_2['date_new'] = df_con_2['date'].dt.strftime('%Y-%m-%d')
                df_con_2['date_new'] = pd.to_datetime(df_con_2['date'], dayfirst = True).dt.strftime('%Y-%m-%d')
                df_con_2 = df_con_2[df_con_2['date_new'] >= pd.to_datetime('today').strftime('%Y-%m-%d')]
                df_con_2 = df_con_2.drop('Timestamp', axis=1)
                df_con_2 = df_con_2.reset_index(drop=True)
                df_con = pd.concat([df_con, df_con_2])

                df_con['Include?'] = False
                last_column = df_con.columns[-1]
                df_con = df_con[[last_column] + list(df_con.columns[:-1])]
                df_con.sort_values(by='date_new', ascending=True, inplace=True)
                df_con = df_con.reset_index(drop=True)
                st.markdown('##### Conferences')
                st.write('''
                Pick item(s) from the 'Include?' column.
                The selected items will be appear in the 'Items to be posted' table below.
                ''')
                df_con = st.data_editor(df_con)

                df_con = df_con[df_con['Include?']==True]
                df_con = df_con.reset_index(drop=True)
                df_con = df_con[['conference_name', 'organiser', 'link', 'venue', 'date_new']]
                df_con = df_con.rename(columns={'conference_name':'event_name'})

                df_cfp = conn.read(spreadsheet='https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/edit#gid=135096406') 
                df_cfp['deadline'] = pd.to_datetime(df_cfp['deadline'])
                df_cfp['deadline'] = df_cfp['deadline'].dt.strftime('%Y-%m-%d')
                df_cfp['deadline'] = pd.to_datetime(df_cfp['deadline'], dayfirst = True).dt.strftime('%Y-%m-%d')
                df_cfp = df_cfp[df_cfp['deadline'] >= pd.to_datetime('today').strftime('%Y-%m-%d')]

                df_cfp_2 = conn.read(spreadsheet='https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/edit#gid=1589739166') 
                df_cfp_2['deadline'] = pd.to_datetime(df_cfp_2['deadline'])
                df_cfp_2['deadline'] = df_cfp_2['deadline'].dt.strftime('%Y-%m-%d')
                df_cfp_2['deadline'] = pd.to_datetime(df_cfp_2['deadline'], dayfirst = True).dt.strftime('%Y-%m-%d')
                df_cfp_2 = df_cfp_2[df_cfp_2['deadline'] >= pd.to_datetime('today').strftime('%Y-%m-%d')]
                df_cfp_2 = df_cfp_2.drop('Timestamp', axis=1)
                df_cfp = pd.concat([df_cfp, df_cfp_2])

                df_cfp['Include?'] = False
                last_column = df_cfp.columns[-1]
                df_cfp = df_cfp[[last_column] + list(df_cfp.columns[:-1])]
                df_cfp.sort_values(by='deadline', ascending=True, inplace=True)
                df_cfp = df_cfp.reset_index(drop=True)
                df_cfp['venue'] = 'Call for Papers'
                st.markdown('##### Call for Papers')
                st.write('''
                Pick item(s) from the 'Include?' column.
                The selected items will be appear in the 'Items to be posted' table below.
                ''')
                df_cfp = st.data_editor(df_cfp)
                df_cfp = df_cfp[df_cfp['Include?']==True]
                df_cfp = df_cfp.rename(columns={'name':'event_name', 'deadline':'date_new'})
                df_cfp = df_cfp[['event_name', 'organiser', 'link', 'venue', 'date_new']]

                df_forms = pd.concat([df_forms, df_con, df_cfp])
                st.markdown('##### Items to be posted')
                df_forms

                post_events_bluesky = st.button('Post events on Bluesky')
                if post_events_bluesky:
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
            post_events()
        else:
                    @st.fragment
                    def item_monitoring_section():
                        st.subheader('Item monitoring section', anchor=False)

                        GITHUB_TOKEN = st.secrets["github_token"]
                        GITHUB_REPO = st.secrets["github_repo"]
                        DISMISSED_PATH = "dismissed.csv"

                        def load_dismissed():
                            url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{DISMISSED_PATH}"
                            headers = {"Authorization": f"token {GITHUB_TOKEN}"}
                            r = requests.get(url, headers=headers)
                            if r.status_code == 200:
                                import base64
                                content = base64.b64decode(r.json()["content"]).decode("utf-8")
                                from io import StringIO
                                df = pd.read_csv(StringIO(content))
                                return df, r.json()["sha"]
                            else:
                                return pd.DataFrame(columns=["DOI", "Title"]), None

                        def save_dismissed(df_dismissed, sha):
                            import base64
                            csv_content = df_dismissed.to_csv(index=False)
                            encoded = base64.b64encode(csv_content.encode("utf-8")).decode("utf-8")
                            url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{DISMISSED_PATH}"
                            headers = {"Authorization": f"token {GITHUB_TOKEN}"}
                            payload = {"message": "Update dismissed.csv via Streamlit admin", "content": encoded, "sha": sha}
                            r = requests.put(url, json=payload, headers=headers)
                            return r.status_code == 200

                        def display_with_dismiss(df, section_label):
                            if df.empty:
                                st.write("No items found.")
                                return
                            df = df.copy().reset_index(drop=True)
                            df.insert(0, "Dismiss?", False)
                            edited = st.data_editor(df, key=f"editor_{section_label}", use_container_width=True)
                            to_dismiss = edited[edited["Dismiss?"] == True]
                            if not to_dismiss.empty:
                                if st.button(f"Dismiss selected ({len(to_dismiss)}) from '{section_label}'", key=f"btn_{section_label}"):
                                    df_dismissed, sha = load_dismissed()
                                    new_rows = []
                                    for _, row in to_dismiss.iterrows():
                                        doi = str(row.get("DOI", "")).strip() if "DOI" in row else ""
                                        title = str(row.get("Title", "")).strip()
                                        already = ((df_dismissed["DOI"] == doi) | (df_dismissed["Title"] == title)).any()
                                        if not already:
                                            new_rows.append({"DOI": doi, "Title": title})
                                    if new_rows:
                                        df_dismissed = pd.concat([df_dismissed, pd.DataFrame(new_rows)], ignore_index=True)
                                        if save_dismissed(df_dismissed, sha):
                                            st.success(f"✅ {len(new_rows)} item(s) dismissed and saved.")
                                            st.rerun(scope="fragment")
                                        else:
                                            st.error("Failed to save to GitHub.")
                                    else:
                                        st.info("All selected items were already dismissed.")

                        # View / manage dismissed list
                        with st.expander("View / manage dismissed items"):
                            df_dismissed_view, sha_view = load_dismissed()
                            if df_dismissed_view.empty:
                                st.write("No dismissed items yet.")
                            else:
                                st.dataframe(df_dismissed_view, use_container_width=True)
                                st.caption(f"{len(df_dismissed_view)} item(s) permanently dismissed.")
                                if st.button("Clear all dismissed items"):
                                    empty = pd.DataFrame(columns=["DOI", "Title"])
                                    if save_dismissed(empty, sha_view):
                                        st.success("Dismissed list cleared.")
                                        st.rerun(scope="fragment")

                        # Main monitoring scan
                        if st.button("Run item monitoring"):
                            with st.status("Scanning sources to find items...", expanded=True) as status:
                                api_links = [
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s33269604&sort=publication_year:desc&per_page=10',
                                    "https://api.openalex.org/works?filter=primary_location.source.id:s205284143&sort=publication_year:desc&per_page=10",
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s4210168073&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s2764506647&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s2764781490&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s93928036&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s962698607&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s199078552&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s145781505&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s120387555&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s161550498&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s164505828&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s99133842&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s4210219209&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s185196701&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s157188123&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s79519963&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s161027966&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s4210201145&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s2764954702&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s200077084&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s27717133&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s4210214688&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s112911512&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s131264395&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s154084123&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s103350616&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s17185278&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s21016770&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s41746314&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s56601287&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s143110675&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s106532728&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s67329160&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s49917718&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s8593340&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s161552967&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s141724154&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s53578506&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s4210184262&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s4210236978&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s120889147&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?filter=primary_location.source.id:s86954274&sort=publication_year:desc&per_page=10',
                                    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s117224066&sort=publication_year:desc',
                                    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s160097506&sort=publication_year:desc',
                                    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s175405714&sort=publication_year:desc',
                                    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s84944781&sort=publication_year:desc',
                                    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s154337186&sort=publication_year:desc',
                                    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s156235965&sort=publication_year:desc',
                                    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s68909633&sort=publication_year:desc',
                                    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s42104779&sort=publication_year:desc',
                                    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s2764513295&sort=publication_year:desc',
                                    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s82119083&sort=publication_year:desc',
                                    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s129176075&sort=publication_year:desc',
                                    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s2764608241&sort=publication_year:desc',
                                    'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s2735957470&sort=publication_year:desc',
                                    'https://api.openalex.org/works?page=1&filter=primary_topic.id:t12572&sort=publication_year:desc',
                                    'https://api.openalex.org/works?page=1&filter=concepts.id:c558872910&sort=publication_year:desc',
                                    'https://api.openalex.org/works?page=1&filter=concepts.id:c173127888&sort=publication_year:desc',
                                ]

                                journals_with_filtered_items = [
                                    'The Historical Journal', 'Journal of Policing, Intelligence and Counter Terrorism', 'Cold War History', 'RUSI Journal',
                                    'Journal of Strategic Studies', 'War in History', 'International History Review', 'Journal of Contemporary History',
                                    'Middle Eastern Studies', 'Diplomacy & Statecraft', 'The international journal of intelligence, security, and public affairs',
                                    'Cryptologia', 'The Journal of Slavic Military Studies', 'International Affairs', 'Political Science Quarterly',
                                    'Journal of intelligence, conflict and warfare', 'The Journal of Conflict Studies', 'Journal of Cold War Studies', 'Survival',
                                    'Security and Defence Quarterly', 'The Journal of Imperial and Commonwealth History', 'Review of International Studies', 'Diplomatic History',
                                    'Cambridge Review of International Affairs', 'Public Policy and Administration', 'Armed Forces & Society', 'Studies in Conflict & Terrorism',
                                    'The English Historical Review', 'World Politics', 'Israel Affairs', 'Australian Journal of International Affairs', 'Contemporary British History',
                                    'The Historian', 'The British Journal of Politics and International Relations', 'Terrorism and Political Violence', "Mariner's Mirror",
                                    'Small Wars & Insurgencies', 'Journal of Cyber Policy', 'South Asia:Journal of South Asian Studies', 'International Journal', 'German Law Journal',
                                    'American Journal of International Law', 'European Journal of International Law', 'Human Rights Law Review', 'Leiden Journal of International Law',
                                    'International & Comparative Law Quarterly', 'Journal of Conflict and Security Law', 'Journal of International Dispute Settlement', 'Security and Human Rights',
                                    'Modern Law Review', 'International Theory', 'Michigan Journal of International Law', 'Journal of Global Security Studies', 'Intelligence Studies and Analysis in Modern Context'
                                ]

                                keywords = [
                                    'intelligence', 'spy', 'counterintelligence', 'espionage', 'covert', 'signal', 'sigint', 'humint', 'decipher', 'cryptanalysis',
                                    'spying', 'spies', 'surveillance', 'targeted killing', 'cyberespionage', ' cia ', 'rendition', ' mi6 ', ' mi5 ', ' sis ', 'security service',
                                    'central intelligence'
                                ]

                                import time
                                dfs = []
                                for api_link in api_links:
                                    response = requests.get(api_link)
                                    if response.status_code == 200:
                                        results = response.json().get('results', [])
                                        titles, dois, publication_dates, dois_without_https, journals = [], [], [], [], []
                                        today = datetime.today().date()
                                        for result in results:
                                            if not result:
                                                continue
                                            pub_date_str = result.get('publication_date')
                                            if not pub_date_str:
                                                continue
                                            try:
                                                pub_date = datetime.strptime(pub_date_str, '%Y-%m-%d').date()
                                            except ValueError:
                                                continue
                                            if today - pub_date <= timedelta(days=90):
                                                title = result.get('title')
                                                if title and any(kw in title.lower() for kw in keywords):
                                                    titles.append(title)
                                                    dois.append(result.get('doi', 'Unknown'))
                                                    publication_dates.append(pub_date_str)
                                                    doi_value = result.get('ids', {}).get('doi', 'Unknown')
                                                    dois_without_https.append(doi_value.split("https://doi.org/")[-1] if doi_value != 'Unknown' else 'Unknown')
                                                    source = (result.get('primary_location') or {}).get('source')
                                                    journals.append(source.get('display_name', 'Unknown') if source else 'Unknown')
                                        if titles:
                                            dfs.append(pd.DataFrame({'Title': titles, 'Link': dois, 'Publication Date': publication_dates, 'DOI': dois_without_https, 'Journal': journals}))

                            # DEBUG - remove after testing
                            st.write(f"**Debug: {len(dfs)} dataframes fetched from {len(api_links)} API links**")
                            if dfs:
                                debug_df = pd.concat(dfs, ignore_index=True)
                                st.write(f"Total raw articles before dedup/filtering: {len(debug_df)}")
                                st.dataframe(debug_df.head(20))
                            else:
                                # Show status codes to diagnose
                                st.write("No data returned. Testing first API link directly:")
                                test_response = requests.get(api_links[0])
                                st.write(f"Status code: {test_response.status_code}")
                                if test_response.status_code == 200:
                                    test_data = test_response.json()
                                    st.write(f"Results returned: {len(test_data.get('results', []))}")
                                    st.write("First result:")
                                    st.json(test_data.get('results', [{}])[0])

                                if not dfs:
                                    st.warning("No articles found.")
                                    status.update(label="Search complete!", state="complete", expanded=True)
                                    return

                                final_df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset='Link')
                                filtered_final_df = pd.concat([
                                    final_df[~final_df['Journal'].isin(journals_with_filtered_items)],
                                    final_df[final_df['Journal'].isin(journals_with_filtered_items)]
                                ], ignore_index=True)

                                df_dedup = pd.read_csv('all_items.csv')
                                df_dois = df_dedup.dropna(subset=['DOI'])[['DOI']].copy()
                                filtered_final_df['DOI'] = filtered_final_df['DOI'].str.lower()
                                df_dois['DOI'] = df_dois['DOI'].str.lower()
                                merged = pd.merge(filtered_final_df, df_dois, on='DOI', how='left', indicator=True)
                                items_not_in_df2 = merged[merged['_merge'] == 'left_only'].drop('_merge', axis=1)
                                items_not_in_df2 = items_not_in_df2[~items_not_in_df2['Title'].str.contains('notwantedwordshere', case=False)]

                                df_titles = df_dedup.dropna(subset=['Title'])[['Title']].copy()
                                merged2 = pd.merge(items_not_in_df2, df_titles, on='Title', how='left', indicator=True)
                                items_not_in_df3 = merged2[merged2['_merge'] == 'left_only'].drop('_merge', axis=1).sort_values('Publication Date', ascending=False).reset_index(drop=True)



                                df_dismissed, sha = load_dismissed()
                                if not df_dismissed.empty:
                                    dismissed_dois = set(df_dismissed['DOI'].str.lower().dropna())
                                    dismissed_titles = set(df_dismissed['Title'].str.lower().dropna())
                                    items_not_in_df3 = items_not_in_df3[
                                        ~items_not_in_df3['DOI'].str.lower().isin(dismissed_dois) &
                                        ~items_not_in_df3['Title'].str.lower().isin(dismissed_titles)
                                    ].reset_index(drop=True)
                                else:
                                    dismissed_titles = set()

                                items_not_in_df3['Publication Date'] = pd.to_datetime(items_not_in_df3['Publication Date'])
                                current_date = datetime.now()
                                future_df = items_not_in_df3[items_not_in_df3['Publication Date'] >= current_date].reset_index(drop=True)
                                last_30_days_df = items_not_in_df3[
                                    (items_not_in_df3['Publication Date'] <= current_date) &
                                    (items_not_in_df3['Publication Date'] >= current_date - timedelta(days=30))
                                ].reset_index(drop=True)

                                df_item_titles = df_dedup.dropna(subset=['Title'])[['Title']].copy()
                                from rss_feed import df_podcast, df_magazines
                                new_podcasts = pd.merge(df_podcast, df_item_titles, on='Title', how='left', indicator=True)
                                new_podcasts = new_podcasts[new_podcasts['_merge'] == 'left_only'].drop('_merge', axis=1).reset_index(drop=True)
                                if dismissed_titles:
                                    new_podcasts = new_podcasts[~new_podcasts['Title'].str.lower().isin(dismissed_titles)].reset_index(drop=True)

                                new_magazines = pd.merge(df_magazines, df_item_titles, on='Title', how='left', indicator=True)
                                new_magazines = new_magazines[new_magazines['_merge'] == 'left_only'].drop('_merge', axis=1).reset_index(drop=True)
                                if dismissed_titles:
                                    new_magazines = new_magazines[~new_magazines['Title'].str.lower().isin(dismissed_titles)].reset_index(drop=True)

                                def fetch_rss_data(url, label):
                                    r = requests.get(url)
                                    root = ET.fromstring(r.content)
                                    data = []
                                    for item in root.findall('.//item')[1:]:
                                        data.append({'title': item.find('title').text, 'link': item.find('link').text, 'label': label, 'pubDate': item.find('pubDate').text})
                                    return data

                                all_data = []
                                for feed in [{"url": "https://www.aspistrategist.org.au/feed/", "label": "Australian Strategic Policy Institute"}]:
                                    all_data.extend(fetch_rss_data(feed["url"], feed["label"]))
                                df_other = pd.DataFrame(all_data)
                                df_other = df_other[df_other['title'].str.contains('|'.join(["intelligence", "espionage", "spy", "oversight"]), case=False, na=False)].reset_index(drop=True)
                                df_other = df_other.rename(columns={'title': 'Title'})
                                df_other['Title'] = df_other['Title'].str.upper()
                                df_titles['Title'] = df_titles['Title'].str.upper()

                                def find_similar_title(title, titles, threshold=80):
                                    for t in titles:
                                        if fuzz.ratio(title, t) >= threshold:
                                            return t
                                    return None

                                df_other['Similar_Title'] = df_other['Title'].apply(lambda x: find_similar_title(x, df_titles['Title'], threshold=80))
                                df_not = df_other.merge(df_titles[['Title']], left_on='Similar_Title', right_on='Title', how='left', indicator=True)
                                df_not = df_not[df_not['_merge'] == 'left_only'].drop(['_merge', 'Similar_Title', 'Title_y'], axis=1).reset_index(drop=True)
                                df_not = df_not.rename(columns={'Title_x': 'Title'})
                                if dismissed_titles:
                                    df_not = df_not[~df_not['Title'].str.lower().isin(dismissed_titles)].reset_index(drop=True)
                                # Save all results to session state
                                st.session_state['monitoring_results'] = {
                                    'future_df': future_df,
                                    'last_30_days_df': last_30_days_df,
                                    'new_podcasts': new_podcasts,
                                    'new_magazines': new_magazines,
                                    'df_not': df_not,
                                    'df_dismissed': df_dismissed,
                                }
                                status.update(label="Search complete!", state="complete", expanded=True)

                        # Display results from session state — persists across fragment reruns from checkbox ticks
                        if 'monitoring_results' in st.session_state:
                            r = st.session_state['monitoring_results']

                            st.write('**Journal articles (future publications)**')
                            display_with_dismiss(r['future_df'], "Future publications")

                            st.write('**Journal articles (published in last 30 days)**')
                            display_with_dismiss(r['last_30_days_df'], "Last 30 days")

                            st.write('**Podcasts**')
                            if r['new_podcasts'].empty:
                                st.write('No new podcasts!')
                            else:
                                display_with_dismiss(r['new_podcasts'].sort_values('PubDate', ascending=False), "Podcasts")

                            st.write('**Magazine articles**')
                            if r['new_magazines'].empty:
                                st.write('No new magazine articles!')
                            else:
                                display_with_dismiss(r['new_magazines'].sort_values('PubDate', ascending=False), "Magazines")

                            st.write('**Other resources**')
                            display_with_dismiss(r['df_not'], "Other resources")

                    item_monitoring_section()
st.write('---')

display_custom_license()
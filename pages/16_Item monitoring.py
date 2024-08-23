# from pyzotero import zotero
# import pandas as pd
# import streamlit as st
# from IPython.display import HTML
# import streamlit.components.v1 as components
# import numpy as np
# # import altair as alt
# # from pandas.io.json import json_normalize
# from datetime import date, timedelta  
# from datetime import datetime
# import datetime 
# import datetime as dt
# # import plotly.express as px
# # import numpy as np
# import re
# # from fpdf import FPDF
# # import base64
# from sidebar_content import sidebar_content
# import requests
# from rss_feed import df_podcast, df_magazines
# from events import evens_conferences
# import xml.etree.ElementTree as ET
# from fuzzywuzzy import fuzz

# from atproto import Client
# import os
# from bs4 import BeautifulSoup
# from grapheme import length as grapheme_length
# import pytz
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


st.set_page_config(layout = "wide", 
                    page_title='Intelligence studies network',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 

st.title("Intelligence studies network")
st.header('Item monitoring')

image = 'https://images.pexels.com/photos/315918/pexels-photo-315918.png'
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


with st.sidebar:

    sidebar_content()

### Bluesky posting functions start here
client = Client(base_url='https://bsky.social')
bluesky_password = st.secrets["bluesky_password"]
client.login('intelbase.bsky.social', bluesky_password)

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
### Bluesky posting functions end here

password_input = st.text_input("Enter the password to access admin dashboard:", type="password")

if not password_input:
    st.info('Enter the password')
else:
    if password_input == st.secrets['item_monitoring_password']:
        st.success('Wellcome to the admin dashboard')

        admin_task = st.radio('Select an option', ['Item monitoring', 'Post publications', 'Post events'])

        if admin_task=='Post publications':
            @st.experimental_fragment
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
                
                item_header = st.radio('Select a header', ['New addition', 'Recently published', 'Custom'])
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
            @st.experimental_fragment
            def post_events():
                st.subheader('Post events on Bluesky', anchor=False)
                conn = st.connection("gsheets", type=GSheetsConnection)
                df_forms = conn.read(spreadsheet='https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/edit#gid=1941981997')
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
                df_con['Include?'] = False
                last_column = df_con.columns[-1]
                df_con = df_con[[last_column] + list(df_con.columns[:-1])]
                df_con.sort_values(by='date_new', ascending=True, inplace=True)
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
                df_cfp = df_cfp.reset_index(drop=True)
                df_cfp['Include?'] = False
                last_column = df_cfp.columns[-1]
                df_cfp = df_cfp[[last_column] + list(df_cfp.columns[:-1])]
                df_cfp.sort_values(by='deadline', ascending=True, inplace=True)
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
            st.subheader('Item monitoring section', anchor=False)
            item_monitoring = st.button("Item monitoring")
            if item_monitoring:
                st.write('The following items are not in the library yet. Book reviews will not be included!')
                with st.status("Scanning sources to find items...", expanded=True) as status:
                # with st.spinner('Scanning sources to find items...'): 
                    api_links = [
                        'https://api.openalex.org/works?filter=primary_location.source.id:s33269604&sort=publication_year:desc&per_page=10', #IJIC
                        "https://api.openalex.org/works?filter=primary_location.source.id:s205284143&sort=publication_year:desc&per_page=10", #The Historical Journal
                        'https://api.openalex.org/works?filter=primary_location.source.id:s4210168073&sort=publication_year:desc&per_page=10', #INS
                        'https://api.openalex.org/works?filter=primary_location.source.id:s2764506647&sort=publication_year:desc&per_page=10', #Journal of Intelligence History
                        'https://api.openalex.org/works?filter=primary_location.source.id:s2764781490&sort=publication_year:desc&per_page=10', #Journal of Policing, Intelligence and Counter Terrorism
                        'https://api.openalex.org/works?filter=primary_location.source.id:s93928036&sort=publication_year:desc&per_page=10', #Cold War History
                        'https://api.openalex.org/works?filter=primary_location.source.id:s962698607&sort=publication_year:desc&per_page=10', #RUSI Journal
                        'https://api.openalex.org/works?filter=primary_location.source.id:s199078552&sort=publication_year:desc&per_page=10', #Journal of Strategic Studies
                        'https://api.openalex.org/works?filter=primary_location.source.id:s145781505&sort=publication_year:desc&per_page=10', #War in History
                        'https://api.openalex.org/works?filter=primary_location.source.id:s120387555&sort=publication_year:desc&per_page=10', #International History Review
                        'https://api.openalex.org/works?filter=primary_location.source.id:s161550498&sort=publication_year:desc&per_page=10', #Journal of Contemporary History
                        'https://api.openalex.org/works?filter=primary_location.source.id:s164505828&sort=publication_year:desc&per_page=10', #Middle Eastern Studies
                        'https://api.openalex.org/works?filter=primary_location.source.id:s99133842&sort=publication_year:desc&per_page=10', #Diplomacy & Statecraft
                        'https://api.openalex.org/works?filter=primary_location.source.id:s4210219209&sort=publication_year:desc&per_page=10', #The international journal of intelligence, security, and public affair
                        'https://api.openalex.org/works?filter=primary_location.source.id:s185196701&sort=publication_year:desc&per_page=10',#Cryptologia
                        'https://api.openalex.org/works?filter=primary_location.source.id:s157188123&sort=publication_year:desc&per_page=10', #The Journal of Slavic Military Studies
                        'https://api.openalex.org/works?filter=primary_location.source.id:s79519963&sort=publication_year:desc&per_page=10',#International Affairs
                        'https://api.openalex.org/works?filter=primary_location.source.id:s161027966&sort=publication_year:desc&per_page=10', #Political Science Quarterly
                        'https://api.openalex.org/works?filter=primary_location.source.id:s4210201145&sort=publication_year:desc&per_page=10', #Journal of intelligence, conflict and warfare
                        'https://api.openalex.org/works?filter=primary_location.source.id:s2764954702&sort=publication_year:desc&per_page=10', #The Journal of Conflict Studies
                        'https://api.openalex.org/works?filter=primary_location.source.id:s200077084&sort=publication_year:desc&per_page=10', #Journal of Cold War Studies
                        'https://api.openalex.org/works?filter=primary_location.source.id:s27717133&sort=publication_year:desc&per_page=10', #Survival
                        'https://api.openalex.org/works?filter=primary_location.source.id:s4210214688&sort=publication_year:desc&per_page=10', #Security and Defence Quarterly
                        'https://api.openalex.org/works?filter=primary_location.source.id:s112911512&sort=publication_year:desc&per_page=10', #The Journal of Imperial and Commonwealth History
                        'https://api.openalex.org/works?filter=primary_location.source.id:s131264395&sort=publication_year:desc&per_page=10', #Review of International Studies
                        'https://api.openalex.org/works?filter=primary_location.source.id:s154084123&sort=publication_year:desc&per_page=10', #Diplomatic History
                        'https://api.openalex.org/works?filter=primary_location.source.id:s103350616&sort=publication_year:desc&per_page=10', #Cambridge Review of International Affairs
                        'https://api.openalex.org/works?filter=primary_location.source.id:s17185278&sort=publication_year:desc&per_page=10', #Public Policy and Administration
                        'https://api.openalex.org/works?filter=primary_location.source.id:s21016770&sort=publication_year:desc&per_page=10', #Armed Forces & Society
                        'https://api.openalex.org/works?filter=primary_location.source.id:s41746314&sort=publication_year:desc&per_page=10', #Studies in Conflict & Terrorism
                        'https://api.openalex.org/works?filter=primary_location.source.id:s56601287&sort=publication_year:desc&per_page=10', #The English Historical Review
                        'https://api.openalex.org/works?filter=primary_location.source.id:s143110675&sort=publication_year:desc&per_page=10', #World Politics
                        'https://api.openalex.org/works?filter=primary_location.source.id:s106532728&sort=publication_year:desc&per_page=10', #Israel Affairs
                        'https://api.openalex.org/works?filter=primary_location.source.id:s67329160&sort=publication_year:desc&per_page=10', #Australian Journal of International Affairs
                        'https://api.openalex.org/works?filter=primary_location.source.id:s49917718&sort=publication_year:desc&per_page=10', #Contemporary British History
                        'https://api.openalex.org/works?filter=primary_location.source.id:s8593340&sort=publication_year:desc&per_page=10', #The Historian
                        'https://api.openalex.org/works?filter=primary_location.source.id:s161552967&sort=publication_year:desc&per_page=10', #The British Journal of Politics and International Relations
                        'https://api.openalex.org/works?filter=primary_location.source.id:s141724154&sort=publication_year:desc&per_page=10', #Terrorism and Political Violence
                        'https://api.openalex.org/works?filter=primary_location.source.id:s53578506&sort=publication_year:desc&per_page=10', #Mariner's Mirror
                        'https://api.openalex.org/works?filter=primary_location.source.id:s4210184262&sort=publication_year:desc&per_page=10', #Small Wars & Insurgencies
                        'https://api.openalex.org/works?filter=primary_location.source.id:s4210236978&sort=publication_year:desc&per_page=10', #Journal of Cyber Policy
                        'https://api.openalex.org/works?filter=primary_location.source.id:s120889147&sort=publication_year:desc&per_page=10', #South Asia:Journal of South Asian Studies
                        'https://api.openalex.org/works?filter=primary_location.source.id:s86954274&sort=publication_year:desc&per_page=10', #International Journal
                        'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s117224066&sort=publication_year:desc', #German Law Journal
                        'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s160097506&sort=publication_year:desc', #American Journal of International Law
                        'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s175405714&sort=publication_year:desc', #European Journal of International Law
                        'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s84944781&sort=publication_year:desc', #Human Rights Law Review
                        'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s154337186&sort=publication_year:desc', #Leiden Journal of International Law
                        'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s156235965&sort=publication_year:desc', #International & Comparative Law Quarterl
                        'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s68909633&sort=publication_year:desc', #Journal of Conflict and Security Law
                        'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s42104779&sort=publication_year:desc', #Journal of International Dispute Settlement
                        'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s2764513295&sort=publication_year:desc', #Security and Human Rights
                        'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s82119083&sort=publication_year:desc', #Modern Law Review
                        'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s129176075&sort=publication_year:desc', #International Theory
                        'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s2764608241&sort=publication_year:desc', #Michigan Journal of International Law
                        'https://api.openalex.org/works?page=1&filter=primary_location.source.id:s2735957470&sort=publication_year:desc', #Journal of Global Security Studies
                        'https://api.openalex.org/works?page=1&filter=primary_topic.id:t12572&sort=publication_year:desc', #Intelligence Studies and Analysis in Modern Context
                        'https://api.openalex.org/works?page=1&filter=concepts.id:c558872910&sort=publication_year:desc', #ConceptEspionage
                        'https://api.openalex.org/works?page=1&filter=concepts.id:c173127888&sort=publication_year:desc', #ConceptCounterintelligence

                        # Add more API links here
                    ]

                    # Define journals to include filtered items
                    journals_with_filtered_items = [
                        'The Historical Journal', 'Journal of Policing, Intelligence and Counter Terrorism', 'Cold War History', 'RUSI Journal',
                        'Journal of Strategic Studies', 'War in History', 'International History Review','Journal of Contemporary History', 
                        'Middle Eastern Studies', 'Diplomacy & Statecraft', 'The international journal of intelligence, security, and public affairs',
                        'Cryptologia', 'The Journal of Slavic Military Studies', 'International Affairs', 'Political Science Quarterly',
                        'Journal of intelligence, conflict and warfare', 'The Journal of Conflict Studies','Journal of Cold War Studies', 'Survival',
                        'Security and Defence Quarterly', 'The Journal of Imperial and Commonwealth History', 'Review of International Studies', 'Diplomatic History',
                        'Cambridge Review of International Affairs', 'Public Policy and Administration', 'Armed Forces & Society', 'Studies in Conflict & Terrorism',
                        'The English Historical Review', 'World Politics', 'Israel Affairs', 'Australian Journal of International Affairs', 'Contemporary British History',
                        'The Historian', 'The British Journal of Politics and International Relations', 'Terrorism and Political Violence', "Mariner's Mirror",
                        'Small Wars & Insurgencies', 'Journal of Cyber Policy', 'South Asia:Journal of South Asian Studies', 'International Journal', 'German Law Journal',
                        'American Journal of International Law', 'European Journal of International Law', 'Human Rights Law Review', 'Leiden Journal of International Law',
                        'International & Comparative Law Quarterly', 'Journal of Conflict and Security Law', 'Journal of International Dispute Settlement', 'Security and Human Rights',
                        'Modern Law Review', 'International Theory', 'Michigan Journal of International Law', 'Journal of Global Security Studies', 'Intelligence Studies and Analysis in Modern Context'
                        ]

                    # Define keywords for filtering
                    keywords = [
                        'intelligence', 'spy', 'counterintelligence', 'espionage', 'covert', 'signal', 'sigint', 'humint', 'decipher', 'cryptanalysis',
                        'spying', 'spies', 'surveillance', 'targeted killing', 'cyberespionage', ' cia ', 'rendition', ' mi6 ', ' mi5 ', ' sis ', 'security service',
                        'central intelligence'
                    ]

                    dfs = []

                    for api_link in api_links:
                        response = requests.get(api_link)

                        if response.status_code == 200:
                            data = response.json()
                            results = data.get('results', [])

                            titles = []
                            dois = []
                            publication_dates = []
                            dois_without_https = []
                            journals = []

                            today = datetime.today().date()

                            for result in results:
                                if result is None:
                                    continue
                                
                                pub_date_str = result.get('publication_date')
                                if pub_date_str is None:
                                    continue

                                try:
                                    pub_date = datetime.strptime(pub_date_str, '%Y-%m-%d').date()
                                except ValueError:
                                    continue  # Skip this result if the date is not in the expected format

                                if today - pub_date <= timedelta(days=90):
                                    title = result.get('title')
                                    
                                    if title is not None and any(keyword in title.lower() for keyword in keywords):
                                        titles.append(title)
                                        dois.append(result.get('doi', 'Unknown'))
                                        publication_dates.append(pub_date_str)
                                        
                                        # Ensure 'ids' and 'doi' are present before splitting
                                        ids = result.get('ids', {})
                                        doi_value = ids.get('doi', 'Unknown')
                                        if doi_value != 'Unknown':
                                            dois_without_https.append(doi_value.split("https://doi.org/")[-1])
                                        else:
                                            dois_without_https.append('Unknown')

                                        # Safely navigate through nested dictionaries using get
                                        primary_location = result.get('primary_location', {})
                                        source = primary_location.get('source')
                                        if source:
                                            journal_name = source.get('display_name', 'Unknown')
                                        else:
                                            journal_name = 'Unknown'

                                        journals.append(journal_name)

                            if titles:  # Ensure DataFrame creation only if there are titles
                                df = pd.DataFrame({
                                    'Title': titles,
                                    'Link': dois,
                                    'Publication Date': publication_dates,
                                    'DOI': dois_without_https,
                                    'Journal': journals,
                                })

                                dfs.append(df)

                    # Combine all DataFrames in dfs list into a single DataFrame
                    if dfs:
                        final_df = pd.concat(dfs, ignore_index=True)
                    else:
                        final_df = pd.DataFrame()  # Create an empty DataFrame if dfs is empty

                        # else:
                        #     print(f"Failed to fetch data from the API: {api_link}")

                    final_df = pd.concat(dfs, ignore_index=True)
                    final_df = final_df.drop_duplicates(subset='Link')

                    historical_journal_filtered = final_df[final_df['Journal'].isin(journals_with_filtered_items)]

                    other_journals = final_df[~final_df['Journal'].isin(journals_with_filtered_items)]

                    filtered_final_df = pd.concat([other_journals, historical_journal_filtered], ignore_index=True)

                    ## DOI based filtering
                    df_dedup = pd.read_csv('all_items.csv')
                    df_dois = df_dedup.copy() 
                    df_dois.dropna(subset=['DOI'], inplace=True) 
                    column_to_keep = 'DOI'
                    df_dois = df_dois[[column_to_keep]]
                    df_dois = df_dois.reset_index(drop=True) 

                    merged_df = pd.merge(filtered_final_df, df_dois[['DOI']], on='DOI', how='left', indicator=True)
                    items_not_in_df2 = merged_df[merged_df['_merge'] == 'left_only']
                    items_not_in_df2.drop('_merge', axis=1, inplace=True)

                    words_to_exclude = ['notwantedwordshere'] #'paperback', 'hardback']

                    mask = ~items_not_in_df2['Title'].str.contains('|'.join(words_to_exclude), case=False)
                    items_not_in_df2 = items_not_in_df2[mask]
                    items_not_in_df2 = items_not_in_df2.reset_index(drop=True)
                    # st.write('**Journal articles (DOI based filtering)**')
                    # row_nu = len(items_not_in_df2.index)
                    # if row_nu == 0:
                    #     st.write('No new podcast published!')
                    # else:
                    #     items_not_in_df2 = items_not_in_df2.sort_values(by=['Publication Date'], ascending=False)
                    #     items_not_in_df2 = items_not_in_df2.reset_index(drop=True)
                    #     items_not_in_df2

                    ## Title based filtering
                    df_titles = df_dedup.copy()
                    df_titles.dropna(subset=['Title'], inplace=True)
                    column_to_keep = 'Title'
                    df_titles = df_titles[[column_to_keep]]
                    df_titles = df_titles.reset_index(drop=True)

                    merged_df_2 = pd.merge(items_not_in_df2, df_titles[['Title']], on='Title', how='left', indicator=True)
                    items_not_in_df3 = merged_df_2[merged_df_2['_merge'] == 'left_only']
                    items_not_in_df3.drop('_merge', axis=1, inplace=True)
                    items_not_in_df3 = items_not_in_df3.sort_values(by=['Publication Date'], ascending=False)
                    items_not_in_df3 = items_not_in_df3.reset_index(drop=True)


                    st.write('**Journal articles (future publications)**')
                    ## Future publications
                    items_not_in_df2['Publication Date'] = pd.to_datetime(items_not_in_df2['Publication Date'])
                    current_date = datetime.now()
                    future_df = items_not_in_df2[items_not_in_df2['Publication Date']>=current_date]
                    future_df = future_df.reset_index(drop=True)
                    future_df

                    ## Published in the last 30 days
                    st.write('**Journal articles (published in last 30 days)**')
                    current_date = datetime.now()
                    date_30_days_ago = current_date - timedelta(days=30)
                    last_30_days_df = items_not_in_df2[(items_not_in_df2['Publication Date']<=current_date) & (items_not_in_df2['Publication Date']>=date_30_days_ago)]
                    last_30_days_df = last_30_days_df.reset_index(drop=True)
                    last_30_days_df

                    # merged_df = pd.merge(filtered_final_df, df_dois[['DOI']], on='DOI', how='left', indicator=True)
                    # items_not_in_df2 = merged_df[merged_df['_merge'] == 'left_only']
                    # items_not_in_df2.drop('_merge', axis=1, inplace=True)

                    # words_to_exclude = ['notwantedwordshere'] #'paperback', 'hardback']

                    # mask = ~items_not_in_df2['Title'].str.contains('|'.join(words_to_exclude), case=False)
                    # items_not_in_df2 = items_not_in_df2[mask]
                    # items_not_in_df2 = items_not_in_df2.reset_index(drop=True)
                    # st.write('**Journal articles**')
                    # row_nu = len(items_not_in_df2.index)
                    # if row_nu == 0:
                    #     st.write('No new podcast published!')
                    # else:
                    #     items_not_in_df2 = items_not_in_df2.sort_values(by=['Publication Date'], ascending=False)
                    #     items_not_in_df2 = items_not_in_df2.reset_index(drop=True)
                    #     items_not_in_df2

                    df_item_podcast = df_dedup.copy()
                    df_item_podcast.dropna(subset=['Title'], inplace=True)
                    column_to_keep = 'Title'
                    df_item_podcast = df_item_podcast[[column_to_keep]]
                    from rss_feed import df_podcast, df_magazines
                    df_podcast = pd.merge(df_podcast, df_item_podcast[['Title']], on='Title', how='left', indicator=True)
                    items_not_in_df_item_podcast = df_podcast[df_podcast['_merge'] == 'left_only']
                    items_not_in_df_item_podcast.drop('_merge', axis=1, inplace=True)
                    items_not_in_df_item_podcast = items_not_in_df_item_podcast.reset_index(drop=True)
                    st.write('**Podcasts**')
                    row_nu = len(items_not_in_df_item_podcast.index)
                    if row_nu == 0: 
                        st.write('No new podcast published!')
                    else:
                        items_not_in_df_item_podcast = items_not_in_df_item_podcast.sort_values(by=['PubDate'], ascending=False)
                        items_not_in_df_item_podcast

                    df_item_magazines = df_dedup.copy()
                    df_item_magazines.dropna(subset=['Title'], inplace=True)
                    column_to_keep = 'Title'
                    df_item_magazines = df_item_magazines[[column_to_keep]]
                    df_magazines = pd.merge(df_magazines, df_item_magazines[['Title']], on='Title', how='left', indicator=True)
                    items_not_in_df_item_magazines = df_magazines[df_magazines['_merge'] == 'left_only']
                    items_not_in_df_item_magazines.drop('_merge', axis=1, inplace=True)
                    items_not_in_df_item_magazines = items_not_in_df_item_magazines.reset_index(drop=True)
                    st.write('**Magazine articles**')
                    row_nu = len(items_not_in_df_item_magazines.index)
                    if row_nu == 0:
                        st.write('No new magazine article published!')
                    else:
                        items_not_in_df_item_magazines = items_not_in_df_item_magazines.sort_values(by=['PubDate'], ascending=False)
                        items_not_in_df_item_magazines        
                    status.update(label="Search complete!", state="complete", expanded=True)


                    st.write('**Other resources**')

                    def fetch_rss_data(url, label):
                        response = requests.get(url)
                        rss_content = response.content
                        root = ET.fromstring(rss_content)
                        items = root.findall('.//item')[1:]
                        data = []
                        for item in items:
                            title = item.find('title').text
                            link = item.find('link').text
                            pub_date = item.find('pubDate').text  # Extracting the pubDate
                            data.append({'title': title, 'link': link, 'label': label, 'pubDate': pub_date})  # Adding pubDate to the data dictionary
                        return data

                    # URLs of the RSS feeds with their respective labels
                    feeds = [
                        {"url":"https://www.aspistrategist.org.au/feed/", "label":"Australian Strategic Policy Institute"}
                    ]

                    # Fetch and combine data from both RSS feeds
                    all_data = []
                    for feed in feeds:
                        all_data.extend(fetch_rss_data(feed["url"], feed["label"]))

                    # Create a DataFrame
                    df = pd.DataFrame(all_data)

                    # The rest of your code remains unchanged

                    words_to_filter = ["intelligence", "espionage", "spy", "oversight"]
                    pattern = '|'.join(words_to_filter)

                    df = df[df['title'].str.contains(pattern, case=False, na=False)].reset_index(drop=True)
                    df = df.rename(columns={'title':'Title'})
                    df['Title'] = df['Title'].str.upper()
                    df_titles['Title'] = df_titles['Title'].str.upper()

                    def find_similar_title(title, titles, threshold=80):
                        for t in titles:
                            similarity = fuzz.ratio(title, t)
                            if similarity >= threshold:
                                return t
                        return None

                    # Adding a column to df with the most similar title from df_titles
                    df['Similar_Title'] = df['Title'].apply(lambda x: find_similar_title(x, df_titles['Title'], threshold=80))

                    # Performing the merge based on the similar titles
                    df_not = df.merge(df_titles[['Title']], left_on='Similar_Title', right_on='Title', how='left', indicator=True)
                    df_not = df_not[df_not['_merge'] == 'left_only']
                    df_not.drop(['_merge', 'Similar_Title'], axis=1, inplace=True)
                    df_not = df_not.reset_index(drop=True)
                    df_not

    else:
        st.error('Incorrect passcode')
st.write('---')

components.html(
"""
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
© 2024 Yusuf Ozkan. All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
"""
)
from pyzotero import zotero
import os
# import tweepy as tw
import pandas as pd
import datetime
import json, sys
from datetime import date, timedelta  
import datetime
import plotly.express as px
import pycountry
import re
import pandas as pd
import requests
import time
from datetime import datetime, timedelta

## ZOTERO LIBRARY KEYS
library_id = '2514686'
library_type = 'group'
api_key = '' # api_key is only needed for private groups and libraries
zot = zotero.Zotero(library_id, library_type)

## RETRIEVING ITEMS FROM THE ZOTERO LIBRARY
items = zot.everything(zot.top())

df_zotero_id = pd.read_csv('zotero_citation_format.csv')
df_all = pd.read_csv('all_items.csv')
df_all = df_all[['Zotero link']]
df_all['zotero_item_key'] = df_all['Zotero link'].str.replace('https://www.zotero.org/groups/intelligence_bibliography/items/', '')
df_all = df_all.drop_duplicates()
df_not_zotero_id = df_all[~df_all['zotero_item_key'].isin(df_zotero_id['zotero_item_key'])]
df_not_zotero_id = df_not_zotero_id[['zotero_item_key']].reset_index(drop=True)

user_id = '2514686'

# Base URL for Zotero API
base_url = 'https://api.zotero.org'

# Initialize an empty string to accumulate bibliographies
all_bibliographies = ""

# List to store bibliographies 
bibliographies = []

# Iterate through each item key in the DataFrame
for item_key in df_not_zotero_id['zotero_item_key']:
    # Endpoint to get item bibliography
    endpoint = f'/groups/{user_id}/items/{item_key}'

    # Parameters for the request
    params = {
        'format': 'bib',
        'linkwrap': 1
    }

    # Make GET request to Zotero API
    response = requests.get(base_url + endpoint, params=params)

    # Check if request was successful
    if response.status_code == 200:
        bibliography = response.text.strip()  # Strip any leading/trailing whitespace
        bibliographies.append(bibliography)
        all_bibliographies += f'<p>{bibliography}</p><br><br>'  # Append bibliography with two newlines for separation
    else:
        error_message = f'Error fetching bibliography for item {item_key}: Status Code {response.status_code}'
        bibliographies.append(error_message)
        all_bibliographies += f'<p>{error_message}</p><br><br>'

# Add bibliographies to the original DataFrame
df_not_zotero_id['bibliography'] = bibliographies

df_zotero_id = pd.read_csv('zotero_citation_format.csv', index_col=False)
df_zotero_id = df_zotero_id.drop(columns={'Unnamed: 0'})
df_zotero_id = pd.concat([df_zotero_id, df_not_zotero_id]).reset_index(drop=True)

df_zotero_id.to_csv('zotero_citation_format.csv')
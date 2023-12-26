from pyzotero import zotero
import os
import tweepy as tw
import pandas as pd
import datetime
import json, sys
from datetime import date, timedelta  
import datetime
import plotly.express as px
import pycountry
import re
import pandas as pd

library_id = '2514686'
library_type = 'group'
api_key = '' # api_key is only needed for private groups and libraries
zot = zotero.Zotero(library_id, library_type)

items = zot.everything(zot.top())

data3=[]
columns3=['Title','Publication type', 'Link to publication', 'Abstract', 'Zotero link', 'Date published', 'FirstName2', 'Publisher', 'Journal', 'Date added', 'Col key']

for item in items:
    # Extracting various types of creators
    authors = ""
    if 'creators' in item['data']:
        authors_info = item['data']['creators']
        author_names = []
        for author in authors_info:
            creator_type = author.get('creatorType')
            valid_types = ['author', 'contributor', 'editor', 'guest', 'podcaster']
            
            if item['data']['itemType'] == 'bookSection' and creator_type == 'editor':
                continue  # Skip adding 'editor' for Book chapters
            
            if creator_type in valid_types:
                if 'firstName' in author and 'lastName' in author:
                    author_names.append(author['firstName'] + ' ' + author['lastName'])
                elif 'name' in author:
                    author_names.append(author['name'])
                
        authors = ', '.join(author_names)
        
    data3.append((
        item['data']['title'], 
        item['data']['itemType'], 
        item['data']['url'], 
        item['data']['abstractNote'], 
        item['links']['alternate']['href'],
        item['data'].get('date'),
        authors,  # Add the concatenated authors string
        item['data'].get('publisher'),
        item['data'].get('publicationTitle'),
        item['data']['dateAdded'],
        item['data']['collections']
        )) 
# pd.set_option('display.max_colwidth', None)

df = pd.DataFrame(data3, columns=columns3)

mapping_types = {
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
    'manuscript': 'Manuscript',
    'document': 'Document',
    'conferencePaper': 'Conference paper',
    'film': 'Film',
    'presentation': 'Presentation',
    'audioRecording':'Podcast'
}
df['Publication type'] = df['Publication type'].replace(mapping_types)

mapping_publisher = {
    'Taylor & Francis Group': 'Taylor and Francis',
    'Taylor and Francis': 'Taylor and Francis',
    'Taylor & Francis': 'Taylor and Francis',
    'Routledge': 'Routledge',
    'Routledge Handbooks Online': 'Routledge',
    'Praeger Security International': 'Praeger',
    'Praeger': 'Praeger'
}
df['Publisher'] = df['Publisher'].replace(mapping_publisher)

# df['Publisher'] = df['Publisher'].replace(['Taylor & Francis Group', 'Taylor and Francis', 'Taylor & Francis'], 'Taylor and Francis')
# df['Publisher'] = df['Publisher'].replace(['Routledge', 'Routledge Handbooks Online'], 'Routledge')
# df['Publisher'] = df['Publisher'].replace(['Praeger Security International', 'Praeger'], 'Praeger')

mapping_journal = {
    'International Journal of Intelligence and Counterintelligence': 'Intl Journal of Intelligence and Counterintelligence',
    'International Journal of Intelligence and CounterIntelligence': 'Intl Journal of Intelligence and Counterintelligence',
    'Intelligence and national security': 'Intelligence and National Security',
    'Intelligence and National Security': 'Intelligence and National Security',
    'Intelligence & National Security': 'Intelligence and National Security'
}

df['Journal'] = df['Journal'].replace(mapping_journal)

# df['Journal'] = df['Journal'].replace(['International Journal of Intelligence and Counterintelligence', 'International Journal of Intelligence and CounterIntelligence'], 'Intl Journal of Intelligence and Counterintelligence')
# df['Journal'] = df['Journal'].replace(['Intelligence and national security', 'Intelligence and National Security', 'Intelligence & National Security'], 'Intelligence and National Security')

df
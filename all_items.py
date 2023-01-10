from pyzotero import zotero
import os
import pandas as pd
import datetime
import json, sys
from datetime import date, timedelta  
import datetime
import plotly.express as px

library_id = '2514686'
library_type = 'group'
api_key = '' # api_key is only needed for private groups and libraries
zot = zotero.Zotero(library_id, library_type)

items = zot.everything(zot.top())

data3=[]
columns3=['Title','Publication type', 'Link to publication', 'Abstract', 'Zotero link', 'Date published', 'FirstName2', 'Publisher', 'Journal']

for item in items:
    data3.append((
        item['data']['title'], 
        item['data']['itemType'], 
        item['data']['url'], 
        item['data']['abstractNote'], 
        item['links']['alternate']['href'],
        item['data'].get('date'),
        item['data']['creators'],
        item['data'].get('publisher'),
        item['data'].get('publicationTitle')
        )) 
pd.set_option('display.max_colwidth', None)

df = pd.DataFrame(data3, columns=columns3)

df['Publication type'] = df['Publication type'].replace(['thesis'], 'Thesis')
df['Publication type'] = df['Publication type'].replace(['journalArticle'], 'Journal article')
df['Publication type'] = df['Publication type'].replace(['book'], 'Book')
df['Publication type'] = df['Publication type'].replace(['bookSection'], 'Book chapter')
df['Publication type'] = df['Publication type'].replace(['blogPost'], 'Blog post')
df['Publication type'] = df['Publication type'].replace(['videoRecording'], 'Video')
df['Publication type'] = df['Publication type'].replace(['podcast'], 'Podcast')
df['Publication type'] = df['Publication type'].replace(['magazineArticle'], 'Magazine article')
df['Publication type'] = df['Publication type'].replace(['webpage'], 'Webpage')
df['Publication type'] = df['Publication type'].replace(['newspaperArticle'], 'Newspaper article')
df['Publication type'] = df['Publication type'].replace(['report'], 'Report')
df['Publication type'] = df['Publication type'].replace(['forumPost'], 'Forum post')
df['Publication type'] = df['Publication type'].replace(['manuscript'], 'Manuscript')
df['Publication type'] = df['Publication type'].replace(['document'], 'Document')
df['Publication type'] = df['Publication type'].replace(['forumPost'], 'Forum post')
df['Publication type'] = df['Publication type'].replace(['conferencePaper'], 'Conference paper')
df['Publication type'] = df['Publication type'].replace(['film'], 'Film')
df['Publication type'] = df['Publication type'].replace(['presentation'], 'Presentation')

df.to_csv('all_items.csv')

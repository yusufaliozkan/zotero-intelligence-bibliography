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

### RETRIEVING ITEMS FROM ZOTERO LIBRARY

library_id = '2514686'
library_type = 'group'
api_key = '' # api_key is only needed for private groups and libraries
zot = zotero.Zotero(library_id, library_type)

items = zot.everything(zot.top())

data3=[]
columns3=['Title','Publication type', 'Link to publication', 'Abstract', 'Zotero link', 'Date published', 'FirstName2', 'Publisher', 'Journal', 'Date added', 'Col key', 'DOI', 'Book_title', 'Thesis_type', 'University']

for item in items:
    # Extracting various types of creators
    authors = ""
    if 'creators' in item['data']:
        authors_info = item['data']['creators']
        author_names = []
        for author in authors_info:
            creator_type = author.get('creatorType')
            valid_types = ['author', 'contributor', 'editor', 'guest', 'podcaster', 'presenter', 'translator', 'programmer']
            
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
        item['data']['collections'],
        item['data'].get('DOI'),
        item['data'].get('bookTitle'),
        item['data'].get('thesisType', ''),
        item['data'].get('university', '')
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
    'audioRecording':'Podcast',
    'preprint':'Preprint',
    'hearing':'Hearing',
    'computerProgram':'Computer program',
    'dataset':'Dataset'
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
    'International Journal of Intelligence and Counter Intelligence': 'Intl Journal of Intelligence and Counterintelligence',
    'Intelligence and national security': 'Intelligence and National Security',
    'Intelligence and National Security': 'Intelligence and National Security',
    'Intelligence & National Security': 'Intelligence and National Security'
}

df['Journal'] = df['Journal'].replace(mapping_journal)

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
# df['Journal'] = df['Journal'].replace(['International Journal of Intelligence and Counterintelligence', 'International Journal of Intelligence and CounterIntelligence'], 'Intl Journal of Intelligence and Counterintelligence')
# df['Journal'] = df['Journal'].replace(['Intelligence and national security', 'Intelligence and National Security', 'Intelligence & National Security'], 'Intelligence and National Security')


### CREATING DUPLICATED CSV FOR COLLECTIONS
def zotero_collections2(library_id, library_type):
    collections = zot.collections()
    data = [(item['data']['key'], item['data']['name'], item['meta']['numItems'], item['links']['alternate']['href']) for item in collections]
    df_collections = pd.DataFrame(data, columns=['Key', 'Name', 'Number', 'Link'])
    return df_collections
df_collections_2 = zotero_collections2(library_id, library_type)

def duplicate_rows_by_col_key(df, df_collections):
    # Duplicate rows based on 'Col key'
    duplicated_rows = []
    for index, row in df.iterrows():
        if isinstance(row['Col key'], list):
            for key in row['Col key']:
                new_row = row.copy()
                collection_info = df_collections[df_collections['Key'] == key]
                if not collection_info.empty:
                    new_row['Collection_Key'] = key
                    new_row['Collection_Name'] = collection_info.iloc[0]['Name']
                    new_row['Collection_Link'] = collection_info.iloc[0]['Link']
                    duplicated_rows.append(new_row)
        else:
            key = row['Col key']
            new_row = row.copy()
            collection_info = df_collections[df_collections['Key'] == key]
            if not collection_info.empty:
                new_row['Collection_Key'] = key
                new_row['Collection_Name'] = collection_info.iloc[0]['Name']
                new_row['Collection_Link'] = collection_info.iloc[0]['Link']
                duplicated_rows.append(new_row)

    # Create a new DataFrame with duplicated rows
    duplicated_df = pd.DataFrame(duplicated_rows)

    return duplicated_df

# Duplicating rows based on 'Col key' and collections information
duplicated_df = duplicate_rows_by_col_key(df, df_collections_2)



### EXTRACTING COUNTRY NAMES

# Dictionary to map non-proper country names to their proper names
country_map = {
    'british': 'UK',
    'great britain': 'UK',
    'UK' : 'UK', 
    'america' : 'United States',
    'United States of America' : 'United States',
    'Soviet Union': 'Russia', 
    'american' : 'United States',
    'United States' : 'United States',
    'russian' : 'Russia'
    # Add more mappings as needed
}

# Find the country names in the "title" column of the dataframe
found_countries = {}
for i, row in df.iterrows():
    title = str(row['Title']).lower()
    for country in pycountry.countries:
        name = country.name.lower()
        if name in title or (name + 's') in title:  # Check for singular and plural forms of country names
            proper_name = country.name
            found_countries[proper_name] = found_countries.get(proper_name, 0) + 1
    for non_proper, proper in country_map.items():
        if non_proper in title:
            found_countries[proper] = found_countries.get(proper, 0) + title.count(non_proper)

# Create a new dataframe containing the found countries and their counts
df_countries = pd.DataFrame({'Country': list(found_countries.keys()), 'Count': list(found_countries.values())})
df_countries.to_csv('countries.csv',index=False)


### NER ANALYSIS
import spacy
import nltk
import ast
from spacy.pipeline import EntityRecognizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Download the model if not present
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
ruler = nlp.add_pipe("entity_ruler")
patterns = [{"label": "ORG", "pattern": "MI6"}]
ruler.add_patterns(patterns)

def extract_entities(text):
    doc = nlp(text)
    orgs = []
    gpes = []
    people = []
    for entity in doc.ents:
        if entity.label_ == 'ORG':
            orgs.append(entity.text)
        elif entity.label_ == 'GPE':
            gpes.append(entity.text)
        elif entity.label_ == 'PERSON':
            people.append(entity.text)
    return pd.Series({'ORG': orgs, 'GPE': gpes, 'PERSON': people})

df_title = df[['Title']].copy()
df_title = df_title.rename(columns={'Title':'Text'})
df_abstract = df[['Abstract']].copy()
df_abstract = df_abstract.rename(columns={'Abstract':'Text'})
df_one = pd.concat([df_title, df_abstract], ignore_index=True)
df_one['Text'] = df_one['Text'].fillna('')

df_one = pd.concat([df_one[['Text']], df_one['Text'].apply(extract_entities)], axis=1)
df_one = df_one.explode('GPE').reset_index(drop=True)
df_one = df_one.explode('ORG').reset_index(drop=True)
df_one = df_one.explode('PERSON').reset_index(drop=True)

df_one_g = df_one.copy()
df_one_g = df_one[['Text', 'GPE']]
# df_one_g = df_one_g.fillna('')
df_one_g = df_one_g.drop_duplicates(subset=['Text', 'GPE'])

gpe_counts = df_one_g['GPE'].value_counts().reset_index()
gpe_counts.columns = ['GPE', 'count']

# pd.options.display.max_rows = None


mapping_locations = {
    'the United States': 'USA',
    'The United States': 'USA',
    'US': 'USA',
    'U.S.': 'USA',
    'United States' : 'USA',
    'America' : 'USA',
    'the United States of America' : 'USA',
    'Britain' : 'UK',
    'the United Kingdom': 'UK',
    'U.K.' : 'UK',
    'Global Britain' : 'UK',
    'United Kingdom' : 'UK', 
    'the Soviet Union' : 'Russia',
    'The Soviet Union' : 'Russia',
    'USSR' : 'Russia',
    'Ukraine - Perspective' : 'Ukraine',
    'Ukrainian' : 'Ukraine',
    'Great Britain' : 'UK',
    'Ottoman Empire' : 'Turkey'
}
gpe_counts['GPE'] =gpe_counts['GPE'].replace(mapping_locations)
gpe_counts = gpe_counts.groupby('GPE').sum().reset_index()
gpe_counts.sort_values('count', ascending=False, inplace=True)
gpe_counts = gpe_counts.reset_index(drop=True)

df_one_p = df_one.copy()
df_one_p = df_one[['Text', 'PERSON']]
# df_one_p = df_one_g.fillna('')
df_one_p = df_one_p.drop_duplicates(subset=['Text', 'PERSON'])

person_counts = df_one_p['PERSON'].value_counts().reset_index()
person_counts.columns = ['PERSON', 'count']

mapping_person = {
    'Putin' : 'Vladimir Putin',
    'Vladimir Putin' : 'Vladimir Putin',
    'Churchill' : 'Winston Churchill',
    'Hitler' : 'Adolf Hitler',
    'Biden' : 'Joe Biden',
    "John le Carré’s" : "John le Carré"
}

person_counts['PERSON'] =person_counts['PERSON'].replace(mapping_person)
person_counts = person_counts.groupby('PERSON').sum().reset_index()
person_counts.sort_values('count', ascending=False, inplace=True)

remove_person = ['MI6', 'Twitter', 'GRU'
          ]
person_counts = person_counts[~person_counts['PERSON'].isin(remove_person)]
person_counts = person_counts.reset_index(drop=True)

df_one_o = df_one.copy()
df_one_o = df_one[['Text', 'ORG']]
# df_one_p = df_one_g.fillna('')
df_one_o = df_one_o.drop_duplicates(subset=['Text', 'ORG'])

org_counts = df_one_o['ORG'].value_counts().reset_index()
org_counts.columns = ['ORG', 'count']

mapping_organisations = {
    'The British Secret Intelligence Service' : 'SIS',
    'the British Secret Intelligence Service' : 'SIS',
    'The Joint Intelligence Committee' : 'Joint Intelligence Committee',
    'the Joint Intelligence Committee' : 'Joint Intelligence Committee',
    'Joint Intelligence Committee' : 'Joint Intelligence Committee',
    'the Joint Intelligence Committee - History' : 'Joint Intelligence Committee',
    'Central Intelligence Agency' : 'CIA',
    'the Central Intelligence Agency' : 'CIA',
    'the Foreign Office' : 'Foreign Office',
    'Schar School' : 'Schar School of Policy and Government',
    'the Secret Intelligence Service' : 'SIS',
    "George Mason University's" : "George Mason University",
    'JIC' : 'Joint Intelligence Committee'
}
org_counts['ORG'] =org_counts['ORG'].replace(mapping_organisations)
org_counts = org_counts.groupby('ORG').sum().reset_index()
org_counts.sort_values('count', ascending=False, inplace=True)

remove_orgs = ['Intelligence', 'Kremlin', 'Ultra', 'International Security', 'Intelligence Studies', 'Intelligence Analysis', 'OSINT']
org_counts = org_counts[~org_counts['ORG'].isin(remove_orgs)]
org_counts = org_counts.reset_index(drop=True)

gpe_counts.head(15).to_csv('gpe.csv')
person_counts.head(15).to_csv('person.csv')
org_counts.head(15).to_csv('org.csv')



### RETRIEVING CITATION COUNT AND OA STATUS FROM OPENALEX
df_doi = df.copy()

df_doi = df_doi[['Zotero link', 'DOI']].dropna()
df_doi = df_doi.drop(df_doi[df_doi['DOI'] == ''].index)
df_doi = df_doi.reset_index(drop=True)
df_doi['DOI'] = df_doi['DOI'].str.replace('https://doi.org/', '')

def fetch_article_metadata(doi):
    base_url = 'https://api.openalex.org/works/https://doi.org/'
    response = requests.get(base_url + doi)
    if response.status_code == 200:
        data = response.json()
        counts_by_year = data.get('counts_by_year', [])
        if counts_by_year:
            first_citation_year = min(entry.get('year') for entry in data['counts_by_year'])
        else:
            first_citation_year = None
        if data.get('counts_by_year'):
            last_citation_year = max(entry.get('year') for entry in data['counts_by_year'])
        else:
            last_citation_year = None

        article_metadata = {
            'ID': data.get('id'),
            'Citation': data.get('cited_by_count'),
            'OA status': data.get('open_access', {}).get('is_oa'),
            'Citation_list': data.get('cited_by_api_url'),
            'First_citation_year': first_citation_year,
            'Last_citation_year': last_citation_year,
            'Publication_year': data.get('publication_year'),
            'OA_link': data.get('open_access', {}).get('oa_url')
        }
        return article_metadata
    else:
        return {
            'ID': None,
            'Citation': None,
            'OA status': None,
            'First_citation_year': None,
            'Last_citation_year': None,
            'Publication_year': None,
            'OA_link': None
        }

df_doi['ID'] = None
df_doi['Citation'] = None
df_doi['OA status'] = None
df_doi['Citation_list'] = None
df_doi['First_citation_year'] = None
df_doi['Last_citation_year'] = None
df_doi['Publication_year'] = None
df_doi['OA_link'] = None

# Iterate over each row in the DataFrame
for index, row in df_doi.iterrows():
    doi = row['DOI']
    article_metadata = fetch_article_metadata(doi)
    if article_metadata:
        # Update DataFrame with fetched information
        df_doi.at[index, 'ID'] = article_metadata['ID']
        df_doi.at[index, 'Citation'] = article_metadata['Citation']
        df_doi.at[index, 'OA status'] = article_metadata['OA status']
        df_doi.at[index, 'First_citation_year'] = article_metadata['First_citation_year']
        df_doi.at[index, 'Last_citation_year'] = article_metadata['Last_citation_year']
        df_doi.at[index, 'Citation_list'] = article_metadata.get('Citation_list', None)
        df_doi.at[index, 'Publication_year'] = article_metadata['Publication_year']
        df_doi.at[index, 'OA_link'] = article_metadata.get('OA_link', None)

# Calculate the difference between First_citation_year and Publication_year
df_doi['Year_difference'] = df_doi['First_citation_year'] - df_doi['Publication_year']
df_doi.to_csv('citations.csv')

df = pd.merge(df, df_doi, on='Zotero link', how='left')
duplicated_df = pd.merge(duplicated_df, df_doi, on='Zotero link', how='left')

df.to_csv('all_items.csv')
duplicated_df.to_csv('all_items_duplicated.csv')
df_countries.to_csv('countries.csv',index=False)
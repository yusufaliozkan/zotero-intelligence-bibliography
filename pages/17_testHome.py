from pyzotero import zotero
import pandas as pd
import streamlit as st
from IPython.display import HTML
import streamlit.components.v1 as components
import numpy as np
import altair as alt
from pandas.io.json import json_normalize
from datetime import date, timedelta  
import datetime
from streamlit_extras.switch_page_button import switch_page
import plotly.express as px
import numpy as np
import re
import matplotlib.pyplot as plt
import nltk
nltk.download('all')
from nltk.corpus import stopwords
nltk.download('stopwords')
from wordcloud import WordCloud
from gsheetsdb import connect
import gsheetsdb as gdb
import datetime as dt
import time
import PIL
from PIL import Image, ImageDraw, ImageFilter
import json
from authors_dict import df_authors, name_replacements
from copyright import display_custom_license

# Connecting Zotero with API 
library_id = '2514686'
library_type = 'group'
api_key = '' # api_key is only needed for private groups and libraries

# Bringing recently changed items

st.set_page_config(layout = "wide", 
                    page_title='Intelligence studies network',
                    page_icon="https://images.pexels.com/photos/315918/pexels-photo-315918.png",
                    initial_sidebar_state="auto") 
pd.set_option('display.max_colwidth', None)

zot = zotero.Zotero(library_id, library_type)

@st.cache_data(ttl=600)
def zotero_data(library_id, library_type):
    items = zot.top(limit=5)

    data=[]
    columns = ['Title','Publication type', 'Link to publication', 'Abstract', 'Zotero link', 'Date added', 'Date published', 'Date modified', 'Col key', 'Authors', 'Pub_venue']

    for item in items:
        creators = item['data']['creators']
        creators_str = ", ".join([creator.get('firstName', '') + ' ' + creator.get('lastName', '') for creator in creators])
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
        item['data'].get('publicationTitle')
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
    'preprint':'Preprint'
}
df['Publication type'] = df['Publication type'].replace(type_map)

df['Date published'] = pd.to_datetime(df['Date published'], errors='coerce')
df['Date published'] = pd.to_datetime(df['Date published'],utc=True).dt.tz_convert('Europe/London')
df['Date published'] = df['Date published'].dt.strftime('%d-%m-%Y')
df['Date published'] = df['Date published'].fillna('No date')
# df['Date published'] = df['Date published'].map(lambda x: x.strftime('%d/%m/%Y') if x else 'No date')

df['Date added'] = pd.to_datetime(df['Date added'], errors='coerce')
df['Date added'] = df['Date added'].dt.strftime('%d/%m/%Y')
df['Date modified'] = pd.to_datetime(df['Date modified'], errors='coerce')
df['Date modified'] = df['Date modified'].dt.strftime('%d/%m/%Y, %H:%M')

# Bringing collections

@st.cache_data(ttl=600)
def zotero_collections2(library_id, library_type):
    collections = zot.collections()
    data = [(item['data']['key'], item['data']['name'], item['meta']['numItems'], item['links']['alternate']['href']) for item in collections]
    df_collections = pd.DataFrame(data, columns=['Key', 'Name', 'Number', 'Link'])
    return df_collections
df_collections_2 = zotero_collections2(library_id, library_type)

@st.cache_data
def zotero_collections(library_id, library_type):
    collections = zot.collections()
    data2 = [(item['data']['key'], item['data']['name'], item['links']['alternate']['href']) for item in collections]
    df_collections = pd.DataFrame(data2, columns=['Key', 'Name', 'Link'])
    pd.set_option('display.max_colwidth', None)
    return df_collections.sort_values(by='Name')
df_collections = zotero_collections(library_id, library_type)

#To be deleted
if 0 in df:
    merged_df = pd.merge(
        left=df,
        right=df_collections,
        left_on=0,
        right_on='Key',
        how='left'
    )
    if 1 in merged_df:
        merged_df = pd.merge(
            left=merged_df,
            right=df_collections,
            left_on=1,
            right_on='Key',
            how='left'
        )
        if 2 in merged_df:
            merged_df = pd.merge(
                left=merged_df,
                right=df_collections,
                left_on=2,
                right_on='Key',
                how='left'
            ) 
df = merged_df.copy()
#To be deleted

df = df.fillna('')

# Streamlit app

st.title("Intelligence studies network")
st.header('Intelligence studies bibliography')
# st.header("[Zotero group library](https://www.zotero.org/groups/2514686/intelligence_bibliography/library)")

into = '''
Welcome to **Intelligence studies bibliography** 
This website lists different sources, events, conferences, and call for papers on intelligence history and intelligence studies. 
The current page shows the recently added or updated items. 
**If you wish to see more sources under different themes, see the sidebar menu** :arrow_left: .
The website has also a dynamic [digest](https://intelligence.streamlit.app/Digest) that you can tract latest publications & events.
Check it out the [short guide](https://medium.com/@yaliozkan/introduction-to-intelligence-studies-network-ed63461d1353) for a quick intoduction.

Links to PhD theses catalouged by the British EThOS may not be working due to the [cyber incident at the British Library](https://www.bl.uk/cyber-incident/). 
'''

with st.spinner('Retrieving data & updating dashboard...'): 

    count = zot.count_items()

    col1, col2 = st.columns([3,5])
    with col2:
        with st.expander('Intro'):
            st.info(into)
    with col1:
        df
        df_intro = pd.read_csv('all_items.csv')
        df_intro['Date added'] = pd.to_datetime(df_intro['Date added'])
        current_date = pd.to_datetime('now', utc=True)
        items_added_this_month = df_intro[
            (df_intro['Date added'].dt.year == current_date.year) & 
            (df_intro['Date added'].dt.month == current_date.month)
        ]
        st.write(f'**{count}** items available in this library. **{len(items_added_this_month)}** items added in {current_date.strftime("%B %Y")}.')
        st.write('The library last updated on ' + '**'+ df.loc[0]['Date modified']+'**')

    image = 'https://images.pexels.com/photos/315918/pexels-photo-315918.png'

    with st.sidebar:
        st.image(image, width=150)
        st.sidebar.markdown("# Intelligence studies network")
        with st.expander('About'):
            st.write('''This website lists secondary sources on intelligence studies and intelligence history.
            The sources are originally listed in the [Intelligence bibliography Zotero library](https://www.zotero.org/groups/2514686/intelligence_bibliography).
            This website uses [Zotero API](https://github.com/urschrei/pyzotero) to connect the *Intelligence bibliography Zotero group library*.
            To see more details about the sources, please visit the group library [here](https://www.zotero.org/groups/2514686/intelligence_bibliography/library). 
            If you need more information about Zotero, visit [this page](https://www.intelligencenetwork.org/zotero).
            ''')
            st.write('This website was built and is managed by [Yusuf Ozkan](https://www.kcl.ac.uk/people/yusuf-ali-ozkan) | [Twitter](https://twitter.com/yaliozkan) | [LinkedIn](https://www.linkedin.com/in/yusuf-ali-ozkan/) | [ORCID](https://orcid.org/0000-0002-3098-275X) | [GitHub](https://github.com/YusufAliOzkan)')
            components.html(
            """
            <a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons Licence" style="border-width:0" 
            src="https://i.creativecommons.org/l/by/4.0/80x15.png" /></a><br />
            © 2022 All rights reserved. This website is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
            """
            )
        with st.expander('Source code'):
            st.info('''
            Source code of this app is available [here](https://github.com/YusufAliOzkan/zotero-intelligence-bibliography).
            ''')
        with st.expander('Disclaimer'):
            st.warning('''
            This website and the Intelligence bibliography Zotero group library do not list all the sources on intelligence studies. 
            The list is created based on the creator's subjective views.
            ''')
        with st.expander('Contact us'):
            st.write('If you have any questions or suggestions, please do get in touch with us by filling the form [here](https://www.intelligencenetwork.org/contact-us).')
            st.write('Report your technical issues or requests [here](https://github.com/YusufAliOzkan/zotero-intelligence-bibliography/issues).')
        st.write('See our dynamic [digest](https://intelligence.streamlit.app/Digest)')
    # Recently added items

    tab1, tab2 = st.tabs(['📑 Publications', '📊 Dashboard'])
    with tab1:
        col1, col2 = st.columns([5,2]) 
        with col1:

            # SEARCH KEYWORD OR AUTHOR NAMES

            def format_entry(row):
                publication_type = str(row['Publication type']) if pd.notnull(row['Publication type']) else ''
                title = str(row['Title']) if pd.notnull(row['Title']) else ''
                authors = str(row['FirstName2'])
                date_published = str(row['Date published']) if pd.notnull(row['Date published']) else ''
                link_to_publication = str(row['Link to publication']) if pd.notnull(row['Link to publication']) else ''
                zotero_link = str(row['Zotero link']) if pd.notnull(row['Zotero link']) else ''
                published_by_or_in = ''
                published_source = ''

                if publication_type == 'Journal article':
                    published_by_or_in = 'Published in'
                    published_source = str(row['Journal']) if pd.notnull(row['Journal']) else ''
                elif publication_type == 'Book':
                    published_by_or_in = 'Published by'
                    published_source = str(row['Publisher']) if pd.notnull(row['Publisher']) else ''
                else:
                    # For other types, leave the fields empty
                    published_by_or_in = ''
                    published_source = ''

                return (
                    '**' + publication_type + '**' + ': ' +
                    title + ' ' +
                    '(by ' + '*' + authors + '*' + ') ' +
                    '(Publication date: ' + str(date_published) + ') ' +
                    ('(' + published_by_or_in + ': ' + '*' + published_source + '*' + ') ' if published_by_or_in else '') +
                    '[[Publication link]](' + link_to_publication + ') ' +
                    '[[Zotero link]](' + zotero_link + ')'
                )

            # Title input from the user
            st.header('Search in database')
            search_option = st.radio("Select search option", ("Search keywords", "Search author", "Search collections", "Publication types"))

            # df_authors = pd.read_csv('all_items.csv')
            # # df_authors['FirstName2'].fillna('', inplace=True)
            # df_authors['Author_name'] = df_authors['FirstName2'].apply(lambda x: x.split(', ') if isinstance(x, str) and x else x)
            # df_authors = df_authors.explode('Author_name')
            # df_authors.reset_index(drop=True, inplace=True)
            # df_authors = df_authors.dropna(subset=['FirstName2'])
            # name_replacements = {
            #     'David Gioe': 'David V. Gioe',
            #     'David Vincent Gioe': 'David V. Gioe',
            #     'Michael Goodman': 'Michael S. Goodman',
            #     'Michael S Goodman': 'Michael S. Goodman',
            #     'Michael Simon Goodman': 'Michael S. Goodman',
            #     'Thomas Maguire':'Thomas J. Maguire',
            #     'Thomas Joseph Maguire':'Thomas J. Maguire',
            #     'Huw John Davies':'Huw J. Davies',
            #     'Huw Davies':'Huw J. Davies',
            #     'Philip H.J. Davies':'Philip H. J. Davies',
            #     'Philip Davies':'Philip H. J. Davies',
            #     'Dan Lomas':'Daniel W. B. Lomas',
            #     'Richard Aldrich':'Richard J. Aldrich',
            #     'Richard J Aldrich':'Richard J. Aldrich',
            #     'Steven Wagner':'Steven B. Wagner',
            #     'Daniel Larsen':'Daniel R. Larsen',
            #     'Daniel Richard Larsen':'Daniel R. Larsen',
            #     'Loch Johnson':'Loch K. Johnson',
            #     'Sir David Omand Gcb':'David Omand',
            #     'Sir David Omand':'David Omand'
            # }
            # df_authors['Author_name'] = df_authors['Author_name'].map(name_replacements).fillna(df_authors['Author_name'])

            if search_option == "Search keywords":
                st.subheader('Search keywords')
                cols, cola = st.columns([2,6])
                with cols:
                    include_abstracts = st.selectbox('🔍 options', ['In title','In title & abstract'])
                with cola:
                    search_term = st.text_input('Search keywords in titles or abstracts')
                
                if search_term:
                    with st.expander('Click to expand', expanded=True):
                        search_terms = re.findall(r'(?:"[^"]*"|\w+)', search_term)  # Updated regex pattern
                        phrase_filter = '|'.join(search_terms)  # Filter for the entire phrase
                        keyword_filters = [term.strip('"') for term in search_terms]  # Separate filters for individual keywords

                        df_csv = pd.read_csv('all_items.csv')

                        # include_abstracts = st.checkbox('Search keywords in abstracts too')
                        display_abstracts = st.checkbox('Display abstracts')

                        if include_abstracts=='In title & abstract':
                            # Search for the entire phrase first
                            filtered_df = df_csv[
                                (df_csv['Title'].str.contains(phrase_filter, case=False, na=False, regex=True)) |
                                # (df_csv['FirstName2'].str.contains(phrase_filter, case=False, na=False, regex=True)) 
                                (df_csv['Abstract'].str.contains(phrase_filter, case=False, na=False, regex=True))
                            ]

                            # Search for individual keywords separately and combine the results
                            for keyword in keyword_filters:
                                keyword_filter_df = df_csv[
                                    (df_csv['Title'].str.contains(keyword, case=False, na=False, regex=True)) |
                                    # (df_csv['FirstName2'].str.contains(keyword, case=False, na=False, regex=True)) 
                                    (df_csv['Abstract'].str.contains(keyword, case=False, na=False, regex=True))
                                ]
                                filtered_df = pd.concat([filtered_df, keyword_filter_df])
                        else:
                            # Search for the entire phrase first
                            filtered_df = df_csv[
                                (df_csv['Title'].str.contains(phrase_filter, case=False, na=False, regex=True))
                                # (df_csv['FirstName2'].str.contains(phrase_filter, case=False, na=False, regex=True))
                            ]

                            # Search for individual keywords separately and combine the results
                            for keyword in keyword_filters:
                                keyword_filter_df = df_csv[
                                    (df_csv['Title'].str.contains(keyword, case=False, na=False, regex=True))
                                    # (df_csv['FirstName2'].str.contains(keyword, case=False, na=False, regex=True))
                                ]
                                filtered_df = pd.concat([filtered_df, keyword_filter_df])

                        # Remove duplicates, if any
                        filtered_df = filtered_df.drop_duplicates()
                        
                        filtered_df['Date published'] = pd.to_datetime(filtered_df['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
                        filtered_df['Date published'] = filtered_df['Date published'].dt.strftime('%Y-%m-%d')
                        filtered_df['Date published'] = filtered_df['Date published'].fillna('')
                        filtered_df['No date flag'] = filtered_df['Date published'].isnull().astype(np.uint8)
                        filtered_df = filtered_df.sort_values(by=['No date flag', 'Date published'], ascending=[True, True])
                        filtered_df = filtered_df.sort_values(by=['Date published'], ascending=False)

                        types = filtered_df['Publication type'].dropna().unique()  # Exclude NaN values
                        types2 = st.multiselect('Publication types', types, types, key='original2')

                        if types2:
                            filtered_df = filtered_df[filtered_df['Publication type'].isin(types2)]

                        if not filtered_df.empty:
                            num_items = len(filtered_df)
                            st.write(f"Matching articles ({num_items} sources found):")  # Display number of items found

                            download_filtered = filtered_df[['Publication type', 'Title', 'Abstract', 'Date published', 'Publisher', 'Journal', 'Link to publication', 'Zotero link']]
                            download_filtered = download_filtered.reset_index(drop=True)

                            def convert_df(download_filtered):
                                return download_filtered.to_csv(index=False).encode('utf-8-sig')
                            
                            csv = convert_df(download_filtered)
                            today = datetime.date.today().isoformat()
                            a = 'search-result-' + today
                            st.download_button('💾 Download search', csv, (a+'.csv'), mime="text/csv", key='download-csv-1')

                            on = st.toggle('Generate dashboard')

                            if on and len(filtered_df) > 0: 
                                st.info(f'Dashboard for search terms: {phrase_filter}')
                                search_df = filtered_df.copy()
                                publications_by_type = search_df['Publication type'].value_counts()
                                fig = px.bar(publications_by_type, x=publications_by_type.index, y=publications_by_type.values,
                                            labels={'x': 'Publication Type', 'y': 'Number of Publications'},
                                            title=f'Publications by Type')
                                st.plotly_chart(fig)

                                search_df = filtered_df.copy()
                                search_df['Year'] = pd.to_datetime(search_df['Date published']).dt.year
                                publications_by_year = search_df['Year'].value_counts().sort_index()
                                fig_year_bar = px.bar(publications_by_year, x=publications_by_year.index, y=publications_by_year.values,
                                                    labels={'x': 'Publication Year', 'y': 'Number of Publications'},
                                                    title=f'Publications by Year')
                                st.plotly_chart(fig_year_bar)
                            
                                search_df = filtered_df.copy()
                                search_df['Author_name'] = search_df['FirstName2'].apply(lambda x: x.split(', ') if isinstance(x, str) and x else x)
                                search_df = search_df.explode('Author_name')
                                search_df.reset_index(drop=True, inplace=True)
                                search_df['Author_name'] = search_df['Author_name'].map(name_replacements).fillna(search_df['Author_name'])
                                search_df = search_df['Author_name'].value_counts().head(10)
                                fig = px.bar(search_df, x=search_df.index, y=search_df.values)
                                fig.update_layout(
                                    title=f'Top 10 Authors by Publication Count',
                                    xaxis_title='Author',
                                    yaxis_title='Number of Publications',
                                    xaxis_tickangle=-45,
                                )
                                st.plotly_chart(fig)

                                search_df = filtered_df.copy()
                                def clean_text (text):
                                    text = text.lower() # lowercasing
                                    text = re.sub(r'[^\w\s]', ' ', text) # this removes punctuation
                                    text = re.sub('[0-9_]', ' ', text) # this removes numbers
                                    text = re.sub('[^a-z_]', ' ', text) # removing all characters except lowercase letters
                                    return text
                                search_df['clean_title'] = search_df['Title'].apply(clean_text)
                                search_df['clean_title'] = search_df['clean_title'].apply(lambda x: ' '.join ([w for w in x.split() if len (w)>2])) # this function removes words less than 2 words
                                def tokenization(text):
                                    text = re.split('\W+', text)
                                    return text    
                                search_df['token_title']=search_df['clean_title'].apply(tokenization)
                                stopword = nltk.corpus.stopwords.words('english')
                                SW = ['york', 'intelligence', 'security', 'pp', 'war','world', 'article', 'twitter', 'nan',
                                    'new', 'isbn', 'book', 'also', 'yet', 'matter', 'erratum', 'commentary', 'studies',
                                    'volume', 'paper', 'study', 'question', 'editorial', 'welcome', 'introduction', 'editorial', 'reader',
                                    'university', 'followed', 'particular', 'based', 'press', 'examine', 'show', 'may', 'result', 'explore',
                                    'examines', 'become', 'used', 'journal', 'london', 'review']
                                stopword.extend(SW)
                                def remove_stopwords(text):
                                    text = [i for i in text if i] # this part deals with getting rid of spaces as it treads as a string
                                    text = [word for word in text if word not in stopword] #keep the word if it is not in stopword
                                    return text
                                search_df['stopword']=search_df['token_title'].apply(remove_stopwords)
                                wn = nltk.WordNetLemmatizer()
                                def lemmatizer(text):
                                    text = [wn.lemmatize(word) for word in text]
                                    return text
                                search_df['lemma_title'] = search_df['stopword'].apply(lemmatizer)
                                listdf = search_df['lemma_title']
                                df_list = [item for sublist in listdf for item in sublist]
                                string = pd.Series(df_list).str.cat(sep=' ')
                                wordcloud_texts = string
                                wordcloud_texts_str = str(wordcloud_texts)
                                wordcloud = WordCloud(stopwords=stopword, width=1500, height=750, background_color='white', collocations=False, colormap='magma').generate(wordcloud_texts_str)
                                plt.figure(figsize=(20,8))
                                plt.axis('off')
                                plt.title(f"Word Cloud for Titles")
                                plt.imshow(wordcloud)
                                plt.axis("off")
                                plt.show()
                                st.set_option('deprecation.showPyplotGlobalUse', False)
                                st.pyplot()

                            else:
                                if num_items > 50:
                                    show_first_50 = st.checkbox("Show only first 50 items (untick to see all)", value=True)
                                    if show_first_50:
                                        filtered_df = filtered_df.head(50)

                                articles_list = []  # Store articles in a list
                                abstracts_list = [] #Store abstracts in a list
                                for index, row in filtered_df.iterrows():
                                    formatted_entry = format_entry(row)
                                    articles_list.append(formatted_entry)  # Append formatted entry to the list
                                    abstract = row['Abstract']
                                    abstracts_list.append(abstract if pd.notnull(abstract) else 'N/A')
                    
                                def highlight_terms(text, terms):
                                    # Regular expression pattern to identify URLs
                                    url_pattern = r'https?://\S+'

                                    # Find all URLs in the text
                                    urls = re.findall(url_pattern, text)
                                    
                                    # Replace URLs in the text with placeholders to avoid highlighting
                                    for url in urls:
                                        text = text.replace(url, f'___URL_PLACEHOLDER_{urls.index(url)}___')

                                    # Create a regex pattern to find the search terms in the text
                                    pattern = re.compile('|'.join(terms), flags=re.IGNORECASE)

                                    # Use HTML tags to highlight the terms in the text, excluding URLs
                                    highlighted_text = pattern.sub(
                                        lambda match: f'<span style="background-color: #FF8581;">{match.group(0)}</span>' 
                                                    if match.group(0) not in urls else match.group(0),
                                        text
                                    )

                                    # Restore the original URLs in the highlighted text
                                    for index, url in enumerate(urls):
                                        highlighted_text = highlighted_text.replace(f'___URL_PLACEHOLDER_{index}___', url)

                                    return highlighted_text
                                    
                                # Display the numbered list using Markdown syntax
                                for i, article in enumerate(articles_list, start=1):
                                    # Display the article with highlighted search terms
                                    highlighted_article = highlight_terms(article, search_terms)
                                    st.markdown(f"{i}. {highlighted_article}", unsafe_allow_html=True)
                                    
                                    # Display abstract under each numbered item only if the checkbox is selected
                                    if display_abstracts:
                                        abstract = abstracts_list[i - 1]  # Get the corresponding abstract for this article
                                        if pd.notnull(abstract):
                                            if include_abstracts=='In title & abstract':
                                                highlighted_abstract = highlight_terms(abstract, search_terms)
                                            else:
                                                highlighted_abstract = abstract 
                                            st.caption(f"Abstract: {highlighted_abstract}", unsafe_allow_html=True)
                                        else:
                                            st.caption(f"Abstract: No abstract")
                        else:
                            st.write("No articles found with the given keyword/phrase.")
                else:
                    st.write("Please enter a keyword or author name to search.")

            # SEARCH AUTHORS
            elif search_option == "Search author":
                st.subheader('Search author') 

                unique_authors = [''] + list(df_authors['Author_name'].unique())

                author_publications = df_authors['Author_name'].value_counts().to_dict()
                sorted_authors_by_publications = sorted(unique_authors, key=lambda author: author_publications.get(author, 0), reverse=True)
                select_options_author_with_counts = [''] + [f"{author} ({author_publications.get(author, 0)})" for author in sorted_authors_by_publications]

                selected_author_display = st.selectbox('Select author', select_options_author_with_counts)
                selected_author = selected_author_display.split(' (')[0] if selected_author_display else None
                # selected_author = st.selectbox('Select author', select_options_author)

                if not selected_author  or selected_author =="":
                    st.write('Select an author to see items')
                else:
                    filtered_collection_df_authors = df_authors[df_authors['Author_name']== selected_author]

                    filtered_collection_df_authors['Date published'] = pd.to_datetime(filtered_collection_df_authors['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
                    filtered_collection_df_authors['Date published'] = filtered_collection_df_authors['Date published'].dt.strftime('%Y-%m-%d')
                    filtered_collection_df_authors['Date published'] = filtered_collection_df_authors['Date published'].fillna('')
                    filtered_collection_df_authors['No date flag'] = filtered_collection_df_authors['Date published'].isnull().astype(np.uint8)
                    filtered_collection_df_authors = filtered_collection_df_authors.sort_values(by=['No date flag', 'Date published'], ascending=[True, True])
                    filtered_collection_df_authors = filtered_collection_df_authors.sort_values(by=['Date published'], ascending=False)
                    filtered_collection_df_authors =filtered_collection_df_authors.reset_index(drop=True)

                    publications_by_type = filtered_collection_df_authors['Publication type'].value_counts()

                    with st.expander('Click to expand', expanded=True):
                        st.markdown('#### Publications by ' + selected_author)
                        num_items_collections = len(filtered_collection_df_authors)
                        breakdown_string = ', '.join([f"{key}: {value}" for key, value in publications_by_type.items()])
                        st.write(f"**{num_items_collections}** sources found ({breakdown_string})")
                        st.write('*Please note that this database **may not show** all research outputs of the author.*')
                        types = st.multiselect('Publication type', filtered_collection_df_authors['Publication type'].unique(), filtered_collection_df_authors['Publication type'].unique(), key='original_authors')
                        filtered_collection_df_authors = filtered_collection_df_authors[filtered_collection_df_authors['Publication type'].isin(types)]
                        filtered_collection_df_authors = filtered_collection_df_authors.reset_index(drop=True)
                        def convert_df(filtered_collection_df_authors):
                            return filtered_collection_df_authors.to_csv(index=False).encode('utf-8-sig')
                        download_filtered = filtered_collection_df_authors[['Publication type', 'Title', 'Abstract', 'Date published', 'Publisher', 'Journal', 'Link to publication', 'Zotero link']]
                        csv = convert_df(download_filtered)
            
                        today = datetime.date.today().isoformat()
                        a = f'{selected_author}_{today}'
                        st.download_button('💾 Download publications', csv, (a+'.csv'), mime="text/csv", key='download-csv-authors')

                        on = st.toggle('Generate dashboard')
                        if on and len(filtered_collection_df_authors) > 0: 
                            st.info(f'Publications dashboard for {selected_author}')
                            author_df = filtered_collection_df_authors
                            publications_by_type = author_df['Publication type'].value_counts()
                            fig = px.bar(publications_by_type, x=publications_by_type.index, y=publications_by_type.values,
                                        labels={'x': 'Publication Type', 'y': 'Number of Publications'},
                                        title=f'Publications by Type ({selected_author})')
                            st.plotly_chart(fig)

                            author_df = filtered_collection_df_authors
                            author_df['Year'] = pd.to_datetime(author_df['Date published']).dt.year
                            publications_by_year = author_df['Year'].value_counts().sort_index()
                            fig_year_bar = px.bar(publications_by_year, x=publications_by_year.index, y=publications_by_year.values,
                                                labels={'x': 'Publication Year', 'y': 'Number of Publications'},
                                                title=f'Publications by Year ({selected_author})')
                            st.plotly_chart(fig_year_bar)

                            author_df = filtered_collection_df_authors
                            def clean_text (text):
                                text = text.lower() # lowercasing
                                text = re.sub(r'[^\w\s]', ' ', text) # this removes punctuation
                                text = re.sub('[0-9_]', ' ', text) # this removes numbers
                                text = re.sub('[^a-z_]', ' ', text) # removing all characters except lowercase letters
                                return text
                            author_df['clean_title'] = author_df['Title'].apply(clean_text)
                            author_df['clean_title'] = author_df['clean_title'].apply(lambda x: ' '.join ([w for w in x.split() if len (w)>2])) # this function removes words less than 2 words
                            def tokenization(text):
                                text = re.split('\W+', text)
                                return text    
                            author_df['token_title']=author_df['clean_title'].apply(tokenization)
                            stopword = nltk.corpus.stopwords.words('english')
                            SW = ['york', 'intelligence', 'security', 'pp', 'war','world', 'article', 'twitter', 'nan',
                                'new', 'isbn', 'book', 'also', 'yet', 'matter', 'erratum', 'commentary', 'studies',
                                'volume', 'paper', 'study', 'question', 'editorial', 'welcome', 'introduction', 'editorial', 'reader',
                                'university', 'followed', 'particular', 'based', 'press', 'examine', 'show', 'may', 'result', 'explore',
                                'examines', 'become', 'used', 'journal', 'london', 'review']
                            stopword.extend(SW)
                            def remove_stopwords(text):
                                text = [i for i in text if i] # this part deals with getting rid of spaces as it treads as a string
                                text = [word for word in text if word not in stopword] #keep the word if it is not in stopword
                                return text
                            author_df['stopword']=author_df['token_title'].apply(remove_stopwords)
                            wn = nltk.WordNetLemmatizer()
                            def lemmatizer(text):
                                text = [wn.lemmatize(word) for word in text]
                                return text
                            author_df['lemma_title'] = author_df['stopword'].apply(lemmatizer)
                            listdf = author_df['lemma_title']
                            df_list = [item for sublist in listdf for item in sublist]
                            string = pd.Series(df_list).str.cat(sep=' ')
                            wordcloud_texts = string
                            wordcloud_texts_str = str(wordcloud_texts)
                            wordcloud = WordCloud(stopwords=stopword, width=1500, height=750, background_color='white', collocations=False, colormap='magma').generate(wordcloud_texts_str)
                            plt.figure(figsize=(20,8))
                            plt.axis('off')
                            plt.title(f"Word Cloud for Titles published by ({selected_author})")
                            plt.imshow(wordcloud)
                            plt.axis("off")
                            plt.show()
                            st.set_option('deprecation.showPyplotGlobalUse', False)
                            st.pyplot()
                        else:
                            if not on:  # If the toggle is off, display the publications
                                for index, row in filtered_collection_df_authors.iterrows():
                                    publication_type = row['Publication type']
                                    title = row['Title']
                                    authors = row['FirstName2']
                                    date_published = row['Date published']
                                    link_to_publication = row['Link to publication']
                                    zotero_link = row['Zotero link']

                                    if publication_type == 'Journal article':
                                        published_by_or_in = 'Published in'
                                        published_source = str(row['Journal']) if pd.notnull(row['Journal']) else ''
                                    elif publication_type == 'Book':
                                        published_by_or_in = 'Published by'
                                        published_source = str(row['Publisher']) if pd.notnull(row['Publisher']) else ''
                                    else:
                                        published_by_or_in = ''
                                        published_source = ''

                                    formatted_entry = (
                                        '**' + str(publication_type) + '**' + ': ' +
                                        str(title) + ' ' +
                                        '(by ' + '*' + str(authors) + '*' + ') ' +
                                        '(Publication date: ' + str(date_published) + ') ' +
                                        ('(' + published_by_or_in + ': ' + '*' + str(published_source) + '*' + ') ' if published_by_or_in else '') +
                                        '[[Publication link]](' + str(link_to_publication) + ') ' +
                                        '[[Zotero link]](' + str(zotero_link) + ')'
                                    )
                                    st.write(f"{index + 1}) {formatted_entry}")

                            else:  # If toggle is on but no publications are available
                                st.write("No publication type selected.")

            # SEARCH IN COLLECTIONS
            elif search_option == "Search collections":
                st.subheader('Search collections')

                df_csv_collections = pd.read_csv('all_items_duplicated.csv')
                excluded_collections = ['97 KCL intelligence']
                numeric_start_collections = df_csv_collections[df_csv_collections['Collection_Name'].str[0].str.isdigit()]['Collection_Name'].unique()
                all_unique_collections = df_csv_collections['Collection_Name'].unique()
                filtered_collections = [col for col in numeric_start_collections if col not in excluded_collections]


                select_options = [''] + sorted(list(filtered_collections))
                selected_collection = st.selectbox('Select a collection', select_options)

                if not selected_collection or selected_collection == '':
                    st.write('Pick a collection to see items')
                else:
                    filtered_collection_df = df_csv_collections[df_csv_collections['Collection_Name'] == selected_collection]
                    # filtered_collection_df = filtered_collection_df.sort_values(by='Date published', ascending=False).reset_index(drop=True)

                    filtered_collection_df['Date published'] = pd.to_datetime(filtered_collection_df['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
                    filtered_collection_df['Date published'] = filtered_collection_df['Date published'].dt.strftime('%Y-%m-%d')
                    filtered_collection_df['Date published'] = filtered_collection_df['Date published'].fillna('')
                    filtered_collection_df['No date flag'] = filtered_collection_df['Date published'].isnull().astype(np.uint8)
                    filtered_collection_df = filtered_collection_df.sort_values(by=['No date flag', 'Date published'], ascending=[True, True])
                    filtered_collection_df = filtered_collection_df.sort_values(by=['Date published'], ascending=False)

                    publications_by_type = filtered_collection_df['Publication type'].value_counts()

                    collection_link = df_csv_collections[df_csv_collections['Collection_Name'] == selected_collection]['Collection_Link'].iloc[0]
                    
                    with st.expander('Click to expand', expanded=True):
                        st.markdown('#### Collection theme: ' + selected_collection)
                        st.write(f"See the collection in [Zotero]({collection_link})")
                        types = st.multiselect('Publication type', filtered_collection_df['Publication type'].unique(),filtered_collection_df['Publication type'].unique(), key='original')
                        filtered_collection_df = filtered_collection_df[filtered_collection_df['Publication type'].isin(types)]
                        filtered_collection_df = filtered_collection_df.reset_index(drop=True)
                        def convert_df(filtered_collection_df):
                            return filtered_collection_df.to_csv(index=False).encode('utf-8-sig')

                        csv = convert_df(filtered_collection_df)
                        today = datetime.date.today().isoformat()
                        num_items_collections = len(filtered_collection_df)
                        breakdown_string = ', '.join([f"{key}: {value}" for key, value in publications_by_type.items()])
                        st.write(f"**{num_items_collections}** sources found ({breakdown_string})")
                        a = f'{selected_collection}_{today}'
                        st.download_button('💾 Download the collection', csv, (a+'.csv'), mime="text/csv", key='download-csv-4')

                        on = st.toggle('Generate dashboard')
                        if on and len(filtered_collection_df) > 0: 
                            st.info(f'Dashboard for {selected_collection}')
                            collection_df = filtered_collection_df.copy()
                            
                            publications_by_type = collection_df['Publication type'].value_counts()
                            fig = px.bar(publications_by_type, x=publications_by_type.index, y=publications_by_type.values,
                                        labels={'x': 'Publication Type', 'y': 'Number of Publications'},
                                        title=f'Publications by Type ({selected_collection})')
                            st.plotly_chart(fig)

                            collection_df = filtered_collection_df.copy()
                            collection_df['Year'] = pd.to_datetime(collection_df['Date published']).dt.year
                            publications_by_year = collection_df['Year'].value_counts().sort_index()
                            fig_year_bar = px.bar(publications_by_year, x=publications_by_year.index, y=publications_by_year.values,
                                                labels={'x': 'Publication Year', 'y': 'Number of Publications'},
                                                title=f'Publications by Year ({selected_collection})')
                            st.plotly_chart(fig_year_bar)
                        
                            collection_author_df = filtered_collection_df.copy()
                            collection_author_df['Author_name'] = collection_author_df['FirstName2'].apply(lambda x: x.split(', ') if isinstance(x, str) and x else x)
                            collection_author_df = collection_author_df.explode('Author_name')
                            collection_author_df.reset_index(drop=True, inplace=True)
                            collection_author_df['Author_name'] = collection_author_df['Author_name'].map(name_replacements).fillna(collection_author_df['Author_name'])
                            collection_author_df = collection_author_df['Author_name'].value_counts().head(10)
                            fig = px.bar(collection_author_df, x=collection_author_df.index, y=collection_author_df.values)
                            fig.update_layout(
                                title=f'Top 10 Authors by Publication Count ({selected_collection})',
                                xaxis_title='Author',
                                yaxis_title='Number of Publications',
                                xaxis_tickangle=-45,
                            )
                            st.plotly_chart(fig)

                            author_df = filtered_collection_df.copy()
                            def clean_text (text):
                                text = text.lower() # lowercasing
                                text = re.sub(r'[^\w\s]', ' ', text) # this removes punctuation
                                text = re.sub('[0-9_]', ' ', text) # this removes numbers
                                text = re.sub('[^a-z_]', ' ', text) # removing all characters except lowercase letters
                                return text
                            author_df['clean_title'] = author_df['Title'].apply(clean_text)
                            author_df['clean_title'] = author_df['clean_title'].apply(lambda x: ' '.join ([w for w in x.split() if len (w)>2])) # this function removes words less than 2 words
                            def tokenization(text):
                                text = re.split('\W+', text)
                                return text    
                            author_df['token_title']=author_df['clean_title'].apply(tokenization)
                            stopword = nltk.corpus.stopwords.words('english')
                            SW = ['york', 'intelligence', 'security', 'pp', 'war','world', 'article', 'twitter', 'nan',
                                'new', 'isbn', 'book', 'also', 'yet', 'matter', 'erratum', 'commentary', 'studies',
                                'volume', 'paper', 'study', 'question', 'editorial', 'welcome', 'introduction', 'editorial', 'reader',
                                'university', 'followed', 'particular', 'based', 'press', 'examine', 'show', 'may', 'result', 'explore',
                                'examines', 'become', 'used', 'journal', 'london', 'review']
                            stopword.extend(SW)
                            def remove_stopwords(text):
                                text = [i for i in text if i] # this part deals with getting rid of spaces as it treads as a string
                                text = [word for word in text if word not in stopword] #keep the word if it is not in stopword
                                return text
                            author_df['stopword']=author_df['token_title'].apply(remove_stopwords)
                            wn = nltk.WordNetLemmatizer()
                            def lemmatizer(text):
                                text = [wn.lemmatize(word) for word in text]
                                return text
                            author_df['lemma_title'] = author_df['stopword'].apply(lemmatizer)
                            listdf = author_df['lemma_title']
                            df_list = [item for sublist in listdf for item in sublist]
                            string = pd.Series(df_list).str.cat(sep=' ')
                            wordcloud_texts = string
                            wordcloud_texts_str = str(wordcloud_texts)
                            wordcloud = WordCloud(stopwords=stopword, width=1500, height=750, background_color='white', collocations=False, colormap='magma').generate(wordcloud_texts_str)
                            plt.figure(figsize=(20,8))
                            plt.axis('off')
                            plt.title(f"Word Cloud for Titles in ({selected_collection})")
                            plt.imshow(wordcloud)
                            plt.axis("off")
                            plt.show()
                            st.set_option('deprecation.showPyplotGlobalUse', False)
                            st.pyplot()

                        else:
                            if not on:
                                if num_items_collections > 25:
                                    show_first_25 = st.checkbox("Show only first 25 items (untick to see all)", value=True)
                                    if show_first_25:
                                        filtered_collection_df = filtered_collection_df.head(25)

                                articles_list = []  # Store articles in a list
                                for index, row in filtered_collection_df.iterrows():
                                    formatted_entry = format_entry(row)  # Assuming format_entry() is a function formatting each row
                                    articles_list.append(formatted_entry)                     
                                
                                for index, row in filtered_collection_df.iterrows():
                                    publication_type = row['Publication type']
                                    title = row['Title']
                                    authors = row['FirstName2']
                                    date_published = row['Date published']
                                    link_to_publication = row['Link to publication']
                                    zotero_link = row['Zotero link']

                                    if publication_type == 'Journal article':
                                        published_by_or_in = 'Published in'
                                        published_source = str(row['Journal']) if pd.notnull(row['Journal']) else ''
                                    elif publication_type == 'Book':
                                        published_by_or_in = 'Published by'
                                        published_source = str(row['Publisher']) if pd.notnull(row['Publisher']) else ''
                                    else:
                                        published_by_or_in = ''
                                        published_source = ''

                                    formatted_entry = (
                                        '**' + str(publication_type) + '**' + ': ' +
                                        str(title) + ' ' +
                                        '(by ' + '*' + str(authors) + '*' + ') ' +
                                        '(Publication date: ' + str(date_published) + ') ' +
                                        ('(' + published_by_or_in + ': ' + '*' + str(published_source) + '*' + ') ' if published_by_or_in else '') +
                                        '[[Publication link]](' + str(link_to_publication) + ') ' +
                                        '[[Zotero link]](' + str(zotero_link) + ')'
                                    )
                                    st.write(f"{index + 1}) {formatted_entry}")
                            else:  # If toggle is on but no publications are available
                                st.write("No publication type selected.")

            elif search_option == "Publication types":
                st.subheader('Publication types')

                df_csv_types = pd.read_csv('all_items.csv')
                unique_types = [''] + list(df_csv_types['Publication type'].unique())  # Adding an empty string as the first option
                selected_type = st.selectbox('Select a publication type', unique_types)

                if not selected_type or selected_type == '':
                    st.write('Pick a publication type to see items')
                else:
                    filtered_type_df = df_csv_types[df_csv_types['Publication type'] == selected_type]
                    # filtered_collection_df = filtered_collection_df.sort_values(by='Date published', ascending=False).reset_index(drop=True)

                    filtered_type_df['Date published'] = pd.to_datetime(filtered_type_df['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
                    filtered_type_df['Date published'] = filtered_type_df['Date published'].dt.strftime('%Y-%m-%d')
                    filtered_type_df['Date published'] = filtered_type_df['Date published'].fillna('')
                    filtered_type_df['No date flag'] = filtered_type_df['Date published'].isnull().astype(np.uint8)
                    filtered_type_df = filtered_type_df.sort_values(by=['No date flag', 'Date published'], ascending=[True, True])
                    filtered_type_df = filtered_type_df.sort_values(by=['Date published'], ascending=False)
                    filtered_type_df = filtered_type_df.reset_index(drop=True)

                    # publications_by_type = filtered_collection_df['Publication type'].value_counts()
                    
                    with st.expander('Click to expand', expanded=True):
                        st.markdown('#### Publication type: ' + selected_type)
                        if selected_type == 'Thesis':
                            st.warning('Links to PhD theses catalouged by the British EThOS may not be working due to the [cyber incident at the British Library](https://www.bl.uk/cyber-incident/).')
                        def convert_df(filtered_type_df):
                            return filtered_type_df.to_csv(index=False).encode('utf-8-sig')

                        csv = convert_df(filtered_type_df)
                        today = datetime.date.today().isoformat()
                        num_items_collections = len(filtered_type_df)
                        st.write(f"**{num_items_collections}** sources found")
                        a = f'{selected_type}_{today}'
                        st.download_button('💾 Download', csv, (a+'.csv'), mime="text/csv", key='download-csv-4')

                        on = st.toggle('Generate dashboard')
                        if on and len (filtered_type_df) > 0:
                            st.info(f'Dashboard for {selected_type}')
                            type_df = filtered_type_df.copy()
                            collection_df = type_df.copy()
                            collection_df['Year'] = pd.to_datetime(collection_df['Date published']).dt.year
                            publications_by_year = collection_df['Year'].value_counts().sort_index()
                            fig_year_bar = px.bar(publications_by_year, x=publications_by_year.index, y=publications_by_year.values,
                                                labels={'x': 'Publication Year', 'y': 'Number of Publications'},
                                                title=f'Publications by Year ({selected_type})')
                            st.plotly_chart(fig_year_bar)

                            collection_author_df = type_df.copy()
                            collection_author_df['Author_name'] = collection_author_df['FirstName2'].apply(lambda x: x.split(', ') if isinstance(x, str) and x else x)
                            collection_author_df = collection_author_df.explode('Author_name')
                            collection_author_df.reset_index(drop=True, inplace=True)
                            collection_author_df['Author_name'] = collection_author_df['Author_name'].map(name_replacements).fillna(collection_author_df['Author_name'])
                            collection_author_df = collection_author_df['Author_name'].value_counts().head(10)
                            fig = px.bar(collection_author_df, x=collection_author_df.index, y=collection_author_df.values)
                            fig.update_layout(
                                title=f'Top 10 Authors by Publication Count ({selected_type})',
                                xaxis_title='Author',
                                yaxis_title='Number of Publications',
                                xaxis_tickangle=-45,
                            )
                            st.plotly_chart(fig)

                            author_df = type_df.copy()
                            def clean_text (text):
                                text = text.lower() # lowercasing
                                text = re.sub(r'[^\w\s]', ' ', text) # this removes punctuation
                                text = re.sub('[0-9_]', ' ', text) # this removes numbers
                                text = re.sub('[^a-z_]', ' ', text) # removing all characters except lowercase letters
                                return text
                            author_df['clean_title'] = author_df['Title'].apply(clean_text)
                            author_df['clean_title'] = author_df['clean_title'].apply(lambda x: ' '.join ([w for w in x.split() if len (w)>2])) # this function removes words less than 2 words
                            def tokenization(text):
                                text = re.split('\W+', text)
                                return text    
                            author_df['token_title']=author_df['clean_title'].apply(tokenization)
                            stopword = nltk.corpus.stopwords.words('english')
                            SW = ['york', 'intelligence', 'security', 'pp', 'war','world', 'article', 'twitter', 'nan',
                                'new', 'isbn', 'book', 'also', 'yet', 'matter', 'erratum', 'commentary', 'studies',
                                'volume', 'paper', 'study', 'question', 'editorial', 'welcome', 'introduction', 'editorial', 'reader',
                                'university', 'followed', 'particular', 'based', 'press', 'examine', 'show', 'may', 'result', 'explore',
                                'examines', 'become', 'used', 'journal', 'london', 'review']
                            stopword.extend(SW)
                            def remove_stopwords(text):
                                text = [i for i in text if i] # this part deals with getting rid of spaces as it treads as a string
                                text = [word for word in text if word not in stopword] #keep the word if it is not in stopword
                                return text
                            author_df['stopword']=author_df['token_title'].apply(remove_stopwords)
                            wn = nltk.WordNetLemmatizer()
                            def lemmatizer(text):
                                text = [wn.lemmatize(word) for word in text]
                                return text
                            author_df['lemma_title'] = author_df['stopword'].apply(lemmatizer)
                            listdf = author_df['lemma_title']
                            df_list = [item for sublist in listdf for item in sublist]
                            string = pd.Series(df_list).str.cat(sep=' ')
                            wordcloud_texts = string
                            wordcloud_texts_str = str(wordcloud_texts)
                            wordcloud = WordCloud(stopwords=stopword, width=1500, height=750, background_color='white', collocations=False, colormap='magma').generate(wordcloud_texts_str)
                            plt.figure(figsize=(20,8))
                            plt.axis('off')
                            plt.title(f"Word Cloud for Titles in ({selected_type})")
                            plt.imshow(wordcloud)
                            plt.axis("off")
                            plt.show()
                            st.set_option('deprecation.showPyplotGlobalUse', False)
                            st.pyplot()

                        else:
                            if num_items_collections > 25:
                                show_first_25 = st.checkbox("Show only first 25 items (untick to see all)", value=True)
                                if show_first_25:
                                    filtered_type_df = filtered_type_df.head(25)                            

                            articles_list = []  # Store articles in a list
                            for index, row in filtered_type_df.iterrows():
                                formatted_entry = format_entry(row)  # Assuming format_entry() is a function formatting each row
                                articles_list.append(formatted_entry)                     
                            
                            for index, row in filtered_type_df.iterrows():
                                publication_type = row['Publication type']
                                title = row['Title']
                                authors = row['FirstName2']
                                date_published = row['Date published'] 
                                link_to_publication = row['Link to publication']
                                zotero_link = row['Zotero link']

                                if publication_type == 'Journal article':
                                    published_by_or_in = 'Published in'
                                    published_source = str(row['Journal']) if pd.notnull(row['Journal']) else ''
                                elif publication_type == 'Book':
                                    published_by_or_in = 'Published by'
                                    published_source = str(row['Publisher']) if pd.notnull(row['Publisher']) else ''
                                else:
                                    published_by_or_in = ''
                                    published_source = ''

                                formatted_entry = (
                                    '**' + str(publication_type) + '**' + ': ' +
                                    str(title) + ' ' +
                                    '(by ' + '*' + str(authors) + '*' + ') ' +
                                    '(Publication date: ' + str(date_published) + ') ' +
                                    ('(' + published_by_or_in + ': ' + '*' + str(published_source) + '*' + ') ' if published_by_or_in else '') +
                                    '[[Publication link]](' + str(link_to_publication) + ') ' +
                                    '[[Zotero link]](' + str(zotero_link) + ')'
                                )
                                st.write(f"{index + 1}) {formatted_entry}")

            # RECENTLY ADDED ITEMS
            st.header('Recently added or updated items')
            df['Abstract'] = df['Abstract'].str.strip()
            df['Abstract'] = df['Abstract'].fillna('No abstract')
            
            df_download = df.iloc[:, [0,1,2,3,4,5,6,9]] 
            df_download = df_download[['Title', 'Publication type', 'Authors', 'Abstract', 'Link to publication', 'Zotero link', 'Date published', 'Date added']]

            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8-sig') # not utf-8 because of the weird character,  Â cp1252
            csv = convert_df(df_download)
            # csv = df_download
            # # st.caption(collection_name)
            today = datetime.date.today().isoformat()
            a = 'recently-added-' + today
            st.download_button('💾 Download recently added items', csv, (a+'.csv'), mime="text/csv", key='download-csv-3')
            
            display = st.checkbox('Display theme and abstract')

            df_last = ('**'+ df['Publication type']+ '**'+ ': ' + df['Title'] +', ' +                        
                        ' (by ' + '*' + df['Authors'] + '*' + ') ' +
                        ' (Published on: ' + df['Date published']+') ' +
                        '[[Publication link]]'+ '('+ df['Link to publication'] + ')' +
                        "[[Zotero link]]" +'('+ df['Zotero link'] + ')' 
                        )
            
            row_nu_1 = len(df_last.index)
            for i in range(row_nu_1):
                publication_type = df['Publication type'].iloc[i]
                if publication_type in ["Journal article", "Magazine article", 'Newspaper article']:
                    df_last = ('**'+ df['Publication type']+ '**'+ ': ' + df['Title'] +', ' +                        
                                ' (by ' + '*' + df['Authors'] + '*' + ') ' +
                                ' (Published on: ' + df['Date published']+') ' +
                                " (Published in: " + "*" + df['Pub_venue'] + "*" + ') '+
                                '[[Publication link]]'+ '('+ df['Link to publication'] + ')' +
                                "[[Zotero link]]" +'('+ df['Zotero link'] + ')' 
                                )
                    st.write(f"{i+1}) " + df_last.iloc[i])
                else:
                    df_last = ('**'+ df['Publication type']+ '**'+ ': ' + df['Title'] +', ' +                        
                                ' (by ' + '*' + df['Authors'] + '*' + ') ' +
                                ' (Published on: ' + df['Date published']+') ' +
                                '[[Publication link]]'+ '('+ df['Link to publication'] + ')' +
                                "[[Zotero link]]" +'('+ df['Zotero link'] + ')' 
                                )
                    st.write(f"{i+1}) " + df_last.iloc[i])
                
                if display:
                    a=''
                    b=''
                    c=''
                    if 'Name_x' in df:
                        a= '['+'['+df['Name_x'].iloc[i]+']' +'('+ df['Link_x'].iloc[i] + ')'+ ']'
                        if df['Name_x'].iloc[i]=='':
                            a=''
                    if 'Name_y' in df:
                        b='['+'['+df['Name_y'].iloc[i]+']' +'('+ df['Link_y'].iloc[i] + ')' +']'
                        if df['Name_y'].iloc[i]=='':
                            b=''
                    if 'Name' in df:
                        c= '['+'['+df['Name'].iloc[i]+']' +'('+ df['Link'].iloc[i] + ')'+ ']'
                        if df['Name'].iloc[i]=='':
                            c=''
                    st.caption('Theme(s):  \n ' + a + ' ' +b+ ' ' + c)
                    if not any([a, b, c]):
                        st.caption('No theme to display!')
                    
                    st.caption('Abstract: '+ df['Abstract'].iloc[i])

            st.header('All items in database')
            with st.expander('Click to expand', expanded=False):
                df_all_items = pd.read_csv('all_items.csv')
                df_all_items = df_all_items[['Publication type', 'Title', 'Abstract', 'Date published', 'Publisher', 'Journal', 'Link to publication', 'Zotero link']]

                def convert_df(df_all_items):
                    return df_all_items.to_csv(index=False).encode('utf-8-sig') # not utf-8 because of the weird character,  Â cp1252
                csv = convert_df(df_all_items)
                # csv = df_download
                # # st.caption(collection_name)
                today = datetime.date.today().isoformat()
                a = 'intelligence-bibliography-all-' + today
                st.download_button('💾 Download all items', csv, (a+'.csv'), mime="text/csv", key='download-csv-2')

                df_all_items

                df_added = pd.read_csv('all_items.csv')
                df_added['Date added'] = pd.to_datetime(df_added['Date added'])
                df_added['YearMonth'] = df_added['Date added'].dt.to_period('M').astype(str)
                monthly_counts = df_added.groupby('YearMonth').size()
                monthly_counts.name = 'Number of items added'
                cumulative_counts = monthly_counts.cumsum()
                cumulative_chart = alt.Chart(pd.DataFrame({'YearMonth': cumulative_counts.index, 'Total items': cumulative_counts})).mark_bar().encode(
                    x='YearMonth',
                    y='Total items',
                    tooltip=['YearMonth', 'Total items']
                ).properties(
                    width=600,
                    title='Total Number of Items Added'
                )
                step = 6
                data_labels = cumulative_chart.mark_text(
                    align='center',
                    baseline='bottom',
                    dy=-5,  # Adjust the vertical position of labels
                    fontSize=10
                ).encode(
                    x=alt.X('YearMonth', title='Year-Month', axis=alt.Axis(labelAngle=-45)),
                    y='Total items:Q',
                    text='Total items:Q'
                ).transform_filter(
                    alt.datum.YearMonth % step == 0
                )
                st.subheader('Total Number of Items Added per Month (cumulative)')
                st.altair_chart(cumulative_chart + data_labels, use_container_width=True)

                def format_entry(row):
                    publication_type = str(row['Publication type']) if pd.notnull(row['Publication type']) else ''
                    title = str(row['Title']) if pd.notnull(row['Title']) else ''
                    authors = str(row['FirstName2'])
                    date_published = str(row['Date published']) if pd.notnull(row['Date published']) else ''
                    link_to_publication = str(row['Link to publication']) if pd.notnull(row['Link to publication']) else ''
                    zotero_link = str(row['Zotero link']) if pd.notnull(row['Zotero link']) else ''
                    published_by_or_in = ''
                    published_source = ''

                    if publication_type == 'Journal article':
                        published_by_or_in = 'Published in'
                        published_source = str(row['Journal']) if pd.notnull(row['Journal']) else ''
                    elif publication_type == 'Book':
                        published_by_or_in = 'Published by'
                        published_source = str(row['Publisher']) if pd.notnull(row['Publisher']) else ''
                    else:
                        # For other types, leave the fields empty
                        published_by_or_in = ''
                        published_source = ''

                    return (
                        '**' + publication_type + '**' + ': ' +
                        title + ' ' +
                        '(by ' + '*' + authors + '*' + ') ' +
                        '(Publication date: ' + str(date_published) + ') ' +
                        ('(' + published_by_or_in + ': ' + '*' + published_source + '*' + ') ' if published_by_or_in else '') +
                        '[[Publication link]](' + link_to_publication + ') ' +
                        '[[Zotero link]](' + zotero_link + ')'
                    )
                df_all = pd.read_csv('all_items.csv')
                df_all['Date published2'] = pd.to_datetime(df_all['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
                df_all['Date year'] = df_all['Date published2'].dt.strftime('%Y')
                df_all['Date year'] = pd.to_numeric(df_all['Date year'], errors='coerce', downcast='integer')
                numeric_years = df_all['Date year'].dropna()
                current_year = date.today().year
                min_y = numeric_years.min()
                max_y = numeric_years.max()

                df_all['Date published'] = pd.to_datetime(df_all['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
                df_all['Date published'] = df_all['Date published'].dt.strftime('%Y-%m-%d')
                df_all['Date published'] = df_all['Date published'].fillna('')
                df_all['No date flag'] = df_all['Date published'].isnull().astype(np.uint8)
                df_all = df_all.sort_values(by=['No date flag', 'Date published'], ascending=[True, True])
                df_all = df_all.sort_values(by=['Date published'], ascending=False)

                current_year = date.today().year
                years = st.slider('Publication years between:', int(min(numeric_years)), int(max_y), (current_year, current_year), key='years')

                filter = (df_all['Date year'] >= years[0]) & (df_all['Date year'] <= years[1])
                df_all = df_all.loc[filter]
                number_of_items = len(df_all)
                st.write(f"{number_of_items} sources found published between {int(years[0])} and {int(years[1])}")
                
                articles_list = []  # Store articles in a list
                abstracts_list = [] #Store abstracts in a list
                for index, row in df_all.iterrows():
                    formatted_entry = format_entry(row)
                    articles_list.append(formatted_entry)  # Append formatted entry to the list
                    abstract = row['Abstract']
                    abstracts_list.append(abstract if pd.notnull(abstract) else 'N/A')
                for i, article in enumerate(articles_list, start=1):
                    # Display the article with highlighted search terms
                    st.markdown(f"{i}. {article}", unsafe_allow_html=True)

        with col2:
            with st.expander('Collections', expanded=True):
                st.caption('[Intelligence history](https://intelligence.streamlit.app/Intelligence_history)')
                st.caption('[Intelligence studies](https://intelligence.streamlit.app/Intelligence_studies)')
                st.caption('[Intelligence analysis](https://intelligence.streamlit.app/Intelligence_analysis)')
                st.caption('[Intelligence organisations](https://intelligence.streamlit.app/Intelligence_organisations)')
                st.caption('[Intelligence failures](https://intelligence.streamlit.app/Intelligence_failures)')
                st.caption('[Intelligence oversight and ethics](https://intelligence.streamlit.app/Intelligence_oversight_and_ethics)')
                st.caption('[Intelligence collection](https://intelligence.streamlit.app/Intelligence_collection)')
                st.caption('[Counterintelligence](https://intelligence.streamlit.app/Counterintelligence)')
                st.caption('[Covert action](https://intelligence.streamlit.app/Covert_action)')
                st.caption('[Intelligence and cybersphere](https://intelligence.streamlit.app/Intelligence_and_cybersphere)')
                st.caption('[Global intelligence](https://intelligence.streamlit.app/Global_intelligence)')
                st.caption('[AI and intelligence](https://intelligence.streamlit.app/AI_and_intelligence)')
                st.caption('[Special collections](https://intelligence.streamlit.app/Special_collections)')

            with st.expander('Events & conferences', expanded=True):
                st.markdown('##### Next event')
                conn = connect()

                # Perform SQL query on the Google Sheet.
                # Uses st.cache to only rerun when the query changes or after 10 min.
                @st.cache_resource(ttl=10)
                def run_query(query):
                    rows = conn.execute(query, headers=1)
                    rows = rows.fetchall()
                    return rows

                sheet_url = st.secrets["public_gsheets_url"]
                rows = run_query(f'SELECT * FROM "{sheet_url}"')

                data = []
                columns = ['event_name', 'organiser', 'link', 'date', 'venue', 'details']

                # Print results.
                for row in rows:
                    data.append((row.event_name, row.organiser, row.link, row.date, row.venue, row.details))

                pd.set_option('display.max_colwidth', None)
                df_gs = pd.DataFrame(data, columns=columns)
                df_gs['date_new'] = pd.to_datetime(df_gs['date'], dayfirst = True).dt.strftime('%d/%m/%Y')

                sheet_url_forms = st.secrets["public_gsheets_url_forms"]
                rows = run_query(f'SELECT * FROM "{sheet_url_forms}"')
                data = []
                columns = ['event_name', 'organiser', 'link', 'date', 'venue', 'details']
                # Print results.
                for row in rows:
                    data.append((row.Event_name, row.Event_organiser, row.Link_to_the_event, row.Date_of_event, row.Event_venue, row.Details))
                pd.set_option('display.max_colwidth', None)
                df_forms = pd.DataFrame(data, columns=columns)

                df_forms['date_new'] = pd.to_datetime(df_forms['date'], dayfirst = True).dt.strftime('%d/%m/%Y')
                df_forms['month'] = pd.to_datetime(df_forms['date'], dayfirst = True).dt.strftime('%m')
                df_forms['year'] = pd.to_datetime(df_forms['date'], dayfirst = True).dt.strftime('%Y')
                df_forms['month_year'] = pd.to_datetime(df_forms['date'], dayfirst = True).dt.strftime('%Y-%m')
                df_forms.sort_values(by='date', ascending = True, inplace=True)
                df_forms = df_forms.drop_duplicates(subset=['event_name', 'link', 'date'], keep='first')
                
                df_forms['details'] = df_forms['details'].fillna('No details')
                df_forms = df_forms.fillna('')
                df_gs = pd.concat([df_gs, df_forms], axis=0)
                df_gs = df_gs.reset_index(drop=True)
                df_gs = df_gs.drop_duplicates(subset=['event_name', 'link', 'date'], keep='first')

                df_gs.sort_values(by='date', ascending = True, inplace=True)
                df_gs = df_gs.drop_duplicates(subset=['event_name', 'link'], keep='first')
                df_gs = df_gs.fillna('')
                today = dt.date.today()
                filter = (df_gs['date']>=today)
                df_gs = df_gs.loc[filter]
                df_gs = df_gs.head(1)
                if df_gs['event_name'].any() in ("", [], None, 0, False):
                    st.write('No upcoming event!')
                df_gs1 = ('['+ df_gs['event_name'] + ']'+ '('+ df_gs['link'] + ')'', organised by ' + '**' + df_gs['organiser'] + '**' + '. Date: ' + df_gs['date_new'] + ', Venue: ' + df_gs['venue'])
                row_nu = len(df_gs.index)
                for i in range(row_nu):
                    st.write(df_gs1.iloc[i])
                
                st.markdown('##### Next conference')
                sheet_url2 = st.secrets["public_gsheets_url2"]
                rows = run_query(f'SELECT * FROM "{sheet_url2}"')
                data = []
                columns = ['conference_name', 'organiser', 'link', 'date', 'date_end', 'venue', 'details', 'location']
                for row in rows:
                    data.append((row.conference_name, row.organiser, row.link, row.date, row.date_end, row.venue, row.details, row.location))
                pd.set_option('display.max_colwidth', None)
                df_con = pd.DataFrame(data, columns=columns)
                df_con['date_new'] = pd.to_datetime(df_con['date'], dayfirst = True).dt.strftime('%d/%m/%Y')
                df_con['date_new_end'] = pd.to_datetime(df_con['date_end'], dayfirst = True).dt.strftime('%d/%m/%Y')
                df_con.sort_values(by='date', ascending = True, inplace=True)
                df_con['details'] = df_con['details'].fillna('No details')
                df_con['location'] = df_con['location'].fillna('No details')
                df_con = df_con.fillna('')            
                filter = (df_con['date_end']>=today)
                df_con = df_con.loc[filter]
                df_con = df_con.head(1)
                if df_con['conference_name'].any() in ("", [], None, 0, False):
                    st.write('No upcoming conference!')
                df_con1 = ('['+ df_con['conference_name'] + ']'+ '('+ df_con['link'] + ')'', organised by ' + '**' + df_con['organiser'] + '**' + '. Date(s): ' + df_con['date_new'] + ' - ' + df_con['date_new_end'] + ', Venue: ' + df_con['venue'])
                row_nu = len(df_con.index)
                for i in range(row_nu):
                    st.write( df_con1.iloc[i])
                st.write('Visit the [Events on intelligence](https://intelligence.streamlit.app/Events) page to see more!')

            with st.expander('Digest', expanded=True):
                st.write('See our dynamic [digest](https://intelligence.streamlit.app/Digest) for the latest updates on intelligence!')

    with tab2:
        st.header('Dashboard')
        on_main_dashboard = st.toggle('Display dashboard') 
        if on_main_dashboard:      
            number0 = st.slider('Select a number collections', 3,30,15)
            df_collections_2.set_index('Name', inplace=True)
            df_collections_2 = df_collections_2.sort_values(['Number'], ascending=[False])
            plot= df_collections_2.head(number0+1)
            # st.bar_chart(plot['Number'].sort_values(), height=600, width=600, use_container_width=True)
            plot = plot.reset_index()

            plot = plot[plot['Name']!='01 Intelligence history']
            fig = px.bar(plot, x='Name', y='Number', color='Name')
            fig.update_layout(
                autosize=False,
                width=600,
                height=600,)
            fig.update_layout(title={'text':'Top ' + str(number0) + ' collections in the library', 'y':0.95, 'x':0.4, 'yanchor':'top'})
            st.plotly_chart(fig, use_container_width = True)

            # Visauls for all items in the library
            df_csv = pd.read_csv('all_items.csv')
            df_csv['Date published'] = pd.to_datetime(df_csv['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
            df_csv['Date year'] = df_csv['Date published'].dt.strftime('%Y')
            df_csv['Date year'] = df_csv['Date year'].fillna('No date')
            df = df_csv.copy()
            df_year=df_csv['Date year'].value_counts()
            df_year=df_year.reset_index()
            df_year=df_year.rename(columns={'index':'Publication year','Date year':'Count'})
            df_year.drop(df_year[df_year['Publication year']== 'No date'].index, inplace = True)
            df_year=df_year.sort_values(by='Publication year', ascending=True)
            df_year=df_year.reset_index(drop=True)
            max_y = int(df_year['Publication year'].max())
            min_y = int(df_year['Publication year'].min())

            with st.expander('Select parameters', expanded=False):
                types = st.multiselect('Publication type', df_csv['Publication type'].unique(), df_csv['Publication type'].unique())
                years = st.slider('Publication years between:', min_y, max_y, (min_y,max_y), key='years2')
                if st.button('Update dashboard'):
                    df_csv = df_csv[df_csv['Publication type'].isin(types)]
                    df_csv = df_csv[df_csv['Date year'] !='No date']
                    filter = (df_csv['Date year'].astype(int)>=years[0]) & (df_csv['Date year'].astype(int)<years[1])
                    df_csv = df_csv.loc[filter]
                    df_year=df_csv['Date year'].value_counts()
                    df_year=df_year.reset_index()
                    df_year=df_year.rename(columns={'index':'Publication year','Date year':'Count'})
                    df_year.drop(df_year[df_year['Publication year']== 'No date'].index, inplace = True)
                    df_year=df_year.sort_values(by='Publication year', ascending=True)
                    df_year=df_year.reset_index(drop=True)

            df_types = pd.DataFrame(df_csv['Publication type'].value_counts())
            df_types = df_types.sort_values(['Publication type'], ascending=[False])
            df_types=df_types.reset_index()
            df_types = df_types.rename(columns={'index':'Publication type','Publication type':'Count'})

            if df_csv['Title'].any() in ("", [], None, 0, False):
                st.write('No data to visualise')
                st.stop()
            col1, col2 = st.columns(2)
            with col1:

                log0 = st.checkbox('Show in log scale', key='log0')

                if log0:
                    fig = px.bar(df_types, x='Publication type', y='Count', color='Publication type', log_y=True)
                    fig.update_layout(
                        autosize=False,
                        width=1200,
                        height=600,)
                    fig.update_xaxes(tickangle=-70)
                    fig.update_layout(title={'text':'Item types in log scale', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                    col1.plotly_chart(fig, use_container_width = True)
                else:
                    fig = px.bar(df_types, x='Publication type', y='Count', color='Publication type')
                    fig.update_layout(
                        autosize=False,
                        width=1200,
                        height=600,)
                    fig.update_xaxes(tickangle=-70)
                    fig.update_layout(title={'text':'Item types', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                    col1.plotly_chart(fig, use_container_width = True)

            with col2:
                fig = px.pie(df_types, values='Count', names='Publication type')
                fig.update_layout(title={'text':'Item types', 'y':0.95, 'x':0.45, 'yanchor':'top'})
                col2.plotly_chart(fig, use_container_width = True)

            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(df_year, x='Publication year', y='Count')
                fig.update_xaxes(tickangle=-70)
                fig.update_layout(
                    autosize=False,
                    width=1200,
                    height=600,)
                fig.update_layout(title={'text':'All items in the library by publication year', 'y':0.95, 'x':0.5, 'yanchor':'top'})
                col1.plotly_chart(fig, use_container_width = True)

            with col2:
                max_authors = len(df_authors['Author_name'].unique())
                num_authors = st.slider('Select number of authors to display:', 1, min(50, max_authors), 20)
                
                # Adding a multiselect widget for publication types
                selected_types = st.multiselect('Select publication types:', df_authors['Publication type'].unique(), default=df_authors['Publication type'].unique())
                
                # Filtering data based on selected publication types
                filtered_authors = df_authors[df_authors['Publication type'].isin(selected_types)]
                
                if len(selected_types) == 0:
                    st.write('No results to display')
                else:
                    publications_by_author = filtered_authors['Author_name'].value_counts().head(num_authors)
                    fig = px.bar(publications_by_author, x=publications_by_author.index, y=publications_by_author.values)
                    fig.update_layout(
                        title=f'Top {num_authors} Authors by Publication Count',
                        xaxis_title='Author',
                        yaxis_title='Number of Publications',
                        xaxis_tickangle=-45,
                    )
                    col2.plotly_chart(fig)

            col1, col2 = st.columns(2)
            with col1:
                number = st.slider('Select a number of publishers', 0, 30, 10)
                df_publisher = pd.DataFrame(df_csv['Publisher'].value_counts())
                df_publisher = df_publisher.sort_values(['Publisher'], ascending=[False])
                df_publisher = df_publisher.reset_index()
                df_publisher = df_publisher.rename(columns={'index':'Publisher','Publisher':'Count'})
                df_publisher = df_publisher.head(number)

                log1 = st.checkbox('Show in log scale', key='log1')
                leg1 = st.checkbox('Disable legend', key='leg1', disabled=False)

                if df_publisher['Publisher'].any() in ("", [], None, 0, False):
                    st.write('No publisher to display')
                else:
                    if log1:
                        if leg1:
                            fig = px.bar(df_publisher, x='Publisher', y='Count', color='Publisher', log_y=True)
                            fig.update_layout(
                                autosize=False,
                                width=1200,
                                height=700,
                                showlegend=False)
                            fig.update_xaxes(tickangle=-70)
                            fig.update_layout(title={'text':'Top ' + str(number) + ' publishers (in log scale)', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                            col1.plotly_chart(fig, use_container_width = True)
                        else:
                            fig = px.bar(df_publisher, x='Publisher', y='Count', color='Publisher', log_y=True)
                            fig.update_layout(
                                autosize=False,
                                width=1200,
                                height=700,
                                showlegend=True)
                            fig.update_xaxes(tickangle=-70)
                            fig.update_layout(title={'text':'Top ' + str(number) + ' publishers (in log scale)', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                            col1.plotly_chart(fig, use_container_width = True)
                    else:
                        if leg1:
                            fig = px.bar(df_publisher, x='Publisher', y='Count', color='Publisher', log_y=False)
                            fig.update_layout(
                                autosize=False,
                                width=1200,
                                height=700,
                                showlegend=False)
                            fig.update_xaxes(tickangle=-70)
                            fig.update_layout(title={'text':'Top ' + str(number) + ' publishers', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                            col1.plotly_chart(fig, use_container_width = True)
                        else:
                            fig = px.bar(df_publisher, x='Publisher', y='Count', color='Publisher', log_y=False)
                            fig.update_layout(
                                autosize=False,
                                width=1200,
                                height=700,
                                showlegend=True)
                            fig.update_xaxes(tickangle=-70)
                            fig.update_layout(title={'text':'Top ' + str(number) + ' publishers', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                            col1.plotly_chart(fig, use_container_width = True)
                    with st.expander('See publishers'):
                        row_nu_collections = len(df_publisher.index)        
                        for i in range(row_nu_collections):
                            st.caption(df_publisher['Publisher'].iloc[i]
                            )

            with col2:
                number2 = st.slider('Select a number of journals', 0,30,10)
                df_journal = df_csv.loc[df_csv['Publication type']=='Journal article']
                df_journal = pd.DataFrame(df_journal['Journal'].value_counts())
                df_journal = df_journal.sort_values(['Journal'], ascending=[False])
                df_journal = df_journal.reset_index()
                df_journal = df_journal.rename(columns={'index':'Journal','Journal':'Count'})
                df_journal = df_journal.head(number2)

                log2 = st.checkbox('Show in log scale', key='log2')
                leg2 = st.checkbox('Disable legend', key='leg2')

                if df_journal['Journal'].any() in ("", [], None, 0, False):
                    st.write('No journal to display')
                else:
                    if log2:
                        if leg2:
                            fig = px.bar(df_journal, x='Journal', y='Count', color='Journal', log_y=True)
                            fig.update_layout(
                                autosize=False,
                                width=1200,
                                height=700,
                                showlegend=False)
                            fig.update_xaxes(tickangle=-70)
                            fig.update_layout(title={'text':'Top ' + str(number2) + ' journals that publish intelligence articles (in log scale)', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                            col2.plotly_chart(fig, use_container_width = True)
                        else:
                            fig = px.bar(df_journal, x='Journal', y='Count', color='Journal', log_y=True)
                            fig.update_layout(
                                autosize=False,
                                width=1200,
                                height=700,
                                showlegend=True)
                            fig.update_xaxes(tickangle=-70)
                            fig.update_layout(title={'text':'Top ' + str(number2) + ' journals that publish intelligence articles (in log scale)', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                            col2.plotly_chart(fig, use_container_width = True)
                    else:
                        if leg2:
                            fig = px.bar(df_journal, x='Journal', y='Count', color='Journal', log_y=False)
                            fig.update_layout(
                                autosize=False,
                                width=1200,
                                height=700,
                                showlegend=False)
                            fig.update_xaxes(tickangle=-70)
                            fig.update_layout(title={'text':'Top ' + str(number2) + ' journals that publish intelligence articles', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                            col2.plotly_chart(fig, use_container_width = True)
                        else:
                            fig = px.bar(df_journal, x='Journal', y='Count', color='Journal', log_y=False)
                            fig.update_layout(
                                autosize=False,
                                width=1200,
                                height=700,
                                showlegend=True)
                            fig.update_xaxes(tickangle=-70)
                            fig.update_layout(title={'text':'Top ' + str(number2) + ' journals that publish intelligence articles', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                            col2.plotly_chart(fig, use_container_width = True)
                    with st.expander('See journals'):
                        row_nu_collections = len(df_journal.index)        
                        for i in range(row_nu_collections):
                            st.caption(df_journal['Journal'].iloc[i]
                            )
            col1, col2 = st.columns([7,2])
            with col1:
                df_countries = pd.read_csv('countries.csv')
                fig = px.choropleth(df_countries, locations='Country', locationmode='country names', color='Count', 
                            title='Country mentions in titles', color_continuous_scale='Viridis',
                            width=900, height=700) # Adjust the size of the map here
                # Display the map
                fig.show()
                col1.plotly_chart(fig, use_container_width=True) 
            with col2:
                st.markdown('##### Top 15 country names mentioned in titles')
                fig = px.bar(df_countries.head(15), x='Count', y='Country', orientation='h', height=600)
                col2.plotly_chart(fig, use_container_width=True)
            
            st.write('---')
            st.subheader('Named Entity Recognition analysis')
            st.caption('[What is Named Entity Recognition?](https://medium.com/mysuperai/what-is-named-entity-recognition-ner-and-how-can-i-use-it-2b68cf6f545d)')
            col1, col2, col3 = st.columns(3)
            with col1:
                gpe_counts = pd.read_csv('gpe.csv')
                fig = px.bar(gpe_counts.head(15), x='GPE', y='count', height=600, title="Top 15 locations mentioned in title & abstract")
                fig.update_xaxes(tickangle=-65)
                col1.plotly_chart(fig, use_container_width=True)
            with col2:
                person_counts = pd.read_csv('person.csv')
                fig = px.bar(person_counts.head(15), x='PERSON', y='count', height=600, title="Top 15 person mentioned in title & abstract")
                fig.update_xaxes(tickangle=-65)
                col2.plotly_chart(fig, use_container_width=True)
            with col3:
                org_counts = pd.read_csv('org.csv')
                fig = px.bar(org_counts.head(15), x='ORG', y='count', height=600, title="Top 15 organisations mentioned in title & abstract")
                fig.update_xaxes(tickangle=-65)
                col3.plotly_chart(fig, use_container_width=True)

            st.write('---')
            df=df_csv.copy()
            def clean_text (text):
                text = text.lower() # lowercasing
                text = re.sub(r'[^\w\s]', ' ', text) # this removes punctuation
                text = re.sub('[0-9_]', ' ', text) # this removes numbers
                text = re.sub('[^a-z_]', ' ', text) # removing all characters except lowercase letters
                return text
            df['clean_title'] = df['Title'].apply(clean_text)
            df['clean_abstract'] = df['Abstract'].astype(str).apply(clean_text)
            df_abs_no = df.dropna(subset=['clean_abstract'])
            df['clean_title'] = df['clean_title'].apply(lambda x: ' '.join ([w for w in x.split() if len (w)>2])) # this function removes words less than 2 words
            df['clean_abstract'] = df['clean_abstract'].apply(lambda x: ' '.join ([w for w in x.split() if len (w)>2])) # this function removes words less than 2 words

            def tokenization(text):
                text = re.split('\W+', text)
                return text
            df['token_title']=df['clean_title'].apply(tokenization)
            df['token_abstract']=df['clean_abstract'].apply(tokenization)
            stopword = nltk.corpus.stopwords.words('english')

            SW = ['york', 'intelligence', 'security', 'pp', 'war','world', 'article', 'twitter', 'nan',
                'new', 'isbn', 'book', 'also', 'yet', 'matter', 'erratum', 'commentary', 'studies',
                'volume', 'paper', 'study', 'question', 'editorial', 'welcome', 'introduction', 'editorial', 'reader',
                'university', 'followed', 'particular', 'based', 'press', 'examine', 'show', 'may', 'result', 'explore',
                'examines', 'become', 'used', 'journal', 'london', 'review']
            stopword.extend(SW)

            def remove_stopwords(text):
                text = [i for i in text if i] # this part deals with getting rid of spaces as it treads as a string
                text = [word for word in text if word not in stopword] #keep the word if it is not in stopword
                return text
            df['stopword']=df['token_title'].apply(remove_stopwords)
            df['stopword_abstract']=df['token_abstract'].apply(remove_stopwords)

            wn = nltk.WordNetLemmatizer()
            def lemmatizer(text):
                text = [wn.lemmatize(word) for word in text]
                return text

            df['lemma_title'] = df['stopword'].apply(lemmatizer) # error occurs in this line
            df['lemma_abstract'] = df['stopword_abstract'].apply(lemmatizer) # error occurs in this line

            listdf = df['lemma_title']
            listdf_abstract = df['lemma_abstract']

            st.subheader('Wordcloud')
            wordcloud_opt = st.radio('Wordcloud of:', ('Titles', 'Abstracts'))
            if wordcloud_opt=='Titles':
                df_list = [item for sublist in listdf for item in sublist]
                string = pd.Series(df_list).str.cat(sep=' ')
                wordcloud_texts = string
                wordcloud_texts_str = str(wordcloud_texts)
                wordcloud = WordCloud(stopwords=stopword, width=1500, height=750, background_color='white', collocations=False, colormap='magma').generate(wordcloud_texts_str)
                plt.figure(figsize=(20,8))
                plt.axis('off')
                plt.title('Top words in title (Intelligence bibliography collection)')
                plt.imshow(wordcloud)
                plt.axis("off")
                plt.show()
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot() 
            else:
                st.warning('Please bear in mind that not all items listed in this bibliography have an abstract. Therefore, this wordcloud should not be considered as authoritative. The number of items that have an abstract is ' + str(len(df_abs_no))+'.')
                df_list_abstract = [item for sublist in listdf_abstract for item in sublist]
                string = pd.Series(df_list_abstract).str.cat(sep=' ')
                wordcloud_texts = string
                wordcloud_texts_str = str(wordcloud_texts)
                wordcloud = WordCloud(stopwords=stopword, width=1500, height=750, background_color='white', collocations=False, colormap='magma').generate(wordcloud_texts_str)
                plt.figure(figsize=(20,8))
                plt.axis('off')
                plt.title('Top words in abstract (Intelligence bibliography collection)')
                plt.imshow(wordcloud)
                plt.axis("off")
                plt.show()
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot() 

            # Bring everything in the library

            df_types = pd.DataFrame(df_csv['Publication type'].value_counts())

            st.header('Items in the library by type: ')
            
            df_types = df_types.sort_values(['Publication type'], ascending=[False])
            plot2= df_types.head(10)

            st.bar_chart(plot2['Publication type'].sort_values(), height=600, width=600, use_container_width=True)

            st.header('Item inclusion history')
            df_added = df_csv.copy()
            time_interval = st.selectbox('Select time interval:', ['Monthly', 'Yearly'])
            col11, col12 = st.columns(2)
            with col11:
                df_added['Date added'] = pd.to_datetime(df_added['Date added'])
                df_added['YearMonth'] = df_added['Date added'].dt.to_period('M').astype(str)
                monthly_counts = df_added.groupby('YearMonth').size()
                monthly_counts.name = 'Number of items added'
                if time_interval == 'Monthly':
                    bar_chart = alt.Chart(monthly_counts.reset_index()).mark_bar().encode(
                        x='YearMonth',
                        y='Number of items added',
                        tooltip=['YearMonth', 'Number of items added']
                    ).properties(
                        width=600,
                        title='Number of Items Added per Month'
                    )
                    st.altair_chart(bar_chart, use_container_width=True)
                else:
                    df_added['Year'] = df_added['Date added'].dt.to_period('Y').astype(str)
                    yearly_counts = df_added.groupby('Year').size()
                    yearly_counts.name = 'Number of items added'
                    bar_chart = alt.Chart(yearly_counts.reset_index()).mark_bar().encode(
                        x='Year',
                        y='Number of items added',
                        tooltip=['Year', 'Number of items added']
                    ).properties(
                        width=600,
                        title='Number of Items Added per Year'
                    )
                    st.altair_chart(bar_chart, use_container_width=True)
            with col12:
                if time_interval == 'Monthly':
                    cumulative_counts = monthly_counts.cumsum()
                    cumulative_chart = alt.Chart(pd.DataFrame({'YearMonth': cumulative_counts.index, 'Cumulative': cumulative_counts})).mark_line().encode(
                        x='YearMonth',
                        y='Cumulative',
                        tooltip=['YearMonth', 'Cumulative']
                    ).properties(
                        width=600,
                        title='Cumulative Number of Items Added'
                    )
                    st.altair_chart(cumulative_chart, use_container_width=True)
                else:
                    cumulative_counts_y = yearly_counts.cumsum()
                    cumulative_chart = alt.Chart(pd.DataFrame({'Year': cumulative_counts_y.index, 'Cumulative': cumulative_counts_y})).mark_line().encode(
                        x='Year',
                        y='Cumulative',
                        tooltip=['Year', 'Cumulative']
                    ).properties(
                        width=600,
                        title='Cumulative Number of Items Added'
                    )
                    st.altair_chart(cumulative_chart, use_container_width=True)
        else:
            st.info('Toggle to see the dashboard!')
    st.write('---')
    with st.expander('Acknowledgements'):
        st.subheader('Acknowledgements')
        st.write('The following sources are used to collate some of the items and events in this website:')
        st.write("1. [King's Centre for the Study of Intelligence (KCSI) digest](https://kcsi.uk/kcsi-digests) compiled by David Schaefer")
        st.write("2. [International Association for Intelligence Education (IAIE) digest](https://www.iafie.org/Login.aspx) compiled by Filip Kovacevic")
        st.write("3. [North American Society for Intelligence History (NASIH)](https://www.intelligencehistory.org/brownbags)")

    display_custom_license()
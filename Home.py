from pyzotero import zotero
import pandas as pd
import streamlit as st
from IPython.display import HTML
import streamlit.components.v1 as components
import numpy as np
import altair as alt
# from pandas.io.json import json_normalize
from datetime import date, timedelta  
from datetime import datetime
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
# from gsheetsdb import connect
# import gsheetsdb as gdb
from streamlit_gsheets import GSheetsConnection
import datetime as dt
import time
import PIL
from PIL import Image, ImageDraw, ImageFilter
import json
from authors_dict import df_authors, name_replacements
from copyright import display_custom_license
from sidebar_content import sidebar_content
import plotly.graph_objs as go
import feedparser
import requests
from format_entry import format_entry
from streamlit_dynamic_filters import DynamicFilters
# from rss_feed import df_podcast, df_magazines


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
    items = zot.top(limit=10)
    items = sorted(items, key=lambda x: x['data']['dateAdded'], reverse=True)
    data=[]
    columns = ['Title','Publication type', 'Link to publication', 'Abstract', 'Zotero link', 'Date added', 'Date published', 'Date modified', 'Col key', 'Authors', 'Pub_venue', 'Book_title']

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
        item['data'].get('bookTitle')
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
    'computerProgram':'Computer program'
}
df['Publication type'] = df['Publication type'].replace(type_map)
df['Date published'] = (
    df['Date published']
    .str.strip()
    .apply(lambda x: pd.to_datetime(x, utc=True, errors='coerce').tz_convert('Europe/London'))
)
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

st.title("Intelligence studies network", anchor=False)
st.header('Intelligence studies bibliography', anchor=False)
# st.header("[Zotero group library](https://www.zotero.org/groups/2514686/intelligence_bibliography/library)")

# cite_today = datetime.date.today().isoformat()
cite_today = datetime.date.today().strftime("%d %B %Y")

into = f'''
Welcome to **Intelligence studies bibliography**.
This website lists **hundreds of sources, events, conferences, and call for papers** on intelligence history and intelligence studies. Finding sources on intelligence
can sometimes be challening. This bibliography will be your helpful research assistant in finding and saving sources.

Check out the following guides for a quick intoduction about the website:

Ozkan, Yusuf Ali. “Introduction to ‘Intelligence Studies Bibliography.’” Medium (blog), December 26, 2023. https://medium.com/@yaliozkan/introduction-to-intelligence-studies-network-ed63461d1353.

Ozkan, Yusuf Ali. ‘Enhancing the “Intelligence Studies Network” Website’. Medium (blog), 20 January 2024. https://medium.com/@yaliozkan/enhancing-the-intelligence-studies-network-website-13aa0c80f7f4.

**Cite this page:** Ozkan, Yusuf A. ‘*Intelligence Studies Network*’, Created 1 June 2020, Accessed {cite_today}. https://intelligence.streamlit.app/.
'''

with st.spinner('Retrieving data & updating dashboard...'): 

    item_count = zot.num_items() 

    df_dedup = pd.read_csv('all_items.csv')
    df_duplicated = pd.read_csv('all_items_duplicated.csv')

    col1, col2 = st.columns([3,5])
    with col2:
        with st.expander('Introduction'):
            st.info(into)
    with col1:
        df_intro = df_dedup.copy()
        df_intro['Date added'] = pd.to_datetime(df_intro['Date added'])
        current_date = pd.to_datetime('now', utc=True)
        items_added_this_month = df_intro[
            (df_intro['Date added'].dt.year == current_date.year) & 
            (df_intro['Date added'].dt.month == current_date.month)
        ]        # st.write(f'**{item_count}** items available in this library. **{len(items_added_this_month)}** items added in {current_date.strftime("%B %Y")}.')
        st.metric(label='Number of items in the library', value=item_count, delta=len(items_added_this_month),label_visibility='visible', help=f' **{len(items_added_this_month)}** items added in {current_date.strftime("%B %Y")}')
        st.write('The library last updated on ' + '**'+ df.loc[0]['Date modified']+'**') 

    sidebar_content() 

    tab1, tab2 = st.tabs(['📑 Publications', '📊 Dashboard']) #, '🔀 Surprise me'])
    with tab1:

        col1, col2 = st.columns([6,2]) 
        with col1: 

            # SEARCH KEYWORD OR AUTHOR NAMES

            # def format_entry(row, include_citation=True):
            #     publication_type = str(row['Publication type']) if pd.notnull(row['Publication type']) else ''
            #     title = str(row['Title']) if pd.notnull(row['Title']) else ''
            #     authors = str(row['FirstName2'])
            #     date_published = str(row['Date published']) if pd.notnull(row['Date published']) else ''
            #     link_to_publication = str(row['Link to publication']) if pd.notnull(row['Link to publication']) else ''
            #     zotero_link = str(row['Zotero link']) if pd.notnull(row['Zotero link']) else ''
            #     published_by_or_in = ''
            #     published_source = ''
            #     citation = str(row['Citation']) if pd.notnull(row['Citation']) else '0'  
            #     citation = int(float(citation))
            #     citation_link = str(row['Citation_list']) if pd.notnull(row['Citation_list']) else ''
            #     citation_link = citation_link.replace('api.', '')

            #     published_by_or_in_dict = {
            #         'Journal article': 'Published in',
            #         'Magazine article': 'Published in',
            #         'Newspaper article': 'Published in',
            #         'Book': 'Published by',
            #     }

            #     publication_type = row['Publication type']

            #     published_by_or_in = published_by_or_in_dict.get(publication_type, '')
            #     published_source = str(row['Journal']) if pd.notnull(row['Journal']) else ''
            #     if publication_type == 'Book':
            #         published_source = str(row['Publisher']) if pd.notnull(row['Publisher']) else ''

            #     citation_text = ('Cited by [' + str(citation) + '](' + citation_link + ')' if citation > 0 
            #          else '')

            #     return (
            #         '**' + publication_type + '**' + ': ' +
            #         title + ' ' +
            #         '(by ' + '*' + authors + '*' + ') ' +
            #         '(Publication date: ' + str(date_published) + ') ' +
            #         ('(' + published_by_or_in + ': ' + '*' + published_source + '*' + ') ' if published_by_or_in else '') +
            #         '[[Publication link]](' + link_to_publication + ') ' +
            #         '[[Zotero link]](' + zotero_link + ') ' + 
            #         (citation_text if include_citation else '')
            #     )

            # Title input from the user

            def parse_search_terms(search_term):
                # Split the search term by spaces while keeping phrases in quotes together
                tokens = re.findall(r'(?:"[^"]*"|\S+)', search_term)
                boolean_tokens = []
                for token in tokens:
                    # Treat "AND", "OR", "NOT" as Boolean operators only if they are uppercase
                    if token in ["AND", "OR", "NOT"]:
                        boolean_tokens.append(token)
                    else:
                        # Don't strip characters within quoted phrases
                        if token.startswith('"') and token.endswith('"'):
                            stripped_token = token.strip('"')
                        else:
                            # Preserve alphanumeric characters, apostrophes, hyphens, en dash, and other special characters
                            stripped_token = re.sub(r'[^a-zA-Z0-9\s\'\-–’]', '', token)
                            # Remove parentheses from the stripped token
                            stripped_token = stripped_token.replace('(', '').replace(')', '')
                        boolean_tokens.append(stripped_token.strip('"'))
                
                # Remove trailing operators
                while boolean_tokens and boolean_tokens[-1] in ["AND", "OR", "NOT"]:
                    boolean_tokens.pop()
                
                return boolean_tokens

            def apply_boolean_search(df, search_tokens, include_abstracts):
                if not search_tokens:
                    return df

                query = ''
                negate_next = False

                for i, token in enumerate(search_tokens):
                    if token == "AND":
                        query += " & "
                        negate_next = False
                    elif token == "OR":
                        query += " | "
                        negate_next = False
                    elif token == "NOT":
                        negate_next = True
                    elif token == "(":
                        query += " ("
                    elif token == ")":
                        query += ") "
                    else:
                        escaped_token = re.escape(token)
                        if include_abstracts == 'In title & abstract':
                            condition = f'(Title.str.contains(r"\\b{escaped_token}\\b", case=False, na=False) | Abstract.str.contains(r"\\b{escaped_token}\\b", case=False, na=False))'
                        else:
                            condition = f'Title.str.contains(r"\\b{escaped_token}\\b", case=False, na=False)'

                        if negate_next:
                            condition = f"~({condition})"
                            negate_next = False

                        if query and query.strip()[-1] not in "&|(":
                            query += " & "

                        query += condition

                # Debugging output
                print(f"Query: {query}")

                try:
                    filtered_df = df.query(query, engine='python')
                except Exception as e:
                    print(f"Error in query: {query}\n{e}")
                    return pd.DataFrame()

                return filtered_df

            def highlight_terms(text, terms):
                boolean_operators = {"AND", "OR", "NOT"}
                url_pattern = r'https?://\S+'
                urls = re.findall(url_pattern, text)
                for url in urls:
                    text = text.replace(url, f'___URL_PLACEHOLDER_{urls.index(url)}___')

                pattern = re.compile('|'.join(rf'\b{re.escape(term)}\b' for term in terms if term not in boolean_operators), flags=re.IGNORECASE)
                highlighted_text = pattern.sub(lambda match: f'<span style="background-color: #FF8581;">{match.group(0)}</span>' if match.group(0) not in urls else match.group(0), text)
                for index, url in enumerate(urls):
                    highlighted_text = highlighted_text.replace(f'___URL_PLACEHOLDER_{index}___', url)
                
                return highlighted_text

            # Example Streamlit code for context
            st.header('Search in database', anchor=None)
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            search_option = st.radio("Select search option", ("Search keywords", "Search author", "Search collection", "Publication types", "Search journal", "Publication year", "Cited papers"))

            if search_option == "Search keywords":
                st.subheader('Search keywords', anchor=None)
                @st.experimental_dialog("Search guide")
                def guide(item):
                    st.write('''
                        The Intelligence Studies Bibliography supports basic-level searches with Boolean operators.

                        Available Boolean operators: **AND**, **OR**, **NOT** (e.g., "covert action" **NOT** British).

                        You can search using double quotes (e.g., "covert action").

                        Multiple Boolean operators are allowed: (e.g. "covert action" **OR** "covert operation" **OR** "covert operations")

                        Please note: Search with parentheses is **not** available.

                        Note that the search function is limited: you will only find exact matches and cannot see search relevance.

                        You can share the link of your search result. Try: https://intelligence.streamlit.app/?query=cia+OR+mi6
                        ''')
                
                if "guide" not in st.session_state:
                    if st.button("Search guide"):
                        guide("Search guide")
                container_refresh_button = st.container()

                # if st.button('Search guide'):
                #     st.toast('''
                #     **Search guide**

                #     The following Boolean operators are available: AND, OR, NOT (e.g. "covert action" NOT british).

                #     Search with double quote is available. (e.g. "covert action")

                #     Search with parantheses is **not** available.                   
                #     ''')

                
                query_params = st.query_params.to_dict() 
                search_term = query_params.get("query", "")
                cols, cola = st.columns([2,6])
                with cols:
                    include_abstracts = st.selectbox('🔍 options', ['In title','In title & abstract'])
                with cola:
                    search_term = st.text_input('Search keywords in titles or abstracts', placeholder='Type your keyword(s)')

                def extract_quoted_phrases(text):
                    quoted_phrases = re.findall(r'"(.*?)"', text)
                    text_without_quotes = re.sub(r'"(.*?)"', '', text)
                    words = text_without_quotes.split()
                    return quoted_phrases + words

                search_term = search_term.strip()
                if search_term:
                    with st.status("Searching publications...", expanded=True) as status:
                        search_tokens = parse_search_terms(search_term)
                        print(f"Search Tokens: {search_tokens}")  # Debugging: Print search tokens
                        df_csv = df_duplicated.copy()

                        filtered_df = apply_boolean_search(df_csv, search_tokens, include_abstracts)
                        print(f"Filtered DataFrame (before dropping duplicates):\n{filtered_df}")  # Debugging: Print DataFrame before dropping duplicates
                        filtered_df = filtered_df.drop_duplicates()
                        print(f"Filtered DataFrame (after dropping duplicates):\n{filtered_df}")  # Debugging: Print DataFrame after dropping duplicates
                        if not filtered_df.empty and 'Date published' in filtered_df.columns:
                            filtered_df['Date published'] = filtered_df['Date published'].astype(str).str.strip()
                            filtered_df['Date published'] = filtered_df['Date published'].str.strip().apply(lambda x: pd.to_datetime(x, utc=True, errors='coerce').tz_convert('Europe/London'))
                            if filtered_df['Date published'].notna().any():
                                filtered_df['Date published'] = filtered_df['Date published'].dt.strftime('%Y-%m-%d')
                            else:
                                filtered_df['Date published'] = ''
                            filtered_df['Date published'] = filtered_df['Date published'].fillna('')
                            filtered_df['No date flag'] = filtered_df['Date published'].isnull().astype(np.uint8)
                            filtered_df = filtered_df.sort_values(by=['No date flag', 'Date published'], ascending=[True, True])
                        else:
                            filtered_df['Date published'] = ''
                            filtered_df['No date flag'] = 1
                        print(f"Final Filtered DataFrame:\n{filtered_df}")  # Debugging: Print final DataFrame
                        
                        types = filtered_df['Publication type'].dropna().unique()  # Exclude NaN values
                        collections = filtered_df['Collection_Name'].dropna().unique()
                        # st.query_params.from_dict({"query": search_term})
            
                        # if container_refresh_button.button('Refresh'):
                        #     st.query_params.clear()
                        #     st.rerun()

                        with st.popover("Filters and more"):
                            types2 = st.multiselect('Publication types', types, key='original2')
                            collections = st.multiselect('Collection', collections, key='original_collection')
                            container_download_button = st.container()

                            col112, col113, col114 = st.columns(3)
                            with col112:
                                display_abstracts = st.checkbox('Display abstracts')
                            with col113:
                                only_citation = st.checkbox('Show cited items only')
                                if only_citation:
                                    df_csv = df_csv[(df_csv['Citation'].notna()) & (df_csv['Citation'] != 0)]
                            with col114:
                                table_view = st.checkbox('See results in table')

                        if types2:
                            filtered_df = filtered_df[filtered_df['Publication type'].isin(types2)]                 

                        if collections:
                            filtered_df = filtered_df[filtered_df['Collection_Name'].isin(collections)] 


                        if not filtered_df.empty:
                            filtered_df = filtered_df.drop_duplicates(subset=['Zotero link'], keep='first')
                            num_items = len(filtered_df)
                            st.write(f"Matching articles (**{num_items}** {'source' if num_items == 1 else 'sources'} found):")

                            download_filtered = filtered_df[['Publication type', 'Title', 'Abstract', 'Date published', 'Publisher', 'Journal', 'Link to publication', 'Zotero link', 'Citation']]
                            download_filtered['Abstract'] = download_filtered['Abstract'].str.replace('\n', ' ')
                            download_filtered = download_filtered.reset_index(drop=True)

                            def convert_df(download_filtered):
                                return download_filtered.to_csv(index=False).encode('utf-8-sig')
                            
                            csv = convert_df(download_filtered)
                            today = datetime.date.today().isoformat()
                            a = 'search-result-' + today
                            container_download_button.download_button('💾 Download search', csv, (a+'.csv'), mime="text/csv", key='download-csv-1')

                            on = st.toggle('Generate dashboard')

                            if on and len(filtered_df) > 0: 
                                st.info(f'Dashboard for search terms: {search_term}')
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

                                custom_stopwords = extract_quoted_phrases(search_term)
                                stopword.extend(custom_stopwords)

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
                                wordcloud = WordCloud(stopwords=stopword, width=1500, height=750, background_color='white', collocations=False).generate(wordcloud_texts_str)
                                plt.figure(figsize=(20,8))
                                plt.axis('off')
                                plt.title(f"Word Cloud for Titles")
                                plt.imshow(wordcloud)
                                plt.axis("off")
                                plt.show()
                                st.set_option('deprecation.showPyplotGlobalUse', False)
                                st.pyplot()

                            else:
                                sort_by = st.radio('Sort by:', ('Publication date :arrow_down:', 'Citation'))
                                if sort_by == 'Publication date :arrow_down:' or filtered_df['Citation'].sum() == 0:
                                    filtered_df = filtered_df.sort_values(by=['Date published'], ascending=False)
                                    filtered_df = filtered_df.reset_index(drop=True)
                                else:
                                    filtered_df = filtered_df.sort_values(by=['Citation'], ascending=False)
                                    filtered_df = filtered_df.reset_index(drop=True)
                                if num_items > 20:
                                    show_first_20 = st.checkbox("Show only first 20 items (untick to see all)", value=True)
                                    if show_first_20:
                                        filtered_df = filtered_df.head(20)

                                articles_list = []  # Store articles in a list
                                abstracts_list = [] #Store abstracts in a list
                                for index, row in filtered_df.iterrows():
                                    formatted_entry = format_entry(row)
                                    articles_list.append(formatted_entry)  # Append formatted entry to the list
                                    abstract = row['Abstract']
                                    abstracts_list.append(abstract if pd.notnull(abstract) else 'N/A')

                                def highlight_terms(text, terms):
                                    # Define boolean operators
                                    boolean_operators = {"AND", "OR", "NOT"}

                                    # Regular expression pattern to identify URLs
                                    url_pattern = r'https?://\S+'

                                    # Find all URLs in the text
                                    urls = re.findall(url_pattern, text)
                                    
                                    # Replace URLs in the text with placeholders to avoid highlighting
                                    for url in urls:
                                        text = text.replace(url, f'___URL_PLACEHOLDER_{urls.index(url)}___')

                                    # Create a regex pattern to find the search terms in the text, excluding boolean operators
                                    pattern = re.compile('|'.join(rf'\b{re.escape(term)}\b' for term in terms if term not in boolean_operators), flags=re.IGNORECASE)

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
                                if table_view:
                                    df_table_view = filtered_df[['Publication type','Title','Date published','FirstName2', 'Abstract','Publisher','Journal','Collection_Name','Link to publication','Zotero link']]
                                    df_table_view = df_table_view.rename(columns={'FirstName2':'Author(s)','Collection_Name':'Collection','Link to publication':'Publication link'})
                                    df_table_view
                                else:
                                # Display the numbered list using Markdown syntax
                                    for i, article in enumerate(articles_list, start=1):
                                        # Display the article with highlighted search terms
                                        highlighted_article = highlight_terms(article, search_tokens)
                                        st.markdown(f"{i}. {highlighted_article}", unsafe_allow_html=True)
                                        
                                        # Display abstract under each numbered item only if the checkbox is selected
                                        if display_abstracts:
                                            abstract = abstracts_list[i - 1]  # Get the corresponding abstract for this article
                                            if pd.notnull(abstract):
                                                if include_abstracts == 'In title & abstract':
                                                    highlighted_abstract = highlight_terms(abstract, search_tokens)
                                                else:
                                                    highlighted_abstract = abstract 
                                                st.caption(f"Abstract: {highlighted_abstract}", unsafe_allow_html=True)
                                            else:
                                                st.caption(f"Abstract: No abstract")
                        else:
                            st.write("No articles found with the given keyword/phrase.")
                        status.update(label="Search completed!", state="complete", expanded=True)
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

                    publications_by_type = filtered_collection_df_authors['Publication type'].value_counts()

                    with st.expander('Click to expand', expanded=True): 
                        st.markdown('#### Publications by ' + selected_author)
                        st.write('*Please note that this database **may not show** all research outputs of the author.*')
                        types = st.multiselect('Publication type', filtered_collection_df_authors['Publication type'].unique(), filtered_collection_df_authors['Publication type'].unique(), key='original_authors')
                        filtered_collection_df_authors = filtered_collection_df_authors[filtered_collection_df_authors['Publication type'].isin(types)]
                        filtered_collection_df_authors = filtered_collection_df_authors.reset_index(drop=True)
                        publications_by_type = filtered_collection_df_authors['Publication type'].value_counts()
                        num_items_collections = len(filtered_collection_df_authors)
                        breakdown_string = ', '.join([f"{key}: {value}" for key, value in publications_by_type.items()])
                        st.write(f"**{num_items_collections}** sources found ({breakdown_string})")

                        true_count = filtered_collection_df_authors[filtered_collection_df_authors['Publication type']=='Journal article']['OA status'].sum()
                        total_count = len(filtered_collection_df_authors[filtered_collection_df_authors['Publication type']=='Journal article'])
                        if total_count == 0:
                            oa_ratio = 0.0
                        else:
                            oa_ratio = true_count / total_count * 100

                        citation_count = filtered_collection_df_authors['Citation'].sum()
                        st.write(f'Number of citations: **{int(citation_count)}**, Open access coverage (journal articles only): **{int(oa_ratio)}%**')

                        def convert_df(filtered_collection_df_authors):
                            return filtered_collection_df_authors.to_csv(index=False).encode('utf-8-sig')
                        download_filtered = filtered_collection_df_authors[['Publication type', 'Title', 'Abstract', 'Date published', 'Publisher', 'Journal', 'Link to publication', 'Zotero link', 'Citation']]
                        download_filtered['Abstract'] = download_filtered['Abstract'].str.replace('\n', ' ')
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
                                sort_by = st.radio('Sort by:', ('Publication date :arrow_down:', 'Citation'))
                                if sort_by == 'Publication date :arrow_down:' or filtered_collection_df_authors['Citation'].sum() == 0:
                                    filtered_collection_df_authors = filtered_collection_df_authors.sort_values(by=['Date published'], ascending=False)
                                    filtered_collection_df_authors =filtered_collection_df_authors.reset_index(drop=True)
                                else:
                                    filtered_collection_df_authors = filtered_collection_df_authors.sort_values(by=['Citation'], ascending=False)
                                    filtered_collection_df_authors =filtered_collection_df_authors.reset_index(drop=True)                                   
                                for index, row in filtered_collection_df_authors.iterrows():
                                    publication_type = row['Publication type']
                                    title = row['Title']
                                    authors = row['FirstName2']
                                    date_published = row['Date published']
                                    link_to_publication = row['Link to publication']
                                    zotero_link = row['Zotero link']
                                    citation = str(row['Citation']) if pd.notnull(row['Citation']) else '0'  
                                    citation = int(float(citation))
                                    citation_link = str(row['Citation_list']) if pd.notnull(row['Citation_list']) else ''
                                    citation_link = citation_link.replace('api.', '')

                                    published_by_or_in_dict = {
                                        'Journal article': 'Published in',
                                        'Magazine article': 'Published in',
                                        'Newspaper article': 'Published in',
                                        'Book': 'Published by',
                                    }

                                    publication_type = row['Publication type']

                                    published_by_or_in = published_by_or_in_dict.get(publication_type, '')
                                    published_source = str(row['Journal']) if pd.notnull(row['Journal']) else ''
                                    if publication_type == 'Book':
                                        published_source = str(row['Publisher']) if pd.notnull(row['Publisher']) else ''

                                    formatted_entry = (
                                        '**' + str(publication_type) + '**' + ': ' +
                                        str(title) + ' ' +
                                        '(by ' + '*' + str(authors) + '*' + ') ' +
                                        '(Publication date: ' + str(date_published) + ') ' +
                                        ('(' + published_by_or_in + ': ' + '*' + str(published_source) + '*' + ') ' if published_by_or_in else '') +
                                        '[[Publication link]](' + str(link_to_publication) + ') ' +
                                        '[[Zotero link]](' + str(zotero_link) + ') ' +
                                        ('Cited by [' + str(citation) + '](' + citation_link + ')' if citation > 0 else '')
                                    )
                                    formatted_entry = format_entry(row)
                                    st.write(f"{index + 1}) {formatted_entry}")

                            else:  # If toggle is on but no publications are available
                                st.write("No publication type selected.")

            # SEARCH IN COLLECTIONS
            elif search_option == "Search collection": 
                st.subheader('Search collection')

                df_csv_collections = df_duplicated.copy()
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

                    publications_by_type = filtered_collection_df['Publication type'].value_counts()

                    collection_link = df_csv_collections[df_csv_collections['Collection_Name'] == selected_collection]['Collection_Link'].iloc[0]
                    
                    with st.expander('Click to expand', expanded=True):
                        st.markdown('#### Collection theme: ' + selected_collection)
                        st.write(f"See the collection in [Zotero]({collection_link})")
                        types = st.multiselect('Publication type', filtered_collection_df['Publication type'].unique(),filtered_collection_df['Publication type'].unique(), key='original')
                        filtered_collection_df = filtered_collection_df[filtered_collection_df['Publication type'].isin(types)]
                        filtered_collection_df = filtered_collection_df.reset_index(drop=True)
                        publications_by_type = filtered_collection_df['Publication type'].value_counts()

                        download_collection = filtered_collection_df[['Publication type', 'Title', 'Abstract', 'Date published', 'Publisher', 'Journal', 'Link to publication', 'Zotero link', 'Citation']]
                        download_collection['Abstract'] = download_collection['Abstract'].str.replace('\n', ' ')
                        download_collection = download_collection.reset_index(drop=True)
                        def convert_df(download_collection):
                            return download_collection.to_csv(index=False).encode('utf-8-sig')
                        csv = convert_df(download_collection)
                        today = datetime.date.today().isoformat()
                        num_items_collections = len(filtered_collection_df)
                        breakdown_string = ', '.join([f"{key}: {value}" for key, value in publications_by_type.items()])
                        st.write(f"**{num_items_collections}** sources found ({breakdown_string})")

                        true_count = filtered_collection_df[filtered_collection_df['Publication type']=='Journal article']['OA status'].sum()
                        total_count = len(filtered_collection_df[filtered_collection_df['Publication type']=='Journal article'])
                        if total_count == 0:
                            oa_ratio = 0.0
                        else:
                            oa_ratio = true_count / total_count * 100

                        citation_count = filtered_collection_df['Citation'].sum()
                        st.write(f'Number of citations: **{int(citation_count)}**, Open access coverage (journal articles only): **{int(oa_ratio)}%**')

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

                            author_citation_df = filtered_collection_df.copy()
                            author_citation_df['Author_name'] = author_citation_df['FirstName2'].apply(lambda x: x.split(', ') if isinstance(x, str) and x else x)
                            author_citation_df = author_citation_df.explode('Author_name')
                            name_replacements = {}  # Assuming name_replacements is defined elsewhere in your code
                            author_citation_df['Author_name'] = author_citation_df['Author_name'].map(name_replacements).fillna(author_citation_df['Author_name'])
                            author_citations = author_citation_df.groupby('Author_name')['Citation'].sum().reset_index()
                            author_citations = author_citations.sort_values(by='Citation', ascending=False)
                            fig = px.bar(author_citations.head(10), x='Author_name', y='Citation',
                                        title=f'Top 10 Authors by Citation Count ({selected_collection})',
                                        labels={'Citation': 'Number of Citations', 'Author_name': 'Author'})
                            fig.update_layout(xaxis_tickangle=-45)
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
                                sort_by = st.radio('Sort by:', ('Publication date :arrow_down:', 'Citation'))
                                if sort_by == 'Publication date :arrow_down:' or filtered_collection_df['Citation'].sum() == 0:
                                    filtered_collection_df = filtered_collection_df.sort_values(by=['Date published'], ascending=False)
                                    filtered_collection_df = filtered_collection_df.reset_index(drop=True)
                                else:
                                    filtered_collection_df = filtered_collection_df.sort_values(by=['Citation'], ascending=False)
                                    filtered_collection_df = filtered_collection_df.reset_index(drop=True)
                                if num_items_collections > 20:
                                    show_first_20 = st.checkbox("Show only first 20 items (untick to see all)", value=True)
                                    if show_first_20:
                                        filtered_collection_df = filtered_collection_df.head(20)

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
                                    citation = str(row['Citation']) if pd.notnull(row['Citation']) else '0'  
                                    citation = int(float(citation))
                                    citation_link = str(row['Citation_list']) if pd.notnull(row['Citation_list']) else ''
                                    citation_link = citation_link.replace('api.', '')

                                    if publication_type == 'Journal article':
                                        published_by_or_in = 'Published in'
                                        published_source = str(row['Journal']) if pd.notnull(row['Journal']) else ''
                                    elif publication_type == 'Book':
                                        published_by_or_in = 'Published by'
                                        published_source = str(row['Publisher']) if pd.notnull(row['Publisher']) else ''
                                    else:
                                        published_by_or_in = ''
                                        published_source = ''

                                    formatted_entry = format_entry(row)
                                    st.write(f"{index + 1}) {formatted_entry}")
                            else:  # If toggle is on but no publications are available
                                st.write("No publication type selected.")

            elif search_option == "Publication types":
                st.subheader('Publication types')

                df_csv_types = df_dedup.copy()
                unique_types =  list(df_csv_types['Publication type'].unique())  # Adding an empty string as the first option The following bit was at the front [''] +
                selected_type = st.multiselect('Select a publication type', unique_types)

                if not selected_type or selected_type == '':
                    st.write('Pick a publication type to see items')
                else:
                    filtered_type_df = df_csv_types[df_csv_types['Publication type'].isin(selected_type)]
                    # filtered_collection_df = filtered_collection_df.sort_values(by='Date published', ascending=False).reset_index(drop=True)

                    # filtered_type_df['Date published'] = pd.to_datetime(filtered_type_df['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
                    filtered_type_df['Date published'] = (
                        filtered_type_df['Date published']
                        .str.strip()
                        .apply(lambda x: pd.to_datetime(x, utc=True, errors='coerce').tz_convert('Europe/London'))
                    )
                    filtered_type_df['Date published'] = filtered_type_df['Date published'].dt.strftime('%Y-%m-%d')
                    filtered_type_df['Date published'] = filtered_type_df['Date published'].fillna('')
                    filtered_type_df['No date flag'] = filtered_type_df['Date published'].isnull().astype(np.uint8)
                    filtered_type_df = filtered_type_df.sort_values(by=['No date flag', 'Date published'], ascending=[True, True])

                    # publications_by_type = filtered_collection_df['Publication type'].value_counts()
                    
                    with st.expander('Click to expand', expanded=True):
                        st.markdown('#### Publication type: ' + str(selected_type))
                        if selected_type == 'Thesis':
                            st.warning('Links to PhD theses catalouged by the British EThOS may not be working due to the [cyber incident at the British Library](https://www.bl.uk/cyber-incident/).')
                        
                        download_types = filtered_type_df[['Publication type', 'Title', 'Abstract', 'Date published', 'Publisher', 'Journal', 'Link to publication', 'Zotero link', 'Citation']]
                        download_types['Abstract'] = download_types['Abstract'].str.replace('\n', ' ')
                        download_types = download_types.reset_index(drop=True)

                        def convert_df(download_types):
                            return download_types.to_csv(index=False).encode('utf-8-sig')

                        csv = convert_df(download_types)
                        today = datetime.date.today().isoformat()
                        num_items_collections = len(filtered_type_df)
                        st.write(f"**{num_items_collections}** sources found")

                        true_count = filtered_type_df[filtered_type_df['Publication type']=='Journal article']['OA status'].sum()
                        total_count = len(filtered_type_df[filtered_type_df['Publication type']=='Journal article'])
                        if total_count == 0:
                            oa_ratio = 0.0
                        else:
                            oa_ratio = true_count / total_count * 100

                        citation_count = filtered_type_df['Citation'].sum()
                        st.write(f'Number of citations: **{int(citation_count)}**, Open access coverage (journal articles only): **{int(oa_ratio)}%**')

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
                            sort_by = st.radio('Sort by:', ('Publication date :arrow_down:', 'Citation'))
                            if sort_by == 'Publication date :arrow_down:' or filtered_type_df['Citation'].sum() == 0:
                                filtered_type_df = filtered_type_df.sort_values(by=['Date published'], ascending=False)
                                filtered_type_df = filtered_type_df.reset_index(drop=True)
                            else:
                                filtered_type_df = filtered_type_df.sort_values(by=['Citation'], ascending=False)
                                filtered_type_df = filtered_type_df.reset_index(drop=True)
                            if num_items_collections > 20:
                                show_first_20 = st.checkbox("Show only first 20 items (untick to see all)", value=True)
                                if show_first_20:
                                    filtered_type_df = filtered_type_df.head(20)                            

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
                                citation = str(row['Citation']) if pd.notnull(row['Citation']) else '0'  
                                citation = int(float(citation))
                                citation_link = str(row['Citation_list']) if pd.notnull(row['Citation_list']) else ''
                                citation_link = citation_link.replace('api.', '')

                                published_by_or_in_dict = {
                                    'Journal article': 'Published in',
                                    'Magazine article': 'Published in',
                                    'Newspaper article': 'Published in',
                                    'Book': 'Published by',
                                }

                                publication_type = row['Publication type']

                                published_by_or_in = published_by_or_in_dict.get(publication_type, '')
                                published_source = str(row['Journal']) if pd.notnull(row['Journal']) else ''
                                if publication_type == 'Book':
                                    published_source = str(row['Publisher']) if pd.notnull(row['Publisher']) else ''

                                formatted_entry = format_entry(row)
                                st.write(f"{index + 1}) {formatted_entry}")

            elif search_option == "Search journal":
                st.subheader('Search journal')

                df_csv = df_dedup.copy()
                df_csv = df_csv[df_csv['Publication type']=='Journal article']
                journal_counts = df_csv['Journal'].value_counts()
                unique_journals_sorted = journal_counts.index.tolist()
                journals = st.multiselect('Select a journal', unique_journals_sorted)

                if not journals:
                    st.write('Pick a journal name to see items')
                else:
                    selected_journal_df = df_csv[df_csv['Journal'].isin(journals)]

                    selected_journal_df['Date published'] = (
                        selected_journal_df['Date published']
                        .str.strip()
                        .apply(lambda x: pd.to_datetime(x, utc=True, errors='coerce').tz_convert('Europe/London'))
                    )
                    # selected_journal_df['Date published'] = pd.to_datetime(selected_journal_df['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
                    selected_journal_df['Date published'] = selected_journal_df['Date published'].dt.strftime('%Y-%m-%d')
                    selected_journal_df['Date published'] = selected_journal_df['Date published'].fillna('')
                    selected_journal_df['No date flag'] = selected_journal_df['Date published'].isnull().astype(np.uint8)
                    selected_journal_df = selected_journal_df.sort_values(by=['No date flag', 'Date published'], ascending=[True, True])

                    # publications_by_type = filtered_collection_df['Publication type'].value_counts()
                    
                    with st.expander('Click to expand', expanded=True):
                        st.markdown('#### Journal: ' + str(journals))

                        non_nan_id = selected_journal_df['ID'].count()

                        download_journal = selected_journal_df[['Publication type', 'Title', 'Abstract', 'Date published', 'Publisher', 'Journal', 'Link to publication', 'Zotero link', 'Citation']]
                        download_journal['Abstract'] = download_journal['Abstract'].str.replace('\n', ' ')
                        download_journal = download_journal.reset_index(drop=True)
                        def convert_df(download_journal):
                            return download_journal.to_csv(index=False).encode('utf-8-sig')

                        csv = convert_df(download_journal)
                        today = datetime.date.today().isoformat()
                        num_items_collections = len(selected_journal_df)
                        citation_count = selected_journal_df['Citation'].sum()
                        true_count = selected_journal_df['OA status'].sum()
                        total_count = len(selected_journal_df['OA status'])

                        if total_count == 0:
                            oa_ratio = 0.0
                        else:
                            oa_ratio = true_count / total_count * 100
                        st.write(f"Sources found: **{num_items_collections}**, Number of citations: **{int(citation_count)}**, Open access coverage: **{int(oa_ratio)}%**")
                        journal_citations = selected_journal_df.groupby('Journal')['Citation'].sum()

                        if len(journal_citations) >1:
                            journal_citations

                        a = f'selected_journal_{today}'
                        st.download_button('💾 Download', csv, (a+'.csv'), mime="text/csv", key='download-csv-4')

                        on = st.toggle('Generate dashboard')
                        if on and len (selected_journal_df) > 0:
                            st.info(f'Dashboard for {journals}')
                            
                            if non_nan_id !=0:

                                colcite1, colcite2, colcite3 = st.columns(3)

                                with colcite1:
                                    st.metric(label=f"Citation average", value=round((citation_count)/(num_items_collections)), label_visibility='visible', 
                                    help=f'''This is for items at least with 1 citation.
                                    Average citation (for all measured items): **{round((citation_count)/(non_nan_id))}**
                                    ''')
                                with colcite2:
                                    mean_citation = selected_journal_df['Citation'].median()
                                    st.metric(label=f"Citation median", value=round(mean_citation), label_visibility='visible', 
                                    help=f'''This is for items at least with 1 citation.
                                    ''')
                                with colcite3:
                                    mean_first_citaion = selected_journal_df['Year_difference'].mean()
                                    st.metric(label=f"First citation occurence (average in year)", value=round(mean_first_citaion), label_visibility='visible', 
                                    help=f'''First citation usually occurs **{round(mean_first_citaion)}** years after publication.
                                    ''')
                            else:
                                st.write('No citation found for selected journal(s)!')

                            type_df = selected_journal_df.copy()
                            collection_df = type_df.copy()
                            collection_df['Year'] = pd.to_datetime(collection_df['Date published']).dt.year
                            publications_by_year = collection_df['Year'].value_counts().sort_index()
                            fig_year_bar = px.bar(publications_by_year, x=publications_by_year.index, y=publications_by_year.values,
                                                labels={'x': 'Publication Year', 'y': 'Number of Publications'},
                                                title=f'Publications by Year for selected journal(s)')
                            st.plotly_chart(fig_year_bar)

                            publications_by_year = collection_df.groupby(['Year', 'Journal']).size().unstack().fillna(0)
                            publications_by_year = publications_by_year.cumsum(axis=0)

                            if len(journal_citations) >1:
                                journal_citations = journal_citations.reset_index()
                                journal_citations = journal_citations[journal_citations['Citation'] > 0]
                                journal_citations = journal_citations.sort_values(by='Citation', ascending=False)
                                fig = px.bar(journal_citations, x='Journal', y='Citation', title='Citations per Journal')
                                st.plotly_chart(fig, use_container_width = True)

                            fig_cumsum_line = px.line(publications_by_year, x=publications_by_year.index,
                                                    y=publications_by_year.columns,
                                                    labels={'x': 'Publication Year', 'y': 'Cumulative Publications'},
                                                    title='Cumulative Publications Over Years for selected journal(s)')
                            st.plotly_chart(fig_cumsum_line, use_container_width = True)

                            collection_author_df = type_df.copy()
                            collection_author_df['Author_name'] = collection_author_df['FirstName2'].apply(lambda x: x.split(', ') if isinstance(x, str) and x else x)
                            collection_author_df = collection_author_df.explode('Author_name')
                            collection_author_df.reset_index(drop=True, inplace=True)
                            collection_author_df['Author_name'] = collection_author_df['Author_name'].map(name_replacements).fillna(collection_author_df['Author_name'])
                            collection_author_df = collection_author_df['Author_name'].value_counts().head(10)
                            fig = px.bar(collection_author_df, x=collection_author_df.index, y=collection_author_df.values)
                            fig.update_layout(
                                title=f'Top 10 Authors by Publication Count for selected journal(s)',
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
                            plt.title(f"Word Cloud for Titles published in selected journal(s)")
                            plt.imshow(wordcloud)
                            plt.axis("off")
                            plt.show()
                            st.set_option('deprecation.showPyplotGlobalUse', False)
                            st.pyplot()

                        else:
                            sort_by = st.radio('Sort by:', ('Publication date :arrow_down:', 'Citation'))
                            if sort_by == 'Publication date :arrow_down:' or selected_journal_df['Citation'].sum() == 0:
                                selected_journal_df = selected_journal_df.sort_values(by=['Date published'], ascending=False)
                                selected_journal_df = selected_journal_df.reset_index(drop=True)
                            else:
                                selected_journal_df = selected_journal_df.sort_values(by=['Citation'], ascending=False)
                                selected_journal_df = selected_journal_df.reset_index(drop=True)
                            if num_items_collections > 20:
                                show_first_20 = st.checkbox("Show only first 20 items (untick to see all)", value=True)
                                if show_first_20:
                                    selected_journal_df = selected_journal_df.head(20)                            

                            articles_list = []  # Store articles in a list
                            for index, row in selected_journal_df.iterrows():
                                formatted_entry = format_entry(row)  # Assuming format_entry() is a function formatting each row
                                articles_list.append(formatted_entry)                     
                            
                            for index, row in selected_journal_df.iterrows():
                                publication_type = row['Publication type']
                                title = row['Title']
                                authors = row['FirstName2']
                                date_published = row['Date published'] 
                                link_to_publication = row['Link to publication']
                                zotero_link = row['Zotero link']
                                citation = str(row['Citation']) if pd.notnull(row['Citation']) else '0'  
                                citation = int(float(citation))
                                citation_link = str(row['Citation_list']) if pd.notnull(row['Citation_list']) else ''
                                citation_link = citation_link.replace('api.', '')

                                if publication_type == 'Journal article':
                                    published_by_or_in = 'Published in'
                                    published_source = str(row['Journal']) if pd.notnull(row['Journal']) else ''
                                elif publication_type == 'Book':
                                    published_by_or_in = 'Published by'
                                    published_source = str(row['Publisher']) if pd.notnull(row['Publisher']) else ''
                                else:
                                    published_by_or_in = ''
                                    published_source = ''

                                formatted_entry = format_entry(row)
                                st.write(f"{index + 1}) {formatted_entry}")

            elif search_option == "Publication year": 
                st.subheader('Items by publication year')

                with st.expander('Click to expand', expanded=True):                    
                    df_all = df_dedup.copy()
                    df_all['Date published2'] = (
                        df_all['Date published']
                        .str.strip()
                        .apply(lambda x: pd.to_datetime(x, utc=True, errors='coerce').tz_convert('Europe/London'))
                    )
                    # df_all['Date published'] = pd.to_datetime(df_all['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
                    # df_all
                    df_all['Date year'] = df_all['Date published2'].dt.strftime('%Y')
                    df_all['Date year'] = pd.to_numeric(df_all['Date year'], errors='coerce', downcast='integer')
                    numeric_years = df_all['Date year'].dropna()
                    current_year = date.today().year
                    min_y = numeric_years.min()
                    max_y = numeric_years.max()

                    df_all['Date published'] = (
                        df_all['Date published']
                        .str.strip()
                        .apply(lambda x: pd.to_datetime(x, utc=True, errors='coerce').tz_convert('Europe/London'))
                    )
                    # df_all['Date published'] = pd.to_datetime(df_all['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
                    df_all['Date published'] = df_all['Date published'].dt.strftime('%Y-%m-%d')
                    df_all['Date published'] = df_all['Date published'].fillna('')
                    df_all['No date flag'] = df_all['Date published'].isnull().astype(np.uint8)
                    df_all = df_all.sort_values(by=['No date flag', 'Date published'], ascending=[True, True])
                    df_all = df_all.sort_values(by=['Date published'], ascending=False)

                    current_year = date.today().year 
                    years = st.slider('Publication years between:', int(min(numeric_years)), int(max_y), (current_year, current_year+1), key='years')

                    filter = (df_all['Date year'] >= years[0]) & (df_all['Date year'] <= years[1])
                    df_all = df_all.loc[filter]

                    pub_types = df_all['Publication type'].unique()
                    selected_type = st.multiselect("Filter by publication type:", pub_types)
                    if selected_type:
                        df_all = df_all[df_all['Publication type'].isin(selected_type)]
                    
                    df_all = df_all.reset_index(drop=True)

                    # if years[0] == years[1] or years[0]==current_year:
                    #     st.markdown(f'#### Items published in **{int(years[0])}**')
                    # else:
                    #     st.markdown(f'#### Items published between **{int(years[0])}** and **{int(years[1])}**')

                    df_all_download = df_all.copy()
                    df_all_download = df_all_download[['Publication type', 'Title', 'Abstract', 'FirstName2', 'Link to publication', 'Zotero link', 'Date published', 'Citation']]
                    df_all_download['Abstract'] = df_all_download['Abstract'].str.replace('\n', ' ')
                    df_all_download = df_all_download.rename(columns={'FirstName2':'Author(s)'})
                    def convert_df(df_all_download):
                        return df_all_download.to_csv(index=False).encode('utf-8-sig') # not utf-8 because of the weird character,  Â cp1252
                    csv_selected = convert_df(df_all_download)
                    # csv = df_download
                    # # st.caption(collection_name)
                    a = 'intelligence-bibliography-items-between-' + str(years[0]) + '-' + str(years[1])
                    st.download_button('💾 Download selected items ', csv_selected, (a+'.csv'), mime="text/csv", key='download-csv-3')
                    number_of_items = len(df_all)

                    publications_by_type = df_all['Publication type'].value_counts()
                    breakdown_string = ', '.join([f"{key}: {value}" for key, value in publications_by_type.items()])

                    if years[0] == years[1] or years[0]==current_year:
                        colyear1, colyear2 = st.columns([2,3])
                        with colyear1:
                            st.metric(label=f"The number of sources published in **{int(years[0])}**", value=f'{number_of_items}', label_visibility='visible', 
                            help=f'({breakdown_string})')
                        with colyear2:
                            true_count = df_all[df_all['Publication type']=='Journal article']['OA status'].sum()
                            total_count = len(df_all[df_all['Publication type']=='Journal article'])
                            if total_count == 0:
                                oa_ratio = 0.0
                            else:
                                oa_ratio = true_count / total_count * 100

                            st.metric(label=f"Open access coverage", value=f"{int(oa_ratio)}%", label_visibility='visible', 
                            help=f'Journal articles only')                           
                    else:
                        colyear11, colyear22 = st.columns([2,3])
                        with colyear11:
                            st.metric(label=f"The number of sources published between **{int(years[0])}** and **{int(years[1])}**", value=f'{number_of_items}', label_visibility='visible', 
                            help=f'({breakdown_string})')    
                        with colyear22:
                            true_count = df_all[df_all['Publication type']=='Journal article']['OA status'].sum()
                            total_count = len(df_all[df_all['Publication type']=='Journal article'])
                            if total_count == 0:
                                oa_ratio = 0.0
                            else:
                                oa_ratio = true_count / total_count * 100                  
                            st.metric(label=f"Open access coverage", value=f"{int(oa_ratio)}%", label_visibility='visible', 
                            help=f'Journal articles only')    
                    st.warning('Items without a publication date are not listed here!')

                    dashboard_all = st.toggle('Generate dashboard')
                    if dashboard_all:
                        if dashboard_all and len(df_all) > 0: 
                            if abs(years[1]-years[0])>0 and years[0]<current_year:
                                st.info(f'Dashboard for items published between {int(years[0])} and {int(years[1])}')
                            else:
                                st.info(f'Dashboard for items published in {int(years[0])}')
                            collection_df = df_all.copy()
                            
                            publications_by_type = collection_df['Publication type'].value_counts()
                            if abs(years[1]-years[0])>0 and years[0]<current_year:
                                fig = px.bar(publications_by_type, x=publications_by_type.index, y=publications_by_type.values,
                                            labels={'x': 'Publication Type', 'y': 'Number of Publications'},
                                            title=f'Publications by Type between {int(years[0])} and {int(years[1])}')
                            else:
                                fig = px.bar(publications_by_type, x=publications_by_type.index, y=publications_by_type.values,
                                            labels={'x': 'Publication Type', 'y': 'Number of Publications'},
                                            title=f'Publications by Type in {int(years[0])}')
                            st.plotly_chart(fig)

                            if abs(years[1]-years[0])>0 and years[0]<current_year:
                                collection_df = df_all.copy()
                                collection_df['Year'] = pd.to_datetime(collection_df['Date published']).dt.year
                                publications_by_year = collection_df['Year'].value_counts().sort_index()
                                fig_year_bar = px.bar(publications_by_year, x=publications_by_year.index, y=publications_by_year.values,
                                                    labels={'x': 'Publication Year', 'y': 'Number of Publications'},
                                                    title=f'Publications by Year between {int(years[0])} and {int(years[1])}')
                                st.plotly_chart(fig_year_bar)
                            else:
                                collection_df = df_all.copy()
                                collection_df['Month'] = pd.to_datetime(collection_df['Date published']).dt.month
                                publications_by_year = collection_df['Month'].value_counts().sort_index()
                                fig_year_bar = px.bar(publications_by_year, x=publications_by_year.index, y=publications_by_year.values,
                                                    labels={'x': 'Publication Month', 'y': 'Number of Publications'},
                                                    title=f'Publications by Month in {int(years[0])}')
                                st.plotly_chart(fig_year_bar)

                            collection_author_df = df_all.copy()
                            collection_author_df['Author_name'] = collection_author_df['FirstName2'].apply(lambda x: x.split(', ') if isinstance(x, str) and x else x)
                            collection_author_df = collection_author_df.explode('Author_name')
                            collection_author_df.reset_index(drop=True, inplace=True)
                            collection_author_df['Author_name'] = collection_author_df['Author_name'].map(name_replacements).fillna(collection_author_df['Author_name'])
                            collection_author_df = collection_author_df['Author_name'].value_counts().head(10)
                            fig = px.bar(collection_author_df, x=collection_author_df.index, y=collection_author_df.values)
                            if abs(years[1]-years[0])>0 and years[0]<current_year:
                                fig.update_layout(
                                    title=f'Top 10 Authors by Publication Count between {int(years[0])} and {int(years[1])}',
                                    xaxis_title='Author',
                                    yaxis_title='Number of Publications',
                                    xaxis_tickangle=-45,
                                )
                            else:
                                fig.update_layout(
                                    title=f'Top 10 Authors by Publication Count in {int(years[0])}',
                                    xaxis_title='Author',
                                    yaxis_title='Number of Publications',
                                    xaxis_tickangle=-45,
                                )
                            st.plotly_chart(fig)

                            author_df = df_all.copy()
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
                            if abs(years[1]-years[0])>0 and years[0]<current_year:
                                plt.title(f"Word Cloud for Titles between {int(years[0])} and {int(years[1])}")
                            else:
                                plt.title(f"Word Cloud for Titles in {int(years[0])}")
                            plt.imshow(wordcloud)
                            plt.axis("off")
                            plt.show()
                            st.set_option('deprecation.showPyplotGlobalUse', False)
                            st.pyplot()
                    else:
                        sort_by = st.radio('Sort by:', ('Publication date :arrow_down:', 'Citation'))
                        if sort_by == 'Publication date :arrow_down:' or df_all['Citation'].sum() == 0:
                            df_all = df_all.sort_values(by=['Date published'], ascending=False)
                            df_all = df_all.reset_index(drop=True)
                        else:
                            df_all = df_all.sort_values(by=['Citation'], ascending=False)
                            df_all = df_all.reset_index(drop=True)
                        if number_of_items > 20:
                            show_first_20 = st.checkbox("Show only first 20 items (untick to see all)", value=True, key='all_items')
                            if show_first_20:
                                df_all = df_all.head(20)
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

            elif search_option == "Cited papers": 
                st.subheader('Cited items in the library')

                with st.expander('Click to expand', expanded=True):                    
                    df_cited = df_dedup.copy()
                    non_nan_id = df_cited['ID'].count()
                    df_cited = df_cited[(df_cited['Citation'].notna()) & (df_cited['Citation'] != 0)]
                    df_cited = df_cited.reset_index(drop=True)

                    citation_type = st.radio('Select:', ('All citations', 'Trends'))
                    if citation_type=='All citations':
                        df_cited = df_cited.reset_index(drop=True)
                    else:
                        current_year = datetime.datetime.now().year
                        df_cited = df_cited[(df_cited['Last_citation_year'] == current_year) | (df_cited['Last_citation_year'] == current_year - 1)]
                        df_cited = df_cited[(df_cited['Publication_year'] == current_year) | (df_cited['Publication_year'] == current_year - 1)]
                        note = st.info(f'''
                        The trends section shows the citations occured in the last two years ({current_year - 1}-{current_year}) to the papers published in the same period. 
                        ''')

                    max_value = int(df_cited['Citation'].max())
                    min_value = 1
                    selected_range = st.slider('Select a citation range:', min_value, max_value, (min_value, max_value), key='')
                    filter = (df_cited['Citation'] >= selected_range[0]) & (df_cited['Citation'] <= selected_range[1])
                    df_cited = df_cited.loc[filter]

                    df_cited['Date published2'] = (
                        df_cited['Date published']
                        .str.strip()
                        .apply(lambda x: pd.to_datetime(x, utc=True, errors='coerce').tz_convert('Europe/London'))
                    )
                    df_cited['Date year'] = df_cited['Date published2'].dt.strftime('%Y')
                    df_cited['Date year'] = pd.to_numeric(df_cited['Date year'], errors='coerce', downcast='integer')

                    df_cited['Date published'] = (
                        df_cited['Date published']
                        .str.strip()
                        .apply(lambda x: pd.to_datetime(x, utc=True, errors='coerce').tz_convert('Europe/London'))
                    )
                    df_cited['Date published'] = df_cited['Date published'].dt.strftime('%Y-%m-%d')
                    df_cited['Date published'] = df_cited['Date published'].fillna('')
                    df_cited['No date flag'] = df_cited['Date published'].isnull().astype(np.uint8)
                    df_cited = df_cited.sort_values(by=['No date flag', 'Date published'], ascending=[True, True])
                    df_cited = df_cited.sort_values(by=['Date published'], ascending=False)

                    # pub_types = df_cited['Publication type'].unique()
                    # selected_type = st.multiselect("Filter by publication type:", pub_types)
                    # if selected_type:
                    #     df_cited = df_cited[df_cited['Publication type'].isin(selected_type)]
                    
                    df_cited = df_cited.reset_index(drop=True)

                    df_cited_download = df_cited.copy()
                    df_cited_download = df_cited_download[['Publication type', 'Title', 'Abstract', 'FirstName2', 'Link to publication', 'Zotero link', 'Date published', 'Citation']]
                    df_cited_download['Abstract'] = df_cited_download['Abstract'].str.replace('\n', ' ')
                    df_cited_download = df_cited_download.rename(columns={'FirstName2':'Author(s)'})
                    def convert_df(df_cited_download):
                        return df_cited_download.to_csv(index=False).encode('utf-8-sig') # not utf-8 because of the weird character,  Â cp1252
                    csv_selected = convert_df(df_cited_download)
                    # csv = df_download
                    # # st.caption(collection_name)
                    a = 'cited-items-'
                    st.download_button('💾 Download selected items ', csv_selected, (a+'.csv'), mime="text/csv", key='download-csv-3')
                    number_of_items = len(df_cited)

                    colcit1, colcit2 = st.columns([2,4])
                    with colcit1:
                        citation_count = df_cited['Citation'].sum()
                        publications_by_type = df_cited['Publication type'].value_counts()
                        breakdown_string = ', '.join([f"{key}: {value}" for key, value in publications_by_type.items()])
                        st.metric(label=f"The number of citations for **{number_of_items}** items", value=int(citation_count), label_visibility='visible', 
                        help=f'''Out of the **{non_nan_id}** items measured for citations, **{number_of_items}** received at least 1 citation.
                        ''')
                    with colcit2:
                        true_count = df_cited[df_cited['Publication type']=='Journal article']['OA status'].sum()
                        total_count = len(df_cited[df_cited['Publication type']=='Journal article'])
                        if total_count == 0:
                            oa_ratio = 0.0
                        else:
                            oa_ratio = true_count / total_count * 100

                        st.metric(label=f"Open access coverage", value=f"{int(oa_ratio)}%", label_visibility='visible', 
                        help=f'Journal articles only') 

                    st.warning('Items without a citation are not listed here! Citation data comes from [OpenAlex](https://openalex.org/).')

                    dashboard_all = st.toggle('Generate dashboard')
                    if dashboard_all:
                        if dashboard_all and len(df_cited) > 0: 
                            st.info(f'Dashboard for cited items in the library')

                            colcite1, colcite2, colcite3 = st.columns(3) 

                            with colcite1:
                                st.metric(label=f"Citation average", value=round((citation_count)/(number_of_items)), label_visibility='visible', 
                                help=f'''This is for items at least with 1 citation.
                                Average citation (for all measured items): **{round((citation_count)/(non_nan_id))}**
                                ''')
                            with colcite2:
                                mean_citation = df_cited['Citation'].median()
                                st.metric(label=f"Citation median", value=round(mean_citation), label_visibility='visible', 
                                help=f'''This is for items at least with 1 citation.
                                ''')
                            with colcite3: 
                                mean_first_citaion = df_cited['Year_difference'].mean()
                                st.metric(label=f"First citation occurence (average in year)", value=round(mean_first_citaion), label_visibility='visible', 
                                help=f'''First citation usually occurs **{round(mean_first_citaion)}** years after publication.
                                ''')

                            citation_distribution = df_cited['Citation'].value_counts().sort_index().reset_index()
                            citation_distribution.columns = ['Number of Citations', 'Number of Articles']

                            fig = px.scatter(citation_distribution, x='Number of Citations', y='Number of Articles', 
                                            title='Distribution of Citations Across Articles', 
                                            labels={'Number of Citations': 'Number of Citations', 'Number of Articles': 'Number of Articles'})

                            # Optional: You can customize scatter plot appearance using various parameters
                            # For example:
                            fig.update_traces(marker=dict(color='red', size=7, opacity=0.5), selector=dict(mode='markers'))
                            st.plotly_chart(fig)

                            fig = go.Figure(data=go.Scatter(x=df_cited['Year_difference'], y=[0] * len(df_cited['Year_difference']), mode='markers'))
                            # Customize layout
                            fig.update_layout(
                                title='First citation occurence (first citation occurs after years)',
                                xaxis_title='Year Difference',
                                yaxis_title='',                            )

                            # Display the Plotly chart using Streamlit
                            st.plotly_chart(fig)

                            collection_df = df_cited.copy()
                            collection_df['Year'] = pd.to_datetime(collection_df['Date published']).dt.year
                            publications_by_year = collection_df['Year'].value_counts().sort_index()
                            fig_year_bar = px.bar(publications_by_year, x=publications_by_year.index, y=publications_by_year.values,
                                                labels={'x': 'Publication Year', 'y': 'Number of Publications'},
                                                title=f'Publications over time')
                            st.plotly_chart(fig_year_bar)

                            collection_df['Author_name'] = collection_df['FirstName2'].apply(lambda x: x.split(', ') if isinstance(x, str) and x else x)
                            collection_df = collection_df.explode('Author_name')
                            collection_df.reset_index(drop=True, inplace=True)
                            collection_df['Author_name'] = collection_df['Author_name'].map(name_replacements).fillna(collection_df['Author_name'])
                            collection_df = collection_df['Author_name'].value_counts().head(10)
                            fig = px.bar(collection_df, x=collection_df.index, y=collection_df.values)
                            fig.update_layout(
                                title=f'Top 10 Authors by Publication Count',
                                xaxis_title='Author',
                                yaxis_title='Number of Publications',
                                xaxis_tickangle=-45,
                            )
                            st.plotly_chart(fig)

                            collection_df = df_cited.copy()
                            collection_df['Author_name'] = collection_df['FirstName2'].apply(lambda x: x.split(', ') if isinstance(x, str) and x else x)
                            collection_df = collection_df.explode('Author_name')
                            name_replacements = {}  # Assuming name_replacements is defined elsewhere in your code
                            collection_df['Author_name'] = collection_df['Author_name'].map(name_replacements).fillna(collection_df['Author_name'])
                            author_citations = collection_df.groupby('Author_name')['Citation'].sum().reset_index()
                            author_citations = author_citations.sort_values(by='Citation', ascending=False)
                            fig = px.bar(author_citations.head(20), x='Author_name', y='Citation',
                                        title=f'Top 20 Authors by Citation Count',
                                        labels={'Citation': 'Number of Citations', 'Author_name': 'Author'})
                            fig.update_layout(xaxis_tickangle=-45)
                            st.plotly_chart(fig)
       
                            author_df = df_cited.copy()
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
                            plt.title(f"Word Cloud for cited papers")
                            plt.imshow(wordcloud)
                            plt.axis("off")
                            plt.show()
                            st.set_option('deprecation.showPyplotGlobalUse', False)
                            st.pyplot()
                    else: 
                        sort_by = st.radio('Sort by:', ('Publication date :arrow_down:', 'Citation'))
                        if sort_by == 'Publication date :arrow_down:' or df_cited['Citation'].sum() == 0:
                            df_cited = df_cited.sort_values(by=['Date published'], ascending=False)
                            df_cited = df_cited.reset_index(drop=True)
                        else:  
                            df_cited = df_cited.sort_values(by=['Citation'], ascending=False)
                            df_cited = df_cited.reset_index(drop=True)
                        if number_of_items > 20:
                            show_first_20 = st.checkbox("Show only first 20 items (untick to see all)", value=True, key='all_items')
                            if show_first_20:
                                df_cited = df_cited.head(20)
                        articles_list = []  # Store articles in a list
                        abstracts_list = [] #Store abstracts in a list
                        for index, row in df_cited.iterrows():
                            formatted_entry = format_entry(row)
                            articles_list.append(formatted_entry)  # Append formatted entry to the list
                            abstract = row['Abstract']
                            abstracts_list.append(abstract if pd.notnull(abstract) else 'N/A')
                        for i, article in enumerate(articles_list, start=1):
                            # Display the article with highlighted search terms
                            st.markdown(f"{i}. {article}", unsafe_allow_html=True) 

            # OVERVIEW
            st.header('Overview', anchor=None)
            tab11, tab12, tab13 = st.tabs(['Recently added items', 'Recently published items', 'Top cited items'])
            with tab11:
                st.markdown('#### Recently added or updated items')
                df['Abstract'] = df['Abstract'].str.strip()
                df['Abstract'] = df['Abstract'].fillna('No abstract')
                
                # df_download = df.iloc[:, [0,1,2,3,4,5,6,9]] 
                # df_download = df_download[['Title', 'Publication type', 'Authors', 'Abstract', 'Link to publication', 'Zotero link', 'Date published', 'Date added']]
                # def convert_df(df):
                #     return df.to_csv(index=False).encode('utf-8-sig') # not utf-8 because of the weird character,  Â cp1252
                # csv = convert_df(df_download)
                # # csv = df_download
                # # # st.caption(collection_name)
                # today = datetime.date.today().isoformat()
                # a = 'recently-added-' + today
                # st.download_button('💾 Download recently added items', csv, (a+'.csv'), mime="text/csv", key='download-csv-3')
                
                display = st.checkbox('Display theme and abstract')

                def format_row(row):
                    if row['Publication type'] == 'Book chapter' and row['Book_title']:
                        return ('**' + row['Publication type'] + '**' + ': ' + row['Title'] + ', ' +
                                ' (by ' + '*' + row['Authors'] + '*' + ') ' +
                                ' (Published on: ' + row['Date published'] + ') ' +
                                '[[Publication link]]' + '(' + row['Link to publication'] + ')' +
                                "[[Zotero link]]" + '(' + row['Zotero link'] + ') ' +
                                '(In: ' + row['Book_title'] + ')')  # Including Book Title for book chapters
                    else:
                        return ('**' + row['Publication type'] + '**' + ': ' + row['Title'] + ', ' +
                                ' (by ' + '*' + row['Authors'] + '*' + ') ' +
                                ' (Published on: ' + row['Date published'] + ') ' +
                                '[[Publication link]]' + '(' + row['Link to publication'] + ')' +
                                "[[Zotero link]]" + '(' + row['Zotero link'] + ')')
                df_last = df.apply(format_row, axis=1)

                # df_last = ('**'+ df['Publication type']+ '**'+ ': ' + df['Title'] +', ' +                        
                #             ' (by ' + '*' + df['Authors'] + '*' + ') ' +
                #             ' (Published on: ' + df['Date published']+') ' +
                #             '[[Publication link]]'+ '('+ df['Link to publication'] + ')' +
                #             "[[Zotero link]]" +'('+ df['Zotero link'] + ')' 
                #             )
                
                row_nu_1 = len(df_last)
                for i in range(row_nu_1):
                    publication_type = df['Publication type'].iloc[i]
                    
                    if publication_type in ["Journal article", "Magazine article", 'Newspaper article']:
                        formatted_row = ('**'+ df['Publication type'].iloc[i]+ '**'+ ': ' + df['Title'].iloc[i] +', ' +                        
                                        ' (by ' + '*' + df['Authors'].iloc[i] + '*' + ') ' +
                                        ' (Published on: ' + df['Date published'].iloc[i]+') ' +
                                        " (Published in: " + "*" + df['Pub_venue'].iloc[i] + "*" + ') '+
                                        '[[Publication link]]'+ '('+ df['Link to publication'].iloc[i] + ')' +
                                        "[[Zotero link]]" +'('+ df['Zotero link'].iloc[i] + ')' 
                                        )
                        st.write(f"{i+1}) " + formatted_row)
                    
                    elif publication_type == 'Book chapter':
                        formatted_row = ('**'+ df['Publication type'].iloc[i]+ '**'+ ': ' + df['Title'].iloc[i] + 
                                        ' (in: ' + '*'+  df['Book_title'].iloc[i] + ')'+ '*'+ ', ' +                        
                                        ' (by ' + '*' + df['Authors'].iloc[i] + '*' + ') ' +
                                        ' (Published on: ' + df['Date published'].iloc[i]+') ' +
                                        '[[Publication link]]'+ '('+ df['Link to publication'].iloc[i] + ')' +
                                        "[[Zotero link]]" +'('+ df['Zotero link'].iloc[i] + ')'
                                        )
                        st.write(f"{i+1}) " + formatted_row)
                    
                    else:
                        formatted_row = ('**'+ df['Publication type'].iloc[i]+ '**'+ ': ' + df['Title'].iloc[i] +', ' +                        
                                        ' (by ' + '*' + df['Authors'].iloc[i] + '*' + ') ' +
                                        ' (Published on: ' + df['Date published'].iloc[i]+') ' +
                                        '[[Publication link]]'+ '('+ df['Link to publication'].iloc[i] + ')' +
                                        "[[Zotero link]]" +'('+ df['Zotero link'].iloc[i] + ')' 
                                        )
                        st.write(f"{i+1}) " + formatted_row)

                    if display:
                        a = ''
                        b = ''
                        c = ''
                        if 'Name_x' in df:
                            a = '[' + '[' + df['Name_x'].iloc[i] + ']' + '(' + df['Link_x'].iloc[i] + ')' + ']'
                            if df['Name_x'].iloc[i] == '':
                                a = ''
                        if 'Name_y' in df:
                            b = '[' + '[' + df['Name_y'].iloc[i] + ']' + '(' + df['Link_y'].iloc[i] + ')' + ']'
                            if df['Name_y'].iloc[i] == '':
                                b = ''
                        if 'Name' in df:
                            c = '[' + '[' + df['Name'].iloc[i] + ']' + '(' + df['Link'].iloc[i] + ')' + ']'
                            if df['Name'].iloc[i] == '':
                                c = ''
                        st.caption('Theme(s):  \n ' + a + ' ' + b + ' ' + c)
                        if not any([a, b, c]):
                            st.caption('No theme to display!')
                        
                        st.caption('Abstract: ' + df['Abstract'].iloc[i])

            with tab12:
                st.markdown('#### Recently published items')
                display2 = st.checkbox('Display abstracts', key='recently_published')
                df_intro = df_dedup.copy()
                df_intro['Date published'] = pd.to_datetime(df_intro['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
                current_date = datetime.datetime.now(datetime.timezone.utc).astimezone(datetime.timezone(datetime.timedelta(hours=1)))  # Current date in London timezone
                df_intro = df_intro[df_intro['Date published'] <= current_date]
                df_intro['Date published'] = df_intro['Date published'].dt.strftime('%Y-%m-%d')
                df_intro['Date published'] = df_intro['Date published'].fillna('')
                df_intro['No date flag'] = df_intro['Date published'].isnull().astype(np.uint8)
                df_intro = df_intro.sort_values(by=['No date flag', 'Date published'], ascending=[True, True])
                df_intro = df_intro.sort_values(by=['Date published'], ascending=False)
                df_intro = df_intro.reset_index(drop=True)
                df_intro = df_intro.head(10)
                # articles_list = [format_entry(row) for _, row in df_intro.iterrows()]
                articles_list = [format_entry(row, include_citation=False) for _, row in df_intro.iterrows()]
                for index, formatted_entry in enumerate(articles_list):
                    st.write(f"{index + 1}) {formatted_entry}")
                    if display2:
                        st.caption(df_intro.iloc[index]['Abstract'])
            with tab13:
                @st.cache_resource(ttl=5000)  # Cache the resource for 5000 seconds
                def load_data():
                    df_top = df_dedup.copy()
                    df_top['Date published'] = (
                        df_top['Date published']
                        .str.strip()
                        .apply(lambda x: pd.to_datetime(x, utc=True, errors='coerce').tz_convert('Europe/London'))
                    )
                    df_top['Date published'] = df_top['Date published'].dt.strftime('%Y-%m-%d')
                    df_top['Date published'] = df_top['Date published'].fillna('')
                    df_top['No date flag'] = df_top['Date published'].isnull().astype(np.uint8)
                    df_top = df_top.sort_values(by=['Citation'], ascending=False)
                    df_top = df_top.reset_index(drop=True)
                    return df_top

                df_top = load_data() 

                st.markdown('#### Top cited items')
                display3 = st.checkbox('Display abstracts', key='top_cited')

                df_top_display = df_top.head(10)  # Take top 5 items for display
                articles_list = [format_entry(row) for _, row in df_top_display.iterrows()]

                for index, formatted_entry in enumerate(articles_list):
                    st.write(f"{index + 1}) {formatted_entry}")
                    if display3:
                        st.caption(df_top_display.iloc[index]['Abstract'])

            st.header('All items in database', anchor=False)
            with st.expander('Click to expand', expanded=False):
                df_all_items = df_dedup.copy()
                df_all_items = df_all_items[['Publication type', 'Title', 'Abstract', 'Date published', 'Publisher', 'Journal', 'Link to publication', 'Zotero link', 'Citation']]

                download_all = df_all_items[['Publication type', 'Title', 'Abstract', 'Date published', 'Publisher', 'Journal', 'Link to publication', 'Zotero link', 'Citation']]
                download_all['Abstract'] = download_all['Abstract'].str.replace('\n', ' ')
                download_all = download_all.reset_index(drop=True)
                def convert_df(download_all):
                    return download_all.to_csv(index=False).encode('utf-8-sig') # not utf-8 because of the weird character,  Â cp1252
                csv = convert_df(download_all)
                # csv = df_download
                # # st.caption(collection_name)
                today = datetime.date.today().isoformat()
                a = 'intelligence-bibliography-all-' + today
                st.download_button('💾 Download all items', csv, (a+'.csv'), mime="text/csv", key='download-csv-2')
                df_all_items

                df_added = df_dedup.copy()
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
                    width=500,
                    height=600,  # Adjust the height here
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
                st.subheader('Growth of the library')
                st.altair_chart(cumulative_chart + data_labels, use_container_width=True)
                    

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

                conn = st.connection("gsheets", type=GSheetsConnection)

                # Read the first spreadsheet
                df_gs = conn.read(spreadsheet='https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/edit#gid=0')

                # Read the second spreadsheet
                df_forms = conn.read(spreadsheet='https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/edit#gid=1941981997')
                df_forms = df_forms.rename(columns={'Event name':'event_name', 'Event organiser':'organiser','Link to the event':'link','Date of event':'date', 'Event venue':'venue', 'Details':'details'})
                df_forms = df_forms.drop(columns=['Timestamp'])

                # Convert and format dates in df_gs
                df_gs['date'] = pd.to_datetime(df_gs['date'])
                df_gs['date_new'] = df_gs['date'].dt.strftime('%Y-%m-%d')

                # Convert and format dates in df_forms
                df_forms['date'] = pd.to_datetime(df_forms['date'])
                df_forms['date_new'] = df_forms['date'].dt.strftime('%Y-%m-%d')
                df_forms['month'] = df_forms['date'].dt.strftime('%m')
                df_forms['year'] = df_forms['date'].dt.strftime('%Y')
                df_forms['month_year'] = df_forms['date'].dt.strftime('%Y-%m')
                df_forms.sort_values(by='date', ascending=True, inplace=True)
                df_forms = df_forms.drop_duplicates(subset=['event_name', 'link', 'date'], keep='first')

                # Fill missing values in df_forms
                df_forms['details'] = df_forms['details'].fillna('No details')
                df_forms = df_forms.fillna('')

                # Concatenate df_gs and df_forms
                df_gs = pd.concat([df_gs, df_forms], axis=0)
                df_gs = df_gs.reset_index(drop=True)
                df_gs = df_gs.drop_duplicates(subset=['event_name', 'link', 'date'], keep='first')

                # Sort the concatenated dataframe by date_new
                df_gs = df_gs.sort_values(by='date_new', ascending=True)

                # Filter events happening today or in the future
                today = dt.date.today()
                df_gs['date'] = pd.to_datetime(df_gs['date'], dayfirst=True)  # Ensure 'date' is datetime
                filter = df_gs['date'] >= pd.to_datetime(today)
                df_gs = df_gs[filter]

                # Display the filtered dataframe
                df_gs = df_gs.loc[filter]
                df_gs = df_gs.fillna('')
                df_gs = df_gs.head(3)
                if df_gs['event_name'].any() in ("", [], None, 0, False):
                    st.write('No upcoming event!')
                df_gs1 = ('['+ df_gs['event_name'] + ']'+ '('+ df_gs['link'] + ')'', organised by ' + '**' + df_gs['organiser'] + '**' + '. Date: ' + df_gs['date_new'] + ', Venue: ' + df_gs['venue'])
                row_nu = len(df_gs.index)
                for i in range(row_nu):
                    st.write(''+str(i+1)+') '+ df_gs1.iloc[i])
                st.write('Visit the [Events on intelligence](https://intelligence.streamlit.app/Events) page to see more!')
                
                st.markdown('##### Next conference')
                df_con = conn.read(spreadsheet='https://docs.google.com/spreadsheets/d/10ezNUOUpzBayqIMJWuS_zsvwklxP49zlfBWsiJI6aqI/edit#gid=939232836')
                df_con['date'] = pd.to_datetime(df_con['date'])
                df_con['date_new'] = df_con['date'].dt.strftime('%Y-%m-%d')
                df_con['date_new'] = pd.to_datetime(df_con['date'], dayfirst = True).dt.strftime('%d/%m/%Y')
                df_con['date_new_end'] = pd.to_datetime(df_con['date_end'], dayfirst = True).dt.strftime('%d/%m/%Y')
                df_con.sort_values(by='date', ascending = True, inplace=True)
                df_con['details'] = df_con['details'].fillna('No details')
                df_con['location'] = df_con['location'].fillna('No details')
                df_con = df_con.fillna('')
                df_con['date_end'] = pd.to_datetime(df_con['date'], dayfirst=True)     
                filter = df_con['date_end']>=pd.to_datetime(today)
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
        st.header('Dashboard', anchor=False)
        on_main_dashboard = st.toggle('Display dashboard')
        
        if on_main_dashboard:

            # number0 = st.slider('Select a number collections', 3,30,15)
            # df_collections_2.set_index('Name', inplace=True)
            # df_collections_2 = df_collections_2.sort_values(['Number'], ascending=[False])
            # plot= df_collections_2.head(number0+1)
            # # st.bar_chart(plot['Number'].sort_values(), height=600, width=600, use_container_width=True)
            # plot = plot.reset_index()

            # plot = plot[plot['Name']!='01 Intelligence history']
            # fig = px.bar(plot, x='Name', y='Number', color='Name')
            # fig.update_layout(
            #     autosize=False,
            #     width=600,
            #     height=600,)
            # fig.update_layout(title={'text':'Top ' + str(number0) + ' collections in the library', 'y':0.95, 'x':0.4, 'yanchor':'top'})
            # st.plotly_chart(fig, use_container_width = True)

            df_csv = df_duplicated.copy()
            df_collections_2 =df_csv.copy()

            df_csv = df_dedup.copy()
            df_csv = df_csv.reset_index(drop=True)

            df_csv['Date published'] = (
                df_csv['Date published']
                .str.strip()
                .apply(lambda x: pd.to_datetime(x, utc=True, errors='coerce').tz_convert('Europe/London'))
            )
            
            # df_csv['Date published'] = pd.to_datetime(df_csv['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
            df_csv['Date year'] = df_csv['Date published'].dt.strftime('%Y')
            df_csv['Date year'] = df_csv['Date year'].fillna('No date')

            df = df_csv.copy()
            df_year=df_csv['Date year'].value_counts()
            df_year=df_year.reset_index()
            df_year=df_year.rename(columns={'index':'Publication year','Date year':'Count'})

            # TEMPORARY SOLUTION FOR COLUMN NAME CHANGE ERROR
            df_year.columns = ['Publication year', 'Count']
            # TEMP SOLUTION ENDS

            df_year.drop(df_year[df_year['Publication year']== 'No date'].index, inplace = True)
            df_year=df_year.sort_values(by='Publication year', ascending=True)
            df_year=df_year.reset_index(drop=True)
            max_y = int(df_year['Publication year'].max())
            min_y = int(df_year['Publication year'].min())

            df_collections_2['Date published'] = (
                df_collections_2['Date published']
                .str.strip()
                .apply(lambda x: pd.to_datetime(x, utc=True, errors='coerce').tz_convert('Europe/London'))
            )
            
            # df_collections_2['Date published'] = pd.to_datetime(df_collections_2['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
            df_collections_2['Date year'] = df_collections_2['Date published'].dt.strftime('%Y')
            df_collections_2['Date year'] = df_collections_2['Date year'].fillna('No date') 
 
            with st.expander('**Select filters**', expanded=False):
                types = st.multiselect('Publication type', df_csv['Publication type'].unique(), df_csv['Publication type'].unique())

                df_journals = df_dedup.copy()
                df_journals = df_journals[df_journals['Publication type'] == 'Journal article']
                journal_counts = df_journals['Journal'].value_counts()
                unique_journals_sorted = journal_counts.index.tolist()
                journals = st.multiselect('Select a journal', unique_journals_sorted, key='big_dashboard_journals')                 

                years = st.slider('Publication years between:', min_y, max_y+1, (min_y,max_y+1), key='years2')
                if st.button('Update dashboard'):
                    df_csv = df_csv[df_csv['Publication type'].isin(types)]
                    if journals:
                        df_csv = df_csv[df_csv['Journal'].isin(journals)]
                    else:
                        df_csv = df_csv.copy()
                    df_csv = df_csv[df_csv['Date year'] !='No date']
                    filter = (df_csv['Date year'].astype(int)>=years[0]) & (df_csv['Date year'].astype(int)<years[1])

                    df_csv = df_csv.loc[filter]
                    df_year=df_csv['Date year'].value_counts()
                    df_year=df_year.reset_index()
                    df_year=df_year.rename(columns={'index':'Publication year','Date year':'Count'})
                    df_year.drop(df_year[df_year['Publication year']== 'No date'].index, inplace = True)
                    df_year=df_year.sort_values(by='Publication year', ascending=True)
                    df_year=df_year.reset_index(drop=True)

                    df_collections_2 = df_collections_2[df_collections_2['Publication type'].isin(types)]
                    if journals:
                        df_collections_2 = df_collections_2[df_collections_2['Journal'].isin(journals)]
                    else:
                        df_collections_2 = df_collections_2.copy()                    
                    df_collections_2 = df_collections_2[df_collections_2['Date year'] !='No date']
                    filter_collection = (df_collections_2['Date year'].astype(int)>=years[0]) & (df_collections_2['Date year'].astype(int)<years[1])
                    df_collections_2 = df_collections_2.loc[filter_collection]

            if df_csv['Title'].any() in ("", [], None, 0, False):
                st.warning('No data to visualise. Select a correct parameter.')

            else:                
                ## COLLECTIONS IN THE LIBRARY
                
                st.markdown(f'#### Intelligence studies bibliography dashboard (publications between {years[0]} and {years[1]})')

                df_collections_21 = df_collections_2.copy()
                df_collections_21 = df_collections_21['Collection_Name'].value_counts().reset_index()
                df_collections_21.columns = ['Collection_Name', 'Number_of_Items']
                number0 = st.slider('Select a number collections', 3,30,15, key='slider01')
                plot= df_collections_21.head(number0+1)
                plot = plot[plot['Collection_Name']!='01 Intelligence history']
                fig = px.bar(plot, x='Collection_Name', y='Number_of_Items', color='Collection_Name')
                fig.update_layout(
                    autosize=False,
                    width=600,
                    height=600,)
                fig.update_layout(title={'text':'Top ' + str(number0) + ' collections in the library', 'y':0.95, 'x':0.4, 'yanchor':'top'})
                st.plotly_chart(fig, use_container_width = True)

                df_collections_22 = df_collections_2.copy()
                collection_counts = df_collections_22.groupby(['Date year', 'Collection_Name']).size().unstack().fillna(0)
                collection_counts = collection_counts.reset_index()
                collection_counts.iloc[:, 1:] = collection_counts.iloc[:, 1:].cumsum()

                selected_collections = df_collections_21.head(number0 + 1)['Collection_Name'].tolist()
                collection_counts_filtered = collection_counts[['Date year'] + selected_collections]
                column_to_exclude = '01 Intelligence history'
                if column_to_exclude in selected_collections:
                    selected_collections.remove(column_to_exclude)

                # Streamlit app
                st.markdown(f'#### Cumulative changes in collection over years')

                collection_counts_filtered = collection_counts[['Date year'] + selected_collections]
                collection_counts_filtered['Date year'] = pd.to_numeric(collection_counts_filtered['Date year'], errors='coerce')
                collection_counts_filtered = collection_counts_filtered.sort_values(by=['Date year'] + selected_collections, ascending=True)

                # Plotting the line graph using Plotly Express
                fig = px.line(collection_counts_filtered, x='Date year', y=selected_collections, 
                            markers=True, line_shape='linear', labels={'value': 'Cumulative Count'},
                            title='Cumulative changes in collection over years')

                # Display the plot in the Streamlit app
                st.plotly_chart(fig, use_container_width=True)

                # PUBLICATION TYPES
                df_types = pd.DataFrame(df_csv['Publication type'].value_counts())
                df_types = df_types.sort_values(['Publication type'], ascending=[False])
                df_types=df_types.reset_index()
                df_types = df_types.rename(columns={'index':'Publication type','Publication type':'Count'})
                # TEMPORARY SOLUTION FOR COLUMN NAME CHANGE ERROR
                df_types.columns = ['Publication type', 'Count']
                # TEMP SOLUTION ENDS


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
                    df_authors = df_csv.copy()
                    df_authors['Author_name'] = df_authors['FirstName2'].apply(lambda x: x.split(', ') if isinstance(x, str) and x else x)
                    df_authors = df_authors.explode('Author_name')
                    df_authors.reset_index(drop=True, inplace=True)
                    max_authors = len(df_authors['Author_name'].unique())
                    num_authors = st.slider('Select number of authors to display:', 5, min(30, max_authors), 20)
                    df_authors['Author_name'] = df_authors['Author_name'].map(name_replacements).fillna(df_authors['Author_name'])
                    df_authors = df_authors['Author_name'].value_counts().head(num_authors)
                    fig = px.bar(df_authors, x=df_authors.index, y=df_authors.values)
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
                    # TEMPORARY SOLUTION FOR COLUMN NAME CHANGE ERROR
                    df_publisher.columns = ['Publisher', 'Count']
                    # TEMP SOLUTION ENDS
                    df_publisher = df_publisher.sort_values(['Count'], ascending=[False])
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
                    # TEMPORARY SOLUTION FOR COLUMN NAME CHANGE ERROR
                    df_journal.columns = ['Journal', 'Count']
                    # TEMP SOLUTION ENDS
                    df_journal = df_journal.sort_values(['Count'], ascending=[False])
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
                df_types = df_types.reset_index()
                df_types.columns = ['Publication type', 'Count']

                st.header('Items in the library by type: ', anchor=False)
                
                df_types = df_types.sort_values(['Count'], ascending=[False])
                plot2= df_types.head(10)

                st.bar_chart(plot2, height=600, width=600, use_container_width=True, x='Publication type')


            st.header('Item inclusion history', anchor=False)
            df_added = df_dedup.copy()
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

    # with tab3: 
    #         st.header('Suggest random sources', anchor=False)
    #         df_intro = df_dedup.copy()
    #         df_intro['Date published'] = pd.to_datetime(df_intro['Date published'],utc=True, errors='coerce').dt.tz_convert('Europe/London')
    #         df_intro['Date published'] = df_intro['Date published'].dt.strftime('%Y-%m-%d')
    #         df_intro['Date published'] = df_intro['Date published'].fillna('')
    #         df_intro['No date flag'] = df_intro['Date published'].isnull().astype(np.uint8)
    #         df_intro = df_intro.sort_values(by=['No date flag', 'Date published'], ascending=[True, True])
    #         df_intro = df_intro.sort_values(by=['Date published'], ascending=False)
    #         df_intro = df_intro.reset_index(drop=True)   
    #         articles_list = [format_entry(row) for _, row in df_intro.iterrows()]
    #         if st.button("Refresh Random 5 Sources"):
    #             # Shuffle the list and select the first 5 elements
    #             random_sources = np.random.choice(articles_list, 5, replace=False)
    #             # Display the selected random sources
    #             for index, formatted_entry in enumerate(random_sources):
    #                 st.write(f"{index + 1}) {formatted_entry}")
    #         else:
    #             # Display the initial 5 sources
    #             for index, formatted_entry in enumerate(articles_list[:5]):
    #                 st.write(f"{index + 1}) {formatted_entry}")
    st.write('---')
    with st.expander('Acknowledgements'):
        st.subheader('Acknowledgements')
        st.write('''
        The following sources are used to collate some of the items and events in this website:
        1. [King's Centre for the Study of Intelligence (KCSI) digest](https://kcsi.uk/kcsi-digests) compiled by David Schaefer
        2. [International Association for Intelligence Education (IAIE) digest](https://www.iafie.org/Login.aspx) compiled by Filip Kovacevic
        ''')
        st.write('''
        Contributors with comments and sources:
        1. Daniela Richterove
        2. Steven Wagner
        3. Sophie Duroy
        ''') 

    display_custom_license()